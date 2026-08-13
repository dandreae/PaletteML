"""Post-hoc inspection helpers for a completed evaluation run.

Pure reporting/analysis over already-computed EvalCaseResults — no new
model logic lives here, just ways of slicing results that were used to
write the "failure cases" section of the evaluation report.
"""

from __future__ import annotations

import numpy as np

from paletteml.evaluation.metrics import EvalCaseResult
from paletteml.modeling.vocabulary import ColorVocabulary


def stratify_by_training_frequency(
    case_results: list[EvalCaseResult],
    color_counts: np.ndarray,
    k: int = 5,
    bucket_edges: tuple[int, ...] = (1, 5, 20),
) -> dict[str, dict]:
    """Bucket cases by how often the hidden color occurred in training, report Hit@k per bucket.

    Answers: does the recommender do worse specifically on colors it
    saw rarely during training ("rare-color failures")? bucket_edges
    (1, 5, 20) makes buckets [0, 1-4, 5-19, 20+] by default.
    """

    def bucket_index(count: int) -> int:
        for i, edge in enumerate(bucket_edges):
            if count < edge:
                return i
        return len(bucket_edges)

    def bucket_label(i: int) -> str:
        lo = 0 if i == 0 else bucket_edges[i - 1]
        if i == len(bucket_edges):
            return f"{lo}+"
        return f"{lo}-{bucket_edges[i] - 1}"

    buckets: dict[int, list[EvalCaseResult]] = {}
    for cr in case_results:
        count = int(color_counts[cr.case.hidden_cluster_id])
        buckets.setdefault(bucket_index(count), []).append(cr)

    return {
        bucket_label(idx): {
            "n": len(crs),
            f"hit_rate_at_{k}": (sum(1 for cr in crs if cr.hit_at(k)) / len(crs)) if crs else 0.0,
        }
        for idx, crs in sorted(buckets.items())
    }


def is_dark_neutral(
    lab: tuple[float, float, float], lightness_threshold: float = 40.0, chroma_threshold: float = 20.0
) -> bool:
    """Is this Lab color dark (low L) and desaturated (low chroma)?

    Threshold choice: L < 40 covers roughly the bottom third of the
    0-100 lightness range (dark shadows, near-black backgrounds);
    chroma = sqrt(a^2 + b^2) < 20 excludes vivid dark colors (a deep
    saturated red or blue isn't "neutral" just because it's dark).
    Deliberately simple and documented rather than tuned — the point
    is a defensible, inspectable definition, not an optimized one.
    """
    lightness, a, b = lab
    chroma = (a**2 + b**2) ** 0.5
    return lightness < lightness_threshold and chroma < chroma_threshold


def stratify_by_dark_neutral(
    case_results: list[EvalCaseResult],
    vocabulary: ColorVocabulary,
    k: int = 5,
    lightness_threshold: float = 40.0,
    chroma_threshold: float = 20.0,
) -> dict[str, dict]:
    """Split cases by whether the HIDDEN color is dark/neutral, report Hit@k per group.

    Directly targets the dark/neutral bias flagged in the previous
    evaluation stage: popularity's advantage was hypothesized to come
    from dark background/shadow colors being both extremely common
    *and* easy to predict regardless of seed. This checks whether a
    recommender's accuracy is concentrated in that group specifically.
    """
    buckets: dict[str, list[EvalCaseResult]] = {"dark_neutral": [], "other": []}
    for cr in case_results:
        lab = vocabulary.entries[cr.case.hidden_cluster_id].lab
        label = "dark_neutral" if is_dark_neutral(lab, lightness_threshold, chroma_threshold) else "other"
        buckets[label].append(cr)

    return {
        label: {
            "n": len(crs),
            f"hit_rate_at_{k}": (sum(1 for cr in crs if cr.hit_at(k)) / len(crs)) if crs else 0.0,
        }
        for label, crs in buckets.items()
    }


def mcnemar_test(
    a_results: list[EvalCaseResult], b_results: list[EvalCaseResult], k: int = 5
) -> dict:
    """Paired significance test: does a's Hit@k really differ from b's Hit@k?

    a_results and b_results must come from evaluate_recommender() calls
    over the identical cases list (as run_full_evaluation always
    produces) — they're paired observations on the same cases, not
    independent samples, so an ordinary two-proportion test would be
    wrong here. McNemar's test is the standard test for exactly this
    "two classifiers scored on the same items" setup: it only looks at
    cases where the two recommenders *disagree* (one hit, one didn't)
    and asks whether that disagreement is lopsided enough to not be
    noise.

    Returns discordant-pair counts, the (continuity-corrected)
    chi-square statistic, and whether it clears the conventional
    p < 0.05 threshold (chi-square critical value 3.841 at 1 df).
    """
    if len(a_results) != len(b_results):
        raise ValueError("a_results and b_results must come from the same cases list")

    a_only = 0  # a hit, b missed
    b_only = 0  # b hit, a missed
    for a_cr, b_cr in zip(a_results, b_results):
        if a_cr.case != b_cr.case:
            raise ValueError("a_results and b_results are not aligned to the same cases")
        a_hit, b_hit = a_cr.hit_at(k), b_cr.hit_at(k)
        if a_hit and not b_hit:
            a_only += 1
        elif b_hit and not a_hit:
            b_only += 1

    discordant = a_only + b_only
    statistic = ((abs(a_only - b_only) - 1) ** 2 / discordant) if discordant > 0 else 0.0
    return {
        "a_only": a_only,
        "b_only": b_only,
        "statistic": statistic,
        "significant_at_0.05": statistic > 3.841,
    }


def find_cases_where_a_loses_to_b(
    a_results: list[EvalCaseResult],
    b_results: list[EvalCaseResult],
    k: int = 5,
) -> list[tuple[EvalCaseResult, EvalCaseResult]]:
    """Pairs of (a, b) case results for the same underlying case where b hit @k and a didn't.

    Requires a_results and b_results to come from evaluate_recommender()
    calls made over the *identical* cases list, in the same order (as
    run_full_evaluation always does) — they're compared positionally.
    """
    if len(a_results) != len(b_results):
        raise ValueError("a_results and b_results must come from the same cases list")

    losses = []
    for a_cr, b_cr in zip(a_results, b_results):
        if a_cr.case != b_cr.case:
            raise ValueError("a_results and b_results are not aligned to the same cases")
        if (not a_cr.hit_at(k)) and b_cr.hit_at(k):
            losses.append((a_cr, b_cr))
    return losses
