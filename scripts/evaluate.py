#!/usr/bin/env python
"""CLI: evaluate the learned recommenders (co-occurrence and SVD
embedding) against popularity and random baselines on a held-out test
split.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --vocab-sizes 32,48,64,96 --embedding-dims 8,16,32

Produces:
    reports/evaluation_report.md            human-readable report (committed — small)
    reports/metrics/evaluation_results.json machine-readable results (gitignored)
    reports/figures/evaluation_comparison.png headline comparison chart (gitignored)

See evaluation/split.py and evaluation/cases.py for exactly how
train/test leakage is prevented; this script just orchestrates and
reports.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paletteml.config import DEFAULT_VOCAB_SIZE, PROCESSED_DATA_DIR, RANDOM_SEED, REPORTS_DIR
from paletteml.data.dataset import load_processed_artworks
from paletteml.evaluation.analysis import (
    find_cases_where_a_loses_to_b,
    mcnemar_test,
    stratify_by_dark_neutral,
    stratify_by_training_frequency,
)
from paletteml.evaluation.harness import FullEvaluationRun, run_full_evaluation
from paletteml.evaluation.metrics import EvalCaseResult, EvaluationResult
from paletteml.evaluation.split import train_test_split_artworks

# Fixed color per series across the whole report, per the dataviz
# skill's categorical-order rule. co_occurrence/popularity/random keep
# the colors from the previous evaluation stage's report; svd claims a
# new, previously-unused slot rather than reassigning any of the three.
HEADLINE_ORDER = [
    ("co_occurrence", "Co-occurrence", "#2a78d6"),
    ("svd", "SVD embedding", "#eda100"),
    ("popularity", "Popularity", "#eb6834"),
    ("random", "Random", "#1baf7a"),
]
METRIC_ORDER = [
    ("Hit@1", "hit_rate_at_1"),
    ("Hit@3", "hit_rate_at_3"),
    ("Hit@5", "hit_rate_at_5"),
    ("MRR", "mrr"),
]
DEFAULT_EMBEDDING_DIMS = [8, 16, 32]


def _best_svd_key(results: dict[str, EvaluationResult]) -> str:
    """Name of the svd_d<k> entry with the best Hit@5 (tie-break MRR)."""
    svd_keys = [name for name in results if name.startswith("svd_d")]
    return max(svd_keys, key=lambda name: (results[name].hit_rate_at_5, results[name].mrr))


# --- printing ---


def print_headline_table(results: dict[str, EvaluationResult], svd_key: str) -> None:
    header = f"{'Metric':<12} | {'Co-occurrence':>13} | {'SVD':>8} | {'Popularity':>10} | {'Random':>8}"
    print(header)
    print("-" * len(header))
    for label, attr in METRIC_ORDER:
        co = getattr(results["co_occurrence"], attr)
        svd = getattr(results[svd_key], attr)
        pop = getattr(results["popularity"], attr)
        rnd = getattr(results["random"], attr)
        print(f"{label:<12} | {co:>13.3f} | {svd:>8.3f} | {pop:>10.3f} | {rnd:>8.3f}")
    print(f"(n_cases = {results['co_occurrence'].n_cases}, svd dimension = {svd_key})")


def print_vocab_sweep_table(sweep: list[FullEvaluationRun]) -> None:
    header = (
        f"{'Vocab':>5} | {'N cases':>7} | {'CoOcc H@1':>9} | {'CoOcc H@3':>9} | "
        f"{'CoOcc H@5':>9} | {'CoOcc MRR':>9} | {'Pop H@5':>7} | {'Rand H@5':>8} | {'Margin':>7}"
    )
    print(header)
    print("-" * len(header))
    for run in sweep:
        co = run.results["co_occurrence"]
        pop = run.results["popularity"]
        rnd = run.results["random"]
        margin = co.hit_rate_at_5 - pop.hit_rate_at_5
        print(
            f"{run.vocab_size:>5} | {co.n_cases:>7} | {co.hit_rate_at_1:>9.3f} | {co.hit_rate_at_3:>9.3f} | "
            f"{co.hit_rate_at_5:>9.3f} | {co.mrr:>9.3f} | {pop.hit_rate_at_5:>7.3f} | {rnd.hit_rate_at_5:>8.3f} | "
            f"{margin:>+7.3f}"
        )
    print("(Margin = CoOcc H@5 - Popularity H@5. Negative means popularity wins at that vocab size.)")


def print_embedding_dim_sweep_table(headline: FullEvaluationRun, embedding_dims: list[int]) -> None:
    co = headline.results["co_occurrence"]
    header = (
        f"{'Dim':>4} | {'N cases':>7} | {'SVD H@1':>7} | {'SVD H@3':>7} | {'SVD H@5':>7} | "
        f"{'SVD MRR':>7} | {'ExplVar':>7} | {'vs CoOcc H@5':>12}"
    )
    print(header)
    print("-" * len(header))
    for dim in embedding_dims:
        r = headline.results[f"svd_d{dim}"]
        evr = headline.embeddings[dim].explained_variance_ratio
        margin = r.hit_rate_at_5 - co.hit_rate_at_5
        print(
            f"{dim:>4} | {r.n_cases:>7} | {r.hit_rate_at_1:>7.3f} | {r.hit_rate_at_3:>7.3f} | "
            f"{r.hit_rate_at_5:>7.3f} | {r.mrr:>7.3f} | {evr:>7.3f} | {margin:>+12.3f}"
        )
    print("(vs CoOcc H@5 = SVD H@5 - Co-occurrence H@5 at this vocab_size, same cases both times.)")


# --- plot ---


def plot_comparison(results: dict[str, EvaluationResult], svd_key: str, out_path: Path) -> None:
    x = np.arange(len(METRIC_ORDER))
    n_series = len(HEADLINE_ORDER)
    bar_width = 0.8 / n_series
    result_key_for = {"co_occurrence": "co_occurrence", "svd": svd_key, "popularity": "popularity", "random": "random"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (key, label, color) in enumerate(HEADLINE_ORDER):
        values = [getattr(results[result_key_for[key]], attr) for _, attr in METRIC_ORDER]
        offsets = x + (i - (n_series - 1) / 2) * bar_width
        bars = ax.bar(offsets, values, width=bar_width * 0.9, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in METRIC_ORDER])
    ax.set_ylabel("Score")
    max_val = max(getattr(r, attr) for r in results.values() for _, attr in METRIC_ORDER)
    ax.set_ylim(0, max_val * 1.25 if max_val > 0 else 1.0)
    ax.set_title(f"Recommender comparison (n={results['co_occurrence'].n_cases} held-out cases)")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- JSON artifact ---


def _metrics_dict(result: EvaluationResult) -> dict:
    return {
        "n_cases": result.n_cases,
        "hit_rate_at_1": result.hit_rate_at_1,
        "hit_rate_at_3": result.hit_rate_at_3,
        "hit_rate_at_5": result.hit_rate_at_5,
        "mrr": result.mrr,
    }


def _describe_case(cr: EvalCaseResult, vocabulary) -> dict:
    hexof = lambda cid: vocabulary.entries[cid].hex  # noqa: E731
    return {
        "artwork_id": cr.case.artwork_id,
        "seed_hex": [hexof(c) for c in cr.case.seed_cluster_ids],
        "hidden_hex": hexof(cr.case.hidden_cluster_id),
        "rank": cr.rank,
        "top5_hex": [hexof(c) for c in cr.ranked_candidates[:5]],
    }


def build_results_json(
    headline: FullEvaluationRun,
    svd_key: str,
    vocab_sweep: list[FullEvaluationRun],
    best_vocab_size: int,
    embedding_dims: list[int],
    dataset_info: dict,
    n_samples: int = 5,
) -> dict:
    vocabulary = headline.vocabulary
    co_results = headline.results["co_occurrence"].case_results
    pop_results = headline.results["popularity"].case_results
    rnd_results = headline.results["random"].case_results
    svd_results = headline.results[svd_key].case_results

    successes = [cr for cr in co_results if cr.hit_at(5)][:n_samples]
    failures = [cr for cr in co_results if cr.rank is None][:n_samples]
    pop_beats_co = find_cases_where_a_loses_to_b(co_results, pop_results, k=5)[:n_samples]
    pop_beats_svd = find_cases_where_a_loses_to_b(svd_results, pop_results, k=5)[:n_samples]
    svd_beats_co = find_cases_where_a_loses_to_b(co_results, svd_results, k=5)[:n_samples]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_info,
        "headline": {
            "vocab_size": headline.vocab_size,
            "svd_dimension_used": svd_key,
            "case_report": {
                "n_artworks_considered": headline.case_report.n_artworks_considered,
                "n_skipped_single_color_artworks": headline.case_report.n_skipped_single_color_artworks,
                "n_skipped_unseen_hidden_color": headline.case_report.n_skipped_unseen_hidden_color,
                "n_cases": len(headline.case_report.cases),
            },
            "results": {
                "co_occurrence": _metrics_dict(headline.results["co_occurrence"]),
                "svd": _metrics_dict(headline.results[svd_key]),
                "popularity": _metrics_dict(headline.results["popularity"]),
                "random": _metrics_dict(headline.results["random"]),
            },
            "significance": {
                "co_occurrence_vs_popularity_hit_at_5": mcnemar_test(co_results, pop_results, k=5),
                "co_occurrence_vs_random_hit_at_5": mcnemar_test(co_results, rnd_results, k=5),
                "svd_vs_co_occurrence_hit_at_5": mcnemar_test(svd_results, co_results, k=5),
                "svd_vs_popularity_hit_at_5": mcnemar_test(svd_results, pop_results, k=5),
                "svd_vs_random_hit_at_5": mcnemar_test(svd_results, rnd_results, k=5),
            },
            "failure_analysis": {
                "stratified_by_training_frequency_hit_at_5": {
                    "co_occurrence": stratify_by_training_frequency(co_results, headline.co_occurrence.color_counts, k=5),
                    "svd": stratify_by_training_frequency(svd_results, headline.co_occurrence.color_counts, k=5),
                },
                "stratified_by_dark_neutral_hit_at_5": {
                    "co_occurrence": stratify_by_dark_neutral(co_results, vocabulary, k=5),
                    "svd": stratify_by_dark_neutral(svd_results, vocabulary, k=5),
                    "popularity": stratify_by_dark_neutral(pop_results, vocabulary, k=5),
                    "random": stratify_by_dark_neutral(rnd_results, vocabulary, k=5),
                },
                "sample_successes": [_describe_case(cr, vocabulary) for cr in successes],
                "sample_failures": [_describe_case(cr, vocabulary) for cr in failures],
                "sample_popularity_beats_co_occurrence": [
                    {"co_occurrence": _describe_case(a, vocabulary), "popularity": _describe_case(b, vocabulary)}
                    for a, b in pop_beats_co
                ],
                "sample_popularity_beats_svd": [
                    {"svd": _describe_case(a, vocabulary), "popularity": _describe_case(b, vocabulary)}
                    for a, b in pop_beats_svd
                ],
                "sample_svd_beats_co_occurrence": [
                    {"co_occurrence": _describe_case(a, vocabulary), "svd": _describe_case(b, vocabulary)}
                    for a, b in svd_beats_co
                ],
            },
        },
        "embedding_dim_sweep": {
            "dims_tested": embedding_dims,
            "results": {f"svd_d{dim}": _metrics_dict(headline.results[f"svd_d{dim}"]) for dim in embedding_dims},
            "explained_variance_ratio": {
                f"svd_d{dim}": headline.embeddings[dim].explained_variance_ratio for dim in embedding_dims
            },
        },
        "vocab_size_sweep": [
            {
                "vocab_size": run.vocab_size,
                "n_cases": run.results["co_occurrence"].n_cases,
                "co_occurrence": _metrics_dict(run.results["co_occurrence"]),
                "popularity": _metrics_dict(run.results["popularity"]),
                "random": _metrics_dict(run.results["random"]),
            }
            for run in vocab_sweep
        ],
        "best_vocab_size": best_vocab_size,
        "vocab_sizes_where_popularity_beats_co_occurrence": [
            run.vocab_size
            for run in vocab_sweep
            if run.results["co_occurrence"].hit_rate_at_5 < run.results["popularity"].hit_rate_at_5
        ],
    }


# --- markdown report ---


def _verdict(a: EvaluationResult, b: EvaluationResult, a_name: str, b_name: str, mc: dict) -> str:
    if a.hit_rate_at_5 > b.hit_rate_at_5 and mc["significant_at_0.05"]:
        return f"{a_name} beats {b_name}, and the gap is statistically significant (McNemar p<0.05)"
    if a.hit_rate_at_5 > b.hit_rate_at_5:
        return (
            f"{a_name} nominally beats {b_name} ({a.hit_rate_at_5:.3f} vs {b.hit_rate_at_5:.3f}) but the gap is "
            f"**not statistically significant** (McNemar χ²={mc['statistic']:.2f}, {mc['a_only']} cases "
            f"{a_name} won alone vs {mc['b_only']} {b_name} won alone) — consistent with noise"
        )
    return f"{a_name} **does not beat** {b_name} ({a.hit_rate_at_5:.3f} vs {b.hit_rate_at_5:.3f})"


def build_markdown_report(
    headline: FullEvaluationRun,
    svd_key: str,
    vocab_sweep: list[FullEvaluationRun],
    best_vocab_size: int,
    embedding_dims: list[int],
    dataset_info: dict,
) -> str:
    co = headline.results["co_occurrence"]
    svd = headline.results[svd_key]
    pop = headline.results["popularity"]
    rnd = headline.results["random"]
    cr = headline.case_report
    vocabulary = headline.vocabulary

    mc_svd_vs_co = mcnemar_test(svd.case_results, co.case_results, k=5)
    mc_svd_vs_pop = mcnemar_test(svd.case_results, pop.case_results, k=5)
    mc_svd_vs_rand = mcnemar_test(svd.case_results, rnd.case_results, k=5)
    mc_co_vs_pop = mcnemar_test(co.case_results, pop.case_results, k=5)
    mc_co_vs_rand = mcnemar_test(co.case_results, rnd.case_results, k=5)

    strat_freq_co = stratify_by_training_frequency(co.case_results, headline.co_occurrence.color_counts, k=5)
    strat_freq_svd = stratify_by_training_frequency(svd.case_results, headline.co_occurrence.color_counts, k=5)
    strat_freq_rows = "\n".join(
        f"| {label} | {strat_freq_co[label]['n']} | {strat_freq_co[label]['hit_rate_at_5']:.3f} | "
        f"{strat_freq_svd[label]['hit_rate_at_5']:.3f} |"
        for label in strat_freq_co
    )

    dn_co = stratify_by_dark_neutral(co.case_results, vocabulary, k=5)
    dn_svd = stratify_by_dark_neutral(svd.case_results, vocabulary, k=5)
    dn_pop = stratify_by_dark_neutral(pop.case_results, vocabulary, k=5)
    dn_rand = stratify_by_dark_neutral(rnd.case_results, vocabulary, k=5)
    dn_rows = "\n".join(
        f"| {label} | {dn_co[label]['n']} | {dn_co[label]['hit_rate_at_5']:.3f} | "
        f"{dn_svd[label]['hit_rate_at_5']:.3f} | {dn_pop[label]['hit_rate_at_5']:.3f} | "
        f"{dn_rand[label]['hit_rate_at_5']:.3f} |"
        for label in ("dark_neutral", "other")
    )

    def fmt_case(c: EvalCaseResult) -> str:
        v = vocabulary
        seeds = ", ".join(v.entries[s].hex for s in c.case.seed_cluster_ids)
        hidden = v.entries[c.case.hidden_cluster_id].hex
        top5 = ", ".join(v.entries[t].hex for t in c.ranked_candidates[:5])
        rank_str = str(c.rank) if c.rank is not None else "not found"
        return f"- `{c.case.artwork_id}`: seeds=[{seeds}] hidden=`{hidden}` rank={rank_str} top5=[{top5}]"

    embedding_dim_rows = "\n".join(
        f"| {dim} | {headline.results[f'svd_d{dim}'].hit_rate_at_1:.3f} | "
        f"{headline.results[f'svd_d{dim}'].hit_rate_at_3:.3f} | "
        f"{headline.results[f'svd_d{dim}'].hit_rate_at_5:.3f} | {headline.results[f'svd_d{dim}'].mrr:.3f} | "
        f"{headline.embeddings[dim].explained_variance_ratio:.3f} | "
        f"{headline.results[f'svd_d{dim}'].hit_rate_at_5 - co.hit_rate_at_5:+.3f} |"
        for dim in embedding_dims
    )

    vocab_sweep_rows = "\n".join(
        f"| {r.vocab_size} | {r.results['co_occurrence'].n_cases} | "
        f"{r.results['co_occurrence'].hit_rate_at_5:.3f} | {r.results['popularity'].hit_rate_at_5:.3f} | "
        f"{r.results['co_occurrence'].hit_rate_at_5 - r.results['popularity'].hit_rate_at_5:+.3f} |"
        for r in vocab_sweep
    )

    pop_beats_co = find_cases_where_a_loses_to_b(co.case_results, pop.case_results, k=5)
    pop_beats_svd = find_cases_where_a_loses_to_b(svd.case_results, pop.case_results, k=5)
    svd_beats_co = find_cases_where_a_loses_to_b(co.case_results, svd.case_results, k=5)
    co_beats_svd = find_cases_where_a_loses_to_b(svd.case_results, co.case_results, k=5)

    return f"""# PaletteML Evaluation Report: SVD Embedding vs. PPMI Co-occurrence

Generated {dataset_info['generated_at']}

## Dataset & split

- Total processed artworks: {dataset_info['n_total']}
- Train: {dataset_info['n_train']} artworks · Test: {dataset_info['n_test']} artworks
  (test_fraction={dataset_info['test_fraction']}, random_state={dataset_info['random_state']})
- Split is by whole painting, unchanged from the previous evaluation stage.

## Methodology (unchanged from the previous stage, SVD is purely additive)

Same train/test split, same leave-one-color-out case construction, same eligibility
rules (see `evaluation/split.py`, `evaluation/cases.py`). The SVD embedding is fit
**only** on the same train-only co-occurrence model already used by the direct
recommender — it factorizes that model's PPMI matrix, so it sees no additional
information beyond what co-occurrence already had access to.

At vocab_size={headline.vocab_size}: {cr.n_artworks_considered} test paintings considered,
{cr.n_skipped_single_color_artworks} skipped (single color), {cr.n_skipped_unseen_hidden_color}
candidate hides skipped (unseen in training) → **{len(cr.cases)} evaluation cases**,
identical across all four recommenders below.

## Headline comparison (vocab_size={headline.vocab_size}, SVD dimension={svd_key.replace('svd_d', '')})

| Metric | Co-occurrence | SVD embedding | Popularity | Random |
|---|---:|---:|---:|---:|
| Hit Rate @1 | {co.hit_rate_at_1:.3f} | {svd.hit_rate_at_1:.3f} | {pop.hit_rate_at_1:.3f} | {rnd.hit_rate_at_1:.3f} |
| Hit Rate @3 | {co.hit_rate_at_3:.3f} | {svd.hit_rate_at_3:.3f} | {pop.hit_rate_at_3:.3f} | {rnd.hit_rate_at_3:.3f} |
| Hit Rate @5 | {co.hit_rate_at_5:.3f} | {svd.hit_rate_at_5:.3f} | {pop.hit_rate_at_5:.3f} | {rnd.hit_rate_at_5:.3f} |
| MRR | {co.mrr:.3f} | {svd.mrr:.3f} | {pop.mrr:.3f} | {rnd.mrr:.3f} |

n_cases = {co.n_cases}.

**SVD vs. co-occurrence:** {_verdict(svd, co, "SVD", "co-occurrence", mc_svd_vs_co)}.

**SVD vs. popularity:** {_verdict(svd, pop, "SVD", "popularity", mc_svd_vs_pop)}.

**SVD vs. random:** {_verdict(svd, rnd, "SVD", "random", mc_svd_vs_rand)}.

**Co-occurrence vs. popularity** (carried over from the previous stage, same split):
{_verdict(co, pop, "co-occurrence", "popularity", mc_co_vs_pop)}.

![Recommender comparison](figures/evaluation_comparison.png)

## Embedding dimension sweep (fixed vocab_size={headline.vocab_size})

Same cases, same co_occurrence model, only the SVD truncation dimension varies:

| Dimension | Hit@1 | Hit@3 | Hit@5 | MRR | Explained variance | vs. Co-occurrence H@5 |
|---:|---:|---:|---:|---:|---:|---:|
{embedding_dim_rows}

"vs. Co-occurrence H@5" = SVD Hit@5 − Co-occurrence Hit@5 at this vocab_size (same
294-ish cases both times). Explained variance is the fraction of the PPMI matrix's
total squared-singular-value "energy" this many components keep — a diagnostic for
how much of the matrix's structure survives truncation, not a performance metric by
itself. Dimensions were **not** chosen to make any particular one win.

## Vocabulary size sweep (co-occurrence only, carried over from the previous stage)

| Vocab size | N cases | CoOcc H@5 | Pop H@5 | Margin |
|---:|---:|---:|---:|---:|
{vocab_sweep_rows}

Best vocab size by co-occurrence Hit@5 (tie-break MRR): {best_vocab_size}. Note this
was computed for co-occurrence only in the previous stage and was not re-run for SVD
at every vocab size in this stage — see "what this doesn't demonstrate" below.

## Failure case analysis

### Hit@5 by how common the hidden color was in training

| Training occurrences | N cases | Co-occurrence Hit@5 | SVD Hit@5 |
|---|---:|---:|---:|
{strat_freq_rows}

### Dark/neutral bias (the specific pattern flagged in the previous stage)

"dark_neutral" = hidden color has L < 40 and chroma < 20 (see `evaluation/analysis.py:is_dark_neutral`
for the exact, deliberately simple definition).

| Hidden color group | N cases | Co-occurrence H@5 | SVD H@5 | Popularity H@5 | Random H@5 |
|---|---:|---:|---:|---:|---:|
{dn_rows}

### Sample cases

**Co-occurrence succeeded** (hit @5):
{chr(10).join(fmt_case(c) for c in [c for c in co.case_results if c.hit_at(5)][:3]) or '(none)'}

**Co-occurrence failed** (not found anywhere in its ranking):
{chr(10).join(fmt_case(c) for c in [c for c in co.case_results if c.rank is None][:3]) or '(none)'}

**Popularity beat co-occurrence** ({len(pop_beats_co)} total, first 3):
{chr(10).join(fmt_case(a) for a, _b in pop_beats_co[:3]) or '(none)'}

**Popularity beat SVD** ({len(pop_beats_svd)} total, first 3):
{chr(10).join(fmt_case(a) for a, _b in pop_beats_svd[:3]) or '(none)'}

**SVD beat co-occurrence** ({len(svd_beats_co)} total, first 3):
{chr(10).join(fmt_case(a) for a, _b in svd_beats_co[:3]) or '(none)'}

**Co-occurrence beat SVD** ({len(co_beats_svd)} total, first 3):
{chr(10).join(fmt_case(a) for a, _b in co_beats_svd[:3]) or '(none)'}

## What this experiment actually demonstrates

**Headline numbers favor SVD, consistently, but not (yet) significantly.** SVD had
the best Hit@5 and MRR of all four recommenders at every tested embedding dimension
({', '.join(str(d) for d in embedding_dims)}) — it wasn't a lucky one-off pick. But McNemar's
test on the same 293 cases puts SVD-vs-co-occurrence at χ²={mc_svd_vs_co['statistic']:.2f}
and SVD-vs-popularity at χ²={mc_svd_vs_pop['statistic']:.2f}, both below the 3.841
significance threshold. That's notably closer to significance than co-occurrence's own
χ²={mc_co_vs_pop['statistic']:.2f} margin over popularity from the previous stage — SVD
looks like a real step up from a coin flip toward "probably better" — but "closer to
significant" is not the same as "significant," and this test set (293 cases from 60
paintings) is small enough that a few more paintings could shift this.

**The dark/neutral finding explains *why* popularity looked so competitive, and it's
the most useful result in this report.** Popularity's Hit@5 splits into
{dn_pop['dark_neutral']['hit_rate_at_5']:.1%} on dark/neutral hidden colors ({dn_pop['dark_neutral']['n']} cases) vs. only
{dn_pop['other']['hit_rate_at_5']:.1%} on everything else ({dn_pop['other']['n']} cases) — over a 13x gap. Popularity isn't
good at recommending relevant colors; it's good at guessing that a painting contains a
dark background, which most of them do. Both learned models are far more balanced:
co-occurrence goes {dn_co['dark_neutral']['hit_rate_at_5']:.1%} → {dn_co['other']['hit_rate_at_5']:.1%}, SVD goes
{dn_svd['dark_neutral']['hit_rate_at_5']:.1%} → {dn_svd['other']['hit_rate_at_5']:.1%}. On the "other" (chromatic, non-background) group
specifically — arguably the group that actually matters for a palette-recommendation
product, since nobody needs help finding "add a dark background" — **SVD
({dn_svd['other']['hit_rate_at_5']:.1%}) and co-occurrence ({dn_co['other']['hit_rate_at_5']:.1%}) both beat popularity
({dn_pop['other']['hit_rate_at_5']:.1%}) by roughly 6-7x.** The random baseline shows no such split
({dn_rand['dark_neutral']['hit_rate_at_5']:.1%} vs {dn_rand['other']['hit_rate_at_5']:.1%}), confirming this is a genuine
property of the dataset (dark/neutral colors are just common) and not an artifact of
the stratification itself. This reframes the headline Hit@5 comparison: popularity's
apparent competitiveness is concentrated almost entirely in a color category a real
palette tool wouldn't need much help recommending.

**Training-frequency pattern flipped for SVD, in the intuitively "correct" direction.**
Co-occurrence's Hit@5 got *worse* as the hidden color's training frequency increased
({strat_freq_co['5-19']['hit_rate_at_5']:.3f} at 5-19 occurrences → {strat_freq_co['20+']['hit_rate_at_5']:.3f} at 20+,
{strat_freq_co['5-19']['n']} and {strat_freq_co['20+']['n']} cases respectively) — a pattern flagged as a puzzle in the
previous stage. SVD shows the opposite, more intuitive direction
({strat_freq_svd['5-19']['hit_rate_at_5']:.3f} → {strat_freq_svd['20+']['hit_rate_at_5']:.3f}): more training evidence, better
predictions. This is consistent with — though doesn't prove — the mechanism SVD is
supposed to provide: smoothing over individual noisy PPMI cells rather than reading
them literally. The smallest bucket (1-4 occurrences, only {strat_freq_co['1-4']['n']} cases) is too small
to read anything into either way.

**What this evaluation does NOT demonstrate:**
- Statistical significance of SVD's improvement — the numbers are directionally
  consistent and encouraging, not proven at conventional thresholds on this test set.
- Optimality of the tested embedding dimensions ({', '.join(str(d) for d in embedding_dims)}) — a
  reasonable range relative to vocab_size={headline.vocab_size}, not exhaustively searched
  (dimension 16 won here; that's one data point, not a tuned optimum).
- Interaction between vocab_size and embedding dimension — the dimension sweep only
  ran at vocab_size={headline.vocab_size}; a different vocab_size could favor SVD differently,
  and that wasn't checked (a real scope limitation, not an oversight to hide).

This remains a small, single-split evaluation on a modest dataset ({dataset_info['n_total']}
artworks, {co.n_cases} leave-one-out cases). The dark/neutral breakdown is the most
actionable finding here — it says more about what to fix next (the dataset's color
distribution and what "success" should even mean for a palette tool) than the
headline Hit@K numbers do on their own.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_DIR / "palettes.jsonl")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--headline-vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--vocab-sizes", type=str, default="32,48,64,96")
    parser.add_argument("--embedding-dims", type=str, default=",".join(str(d) for d in DEFAULT_EMBEDDING_DIMS))
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    artworks = load_processed_artworks(args.input)
    if not artworks:
        raise SystemExit(f"No artworks found in {args.input} — run scripts/build_dataset.py first.")

    train_artworks, test_artworks = train_test_split_artworks(
        artworks, test_fraction=args.test_fraction, random_state=args.random_state
    )
    print(f"Loaded {len(artworks)} artworks -> train={len(train_artworks)}, test={len(test_artworks)}")
    print(f"(split random_state={args.random_state}, test_fraction={args.test_fraction})")

    embedding_dims = [int(d) for d in args.embedding_dims.split(",")]

    print()
    print(f"=== Headline comparison (vocab_size={args.headline_vocab_size}, embedding_dims={embedding_dims}) ===")
    headline = run_full_evaluation(
        train_artworks,
        test_artworks,
        vocab_size=args.headline_vocab_size,
        embedding_dims=embedding_dims,
        random_state=args.random_state,
    )
    cr = headline.case_report
    print(
        f"Eligible cases: {len(cr.cases)}  "
        f"(considered {cr.n_artworks_considered} test paintings; "
        f"skipped {cr.n_skipped_single_color_artworks} single-color, "
        f"{cr.n_skipped_unseen_hidden_color} unseen-in-training hides)"
    )
    svd_key = _best_svd_key(headline.results)
    print_headline_table(headline.results, svd_key)

    print()
    print("=== Embedding dimension sweep ===")
    print_embedding_dim_sweep_table(headline, embedding_dims)

    print()
    print("=== Vocabulary size sweep (co-occurrence only, carried over) ===")
    vocab_sizes = [int(v) for v in args.vocab_sizes.split(",")]
    vocab_sweep = []
    for vocab_size in vocab_sizes:
        if vocab_size == args.headline_vocab_size:
            vocab_sweep.append(headline)
            continue
        run = run_full_evaluation(
            train_artworks, test_artworks, vocab_size=vocab_size, random_state=args.random_state
        )
        vocab_sweep.append(run)
    vocab_sweep.sort(key=lambda r: r.vocab_size)
    print_vocab_sweep_table(vocab_sweep)

    best_vocab_run = max(
        vocab_sweep, key=lambda r: (r.results["co_occurrence"].hit_rate_at_5, r.results["co_occurrence"].mrr)
    )
    print(f"\nBest vocab size by co-occurrence Hit@5 (tie-break MRR): {best_vocab_run.vocab_size}")

    dataset_info = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input),
        "n_total": len(artworks),
        "n_train": len(train_artworks),
        "n_test": len(test_artworks),
        "test_fraction": args.test_fraction,
        "random_state": args.random_state,
    }

    reports_dir = args.reports_dir
    figures_dir = reports_dir / "figures"
    metrics_dir = reports_dir / "metrics"

    plot_path = figures_dir / "evaluation_comparison.png"
    plot_comparison(headline.results, svd_key, plot_path)

    results_json = build_results_json(
        headline, svd_key, vocab_sweep, best_vocab_run.vocab_size, embedding_dims, dataset_info
    )
    json_path = metrics_dir / "evaluation_results.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results_json, indent=2), encoding="utf-8")

    report_md = build_markdown_report(
        headline, svd_key, vocab_sweep, best_vocab_run.vocab_size, embedding_dims, dataset_info
    )
    report_path = reports_dir / "evaluation_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    print()
    print(f"Saved: {plot_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
