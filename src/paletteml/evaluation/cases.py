"""Leave-one-color-out evaluation case construction.

For every eligible test-artwork color, one case holds that color out
and uses the artwork's *other* vocabulary colors as seeds. Cases are
built entirely from already-fitted (train-only) vocabulary +
co-occurrence artifacts plus test artworks that were never used to
fit either — see evaluation/split.py for the full leakage argument.

Design choice worth stating explicitly: a test artwork with N distinct
vocabulary colors contributes up to N cases (one per color taking a
turn as the held-out target), not just one. This is the standard
reading of "leave-one-out" — every element gets held out once — and
it matters in practice: with a modest test set (tens of paintings),
one case per painting would leave Hit@K/MRR too noisy to compare
configurations meaningfully. All N cases from one painting still share
that painting's other colors, so they aren't independent samples in a
strict statistical sense — a caveat worth keeping in mind when reading
the metrics, not a flaw in the case construction itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from paletteml.data.dataset import LoadedArtwork
from paletteml.modeling.co_occurrence import CoOccurrenceModel
from paletteml.modeling.encoding import encode_palette
from paletteml.modeling.vocabulary import ColorVocabulary


@dataclass(frozen=True)
class EvalCase:
    """One leave-one-out evaluation case: predict hidden_cluster_id from seed_cluster_ids."""

    artwork_id: str
    seed_cluster_ids: tuple[int, ...]
    hidden_cluster_id: int


@dataclass
class EvalCaseBuildReport:
    """Cases built, plus exactly how many candidates were skipped and why."""

    cases: list[EvalCase] = field(default_factory=list)
    n_artworks_considered: int = 0
    n_skipped_single_color_artworks: int = 0
    n_skipped_unseen_hidden_color: int = 0


def build_eval_cases(
    test_artworks: list[LoadedArtwork],
    vocabulary: ColorVocabulary,
    co_occurrence: CoOccurrenceModel,
) -> EvalCaseBuildReport:
    """Build leave-one-out cases from test_artworks against a train-fitted model.

    A candidate hide is skipped, and not turned into a case, in two
    situations (both counted separately in the returned report):

      1. The artwork has fewer than 2 distinct vocabulary colors after
         encoding — with only one color, hiding it leaves zero seeds,
         so there's nothing for any recommender to condition on. The
         whole artwork is skipped (not just one color).

      2. The candidate hidden color's vocabulary bin was never
         observed in ANY training painting (color_counts == 0). No
         recommender fit only on the training set could possibly have
         learned anything about this bin, by construction — testing
         whether it gets "recommended" would measure a gap in
         vocabulary coverage between train and test, not
         recommendation quality. This affects only that one candidate
         color, not the rest of the artwork's cases.
    """
    report = EvalCaseBuildReport(n_artworks_considered=len(test_artworks))

    for artwork in test_artworks:
        weights = encode_palette(artwork.palette, vocabulary)
        distinct_ids = sorted(weights.keys())

        if len(distinct_ids) < 2:
            report.n_skipped_single_color_artworks += 1
            continue

        for hidden_id in distinct_ids:
            if co_occurrence.color_counts[hidden_id] == 0:
                report.n_skipped_unseen_hidden_color += 1
                continue
            seeds = tuple(cid for cid in distinct_ids if cid != hidden_id)
            report.cases.append(
                EvalCase(
                    artwork_id=artwork.artwork_id,
                    seed_cluster_ids=seeds,
                    hidden_cluster_id=hidden_id,
                )
            )

    return report
