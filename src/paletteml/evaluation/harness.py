"""Top-level evaluation orchestration.

`run_full_evaluation` is the single place that: fits a vocabulary +
co-occurrence model on train_artworks ONLY, builds leave-one-out cases
from test_artworks, and evaluates co-occurrence / popularity / random
against the exact same case list — the "fair comparison" the whole
evaluation stage depends on. See evaluation/split.py and
evaluation/cases.py for the leakage-prevention argument in full.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paletteml.config import RANDOM_SEED
from paletteml.data.dataset import LoadedArtwork
from paletteml.evaluation.adapters import co_occurrence_rank_fn, popularity_rank_fn, random_rank_fn
from paletteml.evaluation.cases import EvalCaseBuildReport, build_eval_cases
from paletteml.evaluation.metrics import EvaluationResult, evaluate_recommender
from paletteml.modeling.baseline import PopularityBaseline
from paletteml.modeling.co_occurrence import CoOccurrenceModel
from paletteml.modeling.encoding import encode_palette
from paletteml.modeling.random_baseline import RandomBaseline
from paletteml.modeling.recommend import CoOccurrenceRecommender
from paletteml.modeling.vocabulary import ColorVocabulary


@dataclass
class FullEvaluationRun:
    vocab_size: int
    vocabulary: ColorVocabulary
    co_occurrence: CoOccurrenceModel
    case_report: EvalCaseBuildReport
    results: dict[str, EvaluationResult]  # keyed by "co_occurrence" / "popularity" / "random"


def run_full_evaluation(
    train_artworks: list[LoadedArtwork],
    test_artworks: list[LoadedArtwork],
    vocab_size: int,
    top_n: int = 5,
    random_state: int = RANDOM_SEED,
) -> FullEvaluationRun:
    """Fit on train_artworks only, then evaluate all three recommenders
    identically on cases built from test_artworks.

    `top_n` is accepted for interface symmetry with the recommenders'
    own top_n conventions, but every recommender is actually queried
    with a "full ranking" top_n (the vocabulary size) internally — see
    evaluate_recommender()'s docstring for why: Hit@1/3/5 and MRR are
    all derived from that one full ranking per case.
    """
    train_lab = np.array([c.lab for a in train_artworks for c in a.palette])
    vocabulary = ColorVocabulary.fit(train_lab, vocab_size=vocab_size, random_state=random_state)

    train_artwork_vocab_ids = [set(encode_palette(a.palette, vocabulary).keys()) for a in train_artworks]
    co_occurrence = CoOccurrenceModel.fit(train_artwork_vocab_ids, vocab_size=vocabulary.size)

    case_report = build_eval_cases(test_artworks, vocabulary, co_occurrence)
    cases = case_report.cases

    co_recommender = CoOccurrenceRecommender(vocabulary, co_occurrence)
    popularity = PopularityBaseline(vocabulary, co_occurrence)
    random_baseline = RandomBaseline(vocabulary, random_state=random_state)

    full_rank_top_n = vocabulary.size

    results = {
        "co_occurrence": evaluate_recommender(
            "co_occurrence", co_occurrence_rank_fn(co_recommender), cases, full_rank_top_n
        ),
        "popularity": evaluate_recommender(
            "popularity", popularity_rank_fn(popularity), cases, full_rank_top_n
        ),
        "random": evaluate_recommender("random", random_rank_fn(random_baseline), cases, full_rank_top_n),
    }

    return FullEvaluationRun(
        vocab_size=vocab_size,
        vocabulary=vocabulary,
        co_occurrence=co_occurrence,
        case_report=case_report,
        results=results,
    )
