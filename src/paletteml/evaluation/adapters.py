"""Adapters giving every recommender the same evaluation-time shape:
(seed_cluster_ids: list[int], top_n: int) -> ranked list[int] of candidate cluster_ids.

Each recommender's real public API is shaped for its actual use case —
hex-color input for the co-occurrence/SVD recommenders, no seed
argument at all for the popularity baseline (that contract is
intentional and tested in test_modeling_baseline.py). These adapters
exist only so evaluation/metrics.py's evaluate_recommender() can drive
all four identically, without changing any of those public contracts.
"""

from __future__ import annotations

from paletteml.evaluation.metrics import RankFn
from paletteml.modeling.baseline import PopularityBaseline
from paletteml.modeling.random_baseline import RandomBaseline
from paletteml.modeling.recommend import CoOccurrenceRecommender
from paletteml.modeling.svd_recommend import SvdEmbeddingRecommender


def co_occurrence_rank_fn(recommender: CoOccurrenceRecommender) -> RankFn:
    def rank(seed_cluster_ids: list[int], top_n: int) -> list[int]:
        recs = recommender.recommend_from_cluster_ids(seed_cluster_ids, top_n=top_n)
        return [r.cluster_id for r in recs]

    return rank


def svd_rank_fn(recommender: SvdEmbeddingRecommender) -> RankFn:
    def rank(seed_cluster_ids: list[int], top_n: int) -> list[int]:
        recs = recommender.recommend_from_cluster_ids(seed_cluster_ids, top_n=top_n)
        return [r.cluster_id for r in recs]

    return rank


def popularity_rank_fn(baseline: PopularityBaseline) -> RankFn:
    def rank(seed_cluster_ids: list[int], top_n: int) -> list[int]:
        excluded = set(seed_cluster_ids)
        # ask for the FULL ranking (popularity ignores seeds entirely
        # by design — see baseline.py), then exclude seeds ourselves so
        # a seed color can never "recommend itself back", matching how
        # the co-occurrence recommender always excludes its seeds
        full_ranking = baseline.recommend(top_n=baseline.vocabulary.size)
        filtered = [r.cluster_id for r in full_ranking if r.cluster_id not in excluded]
        return filtered[:top_n]

    return rank


def random_rank_fn(baseline: RandomBaseline) -> RankFn:
    def rank(seed_cluster_ids: list[int], top_n: int) -> list[int]:
        return baseline.recommend(exclude=set(seed_cluster_ids), top_n=top_n)

    return rank
