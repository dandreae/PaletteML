"""Tests for ranking metrics and evaluation orchestration (evaluation/metrics.py)."""

import pytest

from paletteml.evaluation.cases import EvalCase
from paletteml.evaluation.metrics import (
    evaluate_recommender,
    hit_rate_at_k,
    mean_reciprocal_rank,
)


class TestHitRateAtK:
    def test_hand_computed_example(self):
        # 4 cases: found at rank 1, rank 3, rank 5, not found
        ranks = [1, 3, 5, None]
        assert hit_rate_at_k(ranks, k=1) == pytest.approx(0.25)  # only rank1
        assert hit_rate_at_k(ranks, k=3) == pytest.approx(0.5)  # rank1, rank3
        assert hit_rate_at_k(ranks, k=5) == pytest.approx(0.75)  # rank1, rank3, rank5

    def test_empty_is_zero_not_error(self):
        assert hit_rate_at_k([], k=5) == 0.0

    def test_all_hits(self):
        assert hit_rate_at_k([1, 1, 1], k=1) == 1.0

    def test_all_misses(self):
        assert hit_rate_at_k([None, None], k=5) == 0.0


class TestMeanReciprocalRank:
    def test_hand_computed_example(self):
        # ranks 1, 2, not found -> (1 + 0.5 + 0) / 3
        ranks = [1, 2, None]
        assert mean_reciprocal_rank(ranks) == pytest.approx((1.0 + 0.5 + 0.0) / 3)

    def test_empty_is_zero_not_error(self):
        assert mean_reciprocal_rank([]) == 0.0

    def test_all_rank_one_gives_mrr_one(self):
        assert mean_reciprocal_rank([1, 1, 1]) == pytest.approx(1.0)


class TestEvaluateRecommender:
    def test_orchestration_matches_hand_computed_metrics(self):
        cases = [
            EvalCase(artwork_id="a", seed_cluster_ids=(0,), hidden_cluster_id=1),
            EvalCase(artwork_id="b", seed_cluster_ids=(0,), hidden_cluster_id=2),
            EvalCase(artwork_id="c", seed_cluster_ids=(0,), hidden_cluster_id=3),
        ]
        # a scripted rank_fn: returns a fixed ranking per call, in the
        # order evaluate_recommender is expected to call it (once per
        # case, in `cases` order), so expected ranks are knowable:
        #   case a: hidden=1 -> found at rank 1
        #   case b: hidden=2 -> found at rank 3
        #   case c: hidden=3 -> not found
        canned_rankings = {
            "a": [1, 9, 9],
            "b": [9, 9, 2],
            "c": [9, 9, 9],
        }
        call_order = iter(["a", "b", "c"])

        def stateful_rank_fn(seed_cluster_ids, top_n):
            return canned_rankings[next(call_order)]

        result = evaluate_recommender("stub", stateful_rank_fn, cases, full_rank_top_n=3)

        assert result.n_cases == 3
        assert [cr.rank for cr in result.case_results] == [1, 3, None]
        assert result.hit_rate_at_1 == pytest.approx(1 / 3)
        assert result.hit_rate_at_3 == pytest.approx(2 / 3)
        assert result.mrr == pytest.approx((1.0 + 1 / 3 + 0.0) / 3)

    def test_empty_case_list(self):
        result = evaluate_recommender("stub", lambda seeds, n: [], [], full_rank_top_n=5)
        assert result.n_cases == 0
        assert result.hit_rate_at_1 == 0.0
        assert result.mrr == 0.0
        assert result.case_results == []

    def test_shorter_ranking_than_full_rank_top_n_is_scored_as_is(self):
        # a recommender that only ever returns 1 candidate (e.g. no
        # positive evidence for more) shouldn't crash or be padded
        case = EvalCase(artwork_id="a", seed_cluster_ids=(0,), hidden_cluster_id=5)
        result = evaluate_recommender("stub", lambda seeds, n: [5], [case], full_rank_top_n=64)
        assert result.case_results[0].rank == 1
        assert result.hit_rate_at_1 == 1.0
