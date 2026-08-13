"""Tests for evaluation/analysis.py (post-hoc failure-case inspection)."""

import numpy as np
import pytest

from paletteml.evaluation.analysis import (
    find_cases_where_a_loses_to_b,
    is_dark_neutral,
    mcnemar_test,
    stratify_by_dark_neutral,
    stratify_by_training_frequency,
)
from paletteml.evaluation.cases import EvalCase
from paletteml.evaluation.metrics import EvalCaseResult
from paletteml.modeling.vocabulary import ColorVocabulary


def _result(hidden_id: int, rank: int | None, artwork_id: str = "a") -> EvalCaseResult:
    case = EvalCase(artwork_id=artwork_id, seed_cluster_ids=(0,), hidden_cluster_id=hidden_id)
    ranked = list(range(1, 10)) if rank is None else [99] * (rank - 1) + [hidden_id]
    reciprocal_rank = (1.0 / rank) if rank is not None else 0.0
    return EvalCaseResult(case=case, ranked_candidates=ranked, rank=rank, reciprocal_rank=reciprocal_rank)


class TestStratifyByTrainingFrequency:
    def test_buckets_and_hit_rates(self):
        # hidden colors: id0 (count=0, rare), id1 (count=3), id2 (count=50, common)
        color_counts = np.array([0, 3, 50], dtype=np.int64)
        results = [
            _result(hidden_id=0, rank=None),  # rare, miss
            _result(hidden_id=1, rank=1),  # count=3 bucket, hit@5
            _result(hidden_id=1, rank=None),  # count=3 bucket, miss
            _result(hidden_id=2, rank=2),  # common, hit@5
        ]

        stratified = stratify_by_training_frequency(results, color_counts, k=5, bucket_edges=(1, 5, 20))

        assert stratified["0-0"]["n"] == 1
        assert stratified["0-0"]["hit_rate_at_5"] == 0.0
        assert stratified["1-4"]["n"] == 2
        assert stratified["1-4"]["hit_rate_at_5"] == pytest.approx(0.5)
        assert stratified["20+"]["n"] == 1
        assert stratified["20+"]["hit_rate_at_5"] == pytest.approx(1.0)

    def test_empty_input(self):
        assert stratify_by_training_frequency([], np.array([0]), k=5) == {}


class TestIsDarkNeutral:
    def test_dark_desaturated_is_dark_neutral(self):
        assert is_dark_neutral((15.0, 2.0, -3.0)) is True  # near-black shadow

    def test_dark_but_saturated_is_not_dark_neutral(self):
        assert is_dark_neutral((25.0, 60.0, 40.0)) is False  # deep saturated red

    def test_light_desaturated_is_not_dark_neutral(self):
        assert is_dark_neutral((80.0, 1.0, -1.0)) is False  # near-white

    def test_thresholds_are_configurable(self):
        lab = (35.0, 15.0, 5.0)
        assert is_dark_neutral(lab, lightness_threshold=30.0) is False  # tightened threshold excludes it
        assert is_dark_neutral(lab, lightness_threshold=40.0) is True


class TestStratifyByDarkNeutral:
    def test_buckets_by_hidden_color_lab(self):
        points = np.array([[15.0, 2.0, -3.0], [80.0, 60.0, 40.0]] * 3, dtype=np.float64)
        vocabulary = ColorVocabulary.fit(points, vocab_size=2, random_state=42)
        dark_id = vocabulary.assign(np.array([15.0, 2.0, -3.0]))
        vivid_id = vocabulary.assign(np.array([80.0, 60.0, 40.0]))

        results = [
            _result(hidden_id=dark_id, rank=1, artwork_id="a"),  # dark, hit
            _result(hidden_id=dark_id, rank=None, artwork_id="b"),  # dark, miss
            _result(hidden_id=vivid_id, rank=2, artwork_id="c"),  # other, hit
        ]

        stratified = stratify_by_dark_neutral(results, vocabulary, k=5)

        assert stratified["dark_neutral"]["n"] == 2
        assert stratified["dark_neutral"]["hit_rate_at_5"] == pytest.approx(0.5)
        assert stratified["other"]["n"] == 1
        assert stratified["other"]["hit_rate_at_5"] == pytest.approx(1.0)

    def test_empty_bucket_is_zero_not_error(self):
        points = np.array([[15.0, 2.0, -3.0]] * 4, dtype=np.float64)
        vocabulary = ColorVocabulary.fit(points, vocab_size=1, random_state=42)
        results = [_result(hidden_id=0, rank=1, artwork_id="a")]
        stratified = stratify_by_dark_neutral(results, vocabulary, k=5)
        assert stratified["other"]["n"] == 0
        assert stratified["other"]["hit_rate_at_5"] == 0.0


class TestFindCasesWhereALosesToB:
    def test_finds_exactly_the_losing_cases(self):
        a_results = [_result(1, rank=None, artwork_id="x"), _result(2, rank=1, artwork_id="y")]
        b_results = [_result(1, rank=2, artwork_id="x"), _result(2, rank=1, artwork_id="y")]

        losses = find_cases_where_a_loses_to_b(a_results, b_results, k=5)

        assert len(losses) == 1
        assert losses[0][0].case.artwork_id == "x"

    def test_no_losses_when_a_always_wins_or_ties(self):
        a_results = [_result(1, rank=1, artwork_id="x")]
        b_results = [_result(1, rank=2, artwork_id="x")]
        assert find_cases_where_a_loses_to_b(a_results, b_results, k=5) == []

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            find_cases_where_a_loses_to_b([_result(1, rank=1)], [], k=5)

    def test_misaligned_cases_raise(self):
        a_results = [_result(1, rank=1, artwork_id="x")]
        b_results = [_result(1, rank=1, artwork_id="different")]
        with pytest.raises(ValueError):
            find_cases_where_a_loses_to_b(a_results, b_results, k=5)


class TestMcNemarTest:
    def test_identical_performance_gives_zero_statistic(self):
        a_results = [_result(1, rank=1, artwork_id="x"), _result(2, rank=None, artwork_id="y")]
        b_results = [_result(1, rank=2, artwork_id="x"), _result(2, rank=None, artwork_id="y")]
        # both hit@5 on x, both miss on y -> no discordant pairs
        result = mcnemar_test(a_results, b_results, k=5)
        assert result["a_only"] == 0
        assert result["b_only"] == 0
        assert result["statistic"] == 0.0
        assert result["significant_at_0.05"] is False

    def test_counts_discordant_pairs_correctly(self):
        a_results = [
            _result(1, rank=1, artwork_id="x"),  # a hits, b misses -> a_only
            _result(2, rank=None, artwork_id="y"),  # a misses, b hits -> b_only
            _result(3, rank=1, artwork_id="z"),  # both hit -> concordant
        ]
        b_results = [
            _result(1, rank=None, artwork_id="x"),
            _result(2, rank=1, artwork_id="y"),
            _result(3, rank=2, artwork_id="z"),
        ]
        result = mcnemar_test(a_results, b_results, k=5)
        assert result["a_only"] == 1
        assert result["b_only"] == 1

    def test_large_lopsided_difference_is_significant(self):
        # 20 cases where a hits and b misses, 0 the other way
        a_results = [_result(i, rank=1, artwork_id=str(i)) for i in range(20)]
        b_results = [_result(i, rank=None, artwork_id=str(i)) for i in range(20)]
        result = mcnemar_test(a_results, b_results, k=5)
        assert result["a_only"] == 20
        assert result["b_only"] == 0
        assert result["significant_at_0.05"] is True

    def test_small_difference_is_not_significant(self):
        # a single discordant pair is nowhere near significant
        a_results = [_result(1, rank=1, artwork_id="x"), _result(2, rank=None, artwork_id="y")]
        b_results = [_result(1, rank=None, artwork_id="x"), _result(2, rank=None, artwork_id="y")]
        result = mcnemar_test(a_results, b_results, k=5)
        assert result["significant_at_0.05"] is False

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            mcnemar_test([_result(1, rank=1)], [], k=5)

    def test_misaligned_cases_raise(self):
        a_results = [_result(1, rank=1, artwork_id="x")]
        b_results = [_result(1, rank=1, artwork_id="different")]
        with pytest.raises(ValueError):
            mcnemar_test(a_results, b_results, k=5)
