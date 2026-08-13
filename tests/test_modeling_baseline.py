"""Tests for the popularity baseline (modeling/baseline.py)."""

import numpy as np
import pytest

from paletteml.modeling.baseline import PopularityBaseline
from paletteml.modeling.co_occurrence import CoOccurrenceModel
from paletteml.modeling.vocabulary import ColorVocabulary

RANDOM_STATE = 42
LAB_A = [40.0, 55.0, 35.0]
LAB_B = [55.0, -40.0, 30.0]
LAB_C = [30.0, 10.0, -55.0]


@pytest.fixture
def vocabulary() -> ColorVocabulary:
    points = np.array([LAB_A, LAB_B, LAB_C] * 3, dtype=np.float64)
    return ColorVocabulary.fit(points, vocab_size=3, random_state=RANDOM_STATE)


def _model_with_counts(vocabulary: ColorVocabulary, counts: dict[str, int]) -> tuple[CoOccurrenceModel, dict]:
    ids = {"a": vocabulary.assign(LAB_A), "b": vocabulary.assign(LAB_B), "c": vocabulary.assign(LAB_C)}
    color_counts = np.zeros(3, dtype=np.int64)
    for name, count in counts.items():
        color_counts[ids[name]] = count
    model = CoOccurrenceModel(
        vocab_size=3, n_artworks=10, color_counts=color_counts, pair_counts=np.zeros((3, 3), dtype=np.int64)
    )
    return model, ids


class TestPopularityBaseline:
    def test_ranks_by_descending_popularity(self, vocabulary):
        model, ids = _model_with_counts(vocabulary, {"a": 2, "b": 9, "c": 5})
        baseline = PopularityBaseline(vocabulary, model)

        results = baseline.recommend(top_n=3)

        assert [r.cluster_id for r in results] == [ids["b"], ids["c"], ids["a"]]

    def test_score_is_fraction_of_artworks(self, vocabulary):
        model, ids = _model_with_counts(vocabulary, {"a": 2, "b": 9, "c": 5})
        baseline = PopularityBaseline(vocabulary, model)

        results = baseline.recommend(top_n=1)

        assert results[0].score == pytest.approx(0.9)  # 9 / 10 artworks
        assert results[0].supporting_artworks == 9

    def test_respects_top_n(self, vocabulary):
        model, _ids = _model_with_counts(vocabulary, {"a": 2, "b": 9, "c": 5})
        baseline = PopularityBaseline(vocabulary, model)
        assert len(baseline.recommend(top_n=2)) == 2

    def test_ignores_any_notion_of_seed_color(self, vocabulary):
        # PopularityBaseline.recommend() takes no seed argument at all —
        # this test just documents/locks that contract.
        model, _ids = _model_with_counts(vocabulary, {"a": 2, "b": 9, "c": 5})
        baseline = PopularityBaseline(vocabulary, model)
        assert "seed" not in baseline.recommend.__code__.co_varnames

    def test_vocab_size_mismatch_raises(self, vocabulary):
        mismatched = CoOccurrenceModel(
            vocab_size=2, n_artworks=1, color_counts=np.zeros(2, dtype=np.int64), pair_counts=np.zeros((2, 2), dtype=np.int64)
        )
        with pytest.raises(ValueError):
            PopularityBaseline(vocabulary, mismatched)
