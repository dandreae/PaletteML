"""Tests for leave-one-out evaluation case construction (evaluation/cases.py)."""

import numpy as np
import pytest

from paletteml.modeling.co_occurrence import CoOccurrenceModel
from paletteml.modeling.vocabulary import ColorVocabulary
from paletteml.evaluation.cases import build_eval_cases
from tests.factories import make_artwork

RANDOM_STATE = 42
LAB_A = (40.0, 55.0, 35.0)
LAB_B = (55.0, -40.0, 30.0)
LAB_C = (30.0, 10.0, -55.0)


@pytest.fixture
def vocabulary() -> ColorVocabulary:
    points = np.array([LAB_A, LAB_B, LAB_C] * 3, dtype=np.float64)
    return ColorVocabulary.fit(points, vocab_size=3, random_state=RANDOM_STATE)


def _ids(vocabulary):
    return {
        "a": vocabulary.assign(np.array(LAB_A)),
        "b": vocabulary.assign(np.array(LAB_B)),
        "c": vocabulary.assign(np.array(LAB_C)),
    }


class TestBuildEvalCases:
    def test_one_case_per_distinct_color_in_eligible_artwork(self, vocabulary):
        ids = _ids(vocabulary)
        co_occurrence = CoOccurrenceModel(
            vocab_size=3, n_artworks=5,
            color_counts=np.array([5, 5, 5], dtype=np.int64),
            pair_counts=np.zeros((3, 3), dtype=np.int64),
        )
        artwork = make_artwork("t:1", [("#a", LAB_A, 0.5), ("#b", LAB_B, 0.3), ("#c", LAB_C, 0.2)])

        report = build_eval_cases([artwork], vocabulary, co_occurrence)

        assert len(report.cases) == 3  # one hide per distinct color
        hidden_ids = {c.hidden_cluster_id for c in report.cases}
        assert hidden_ids == {ids["a"], ids["b"], ids["c"]}
        for case in report.cases:
            assert case.hidden_cluster_id not in case.seed_cluster_ids
            assert len(case.seed_cluster_ids) == 2  # the other two colors

    def test_skips_single_color_artwork(self, vocabulary):
        co_occurrence = CoOccurrenceModel(
            vocab_size=3, n_artworks=5,
            color_counts=np.array([5, 5, 5], dtype=np.int64),
            pair_counts=np.zeros((3, 3), dtype=np.int64),
        )
        # two extracted colors that both map to the SAME vocab bin ->
        # only 1 distinct vocabulary color after encoding
        artwork = make_artwork(
            "t:1",
            [("#a1", tuple(np.array(LAB_A) + 0.01), 0.6), ("#a2", tuple(np.array(LAB_A) - 0.01), 0.4)],
        )

        report = build_eval_cases([artwork], vocabulary, co_occurrence)

        assert report.cases == []
        assert report.n_skipped_single_color_artworks == 1
        assert report.n_skipped_unseen_hidden_color == 0

    def test_skips_hidden_color_unseen_in_training(self, vocabulary):
        ids = _ids(vocabulary)
        color_counts = np.array([5, 5, 5], dtype=np.int64)
        color_counts[ids["c"]] = 0  # color c never seen in training
        co_occurrence = CoOccurrenceModel(
            vocab_size=3, n_artworks=5, color_counts=color_counts, pair_counts=np.zeros((3, 3), dtype=np.int64)
        )
        artwork = make_artwork("t:1", [("#a", LAB_A, 0.5), ("#b", LAB_B, 0.3), ("#c", LAB_C, 0.2)])

        report = build_eval_cases([artwork], vocabulary, co_occurrence)

        # a and b are still eligible hidden targets; c is skipped
        assert len(report.cases) == 2
        assert report.n_skipped_unseen_hidden_color == 1
        hidden_ids = {c.hidden_cluster_id for c in report.cases}
        assert ids["c"] not in hidden_ids

    def test_report_counts_are_consistent(self, vocabulary):
        co_occurrence = CoOccurrenceModel(
            vocab_size=3, n_artworks=5,
            color_counts=np.array([5, 5, 5], dtype=np.int64),
            pair_counts=np.zeros((3, 3), dtype=np.int64),
        )
        eligible = make_artwork("t:1", [("#a", LAB_A, 0.5), ("#b", LAB_B, 0.5)])
        single = make_artwork(
            "t:2", [("#a1", tuple(np.array(LAB_A) + 0.01), 0.6), ("#a2", tuple(np.array(LAB_A) - 0.01), 0.4)]
        )

        report = build_eval_cases([eligible, single], vocabulary, co_occurrence)

        assert report.n_artworks_considered == 2
        assert report.n_skipped_single_color_artworks == 1
        assert len(report.cases) == 2  # both colors of the eligible artwork
