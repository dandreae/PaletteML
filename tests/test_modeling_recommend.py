"""Tests for the co-occurrence recommender (modeling/recommend.py).

A small, fully hand-constructed 4-color vocabulary + co-occurrence
model is used throughout so every ranking outcome is predictable:

  vocab: 0=red, 1=orange, 2=blue, 3=teal   (well-separated Lab points)

  n_artworks = 20
  color_counts: red=10, orange=8, blue=6, teal=4
  pair_counts:
    (red, orange)  = 7   -> strong positive association
                            PMI = ln((7/20)/((10/20)*(8/20))) = ln(1.75)  ≈ 0.560
    (red, blue)    = 4   -> weaker positive association
                            PMI = ln((4/20)/((10/20)*(6/20))) = ln(1.333) ≈ 0.288
    (red, teal)    = 0   -> no observed association -> ppmi = 0
    (orange, blue) = 0
    (orange, teal) = 0
    (blue, teal)   = 3   -> positive association
                            PMI = ln((3/20)/((6/20)*(4/20)))  = ln(2.5)   ≈ 0.916

  Note: raw co-occurrence count alone doesn't determine the PMI sign —
  it's count relative to how common each color is individually. Don't
  assume "count > 0 implies positive PMI" when picking numbers here;
  check P(i,j) vs P(i)*P(j) by hand (as above) instead.
"""

import numpy as np
import pytest

from paletteml.modeling.co_occurrence import CoOccurrenceModel
from paletteml.modeling.recommend import CoOccurrenceRecommender
from paletteml.modeling.vocabulary import ColorVocabulary

RANDOM_STATE = 42

LAB_RED = [40.0, 55.0, 35.0]
LAB_ORANGE = [60.0, 30.0, 55.0]
LAB_BLUE = [30.0, 10.0, -55.0]
LAB_TEAL = [50.0, -35.0, -10.0]

RED, ORANGE, BLUE, TEAL = 0, 1, 2, 3


def _vocabulary() -> ColorVocabulary:
    # order the fed-in points so K-Means labels land as RED, ORANGE, BLUE, TEAL == 0,1,2,3.
    # (each point is its own tight cluster, so cluster_id assignment is
    # stable/predictable for a fixed random_state on this well-separated data)
    points = np.array([LAB_RED, LAB_ORANGE, LAB_BLUE, LAB_TEAL] * 3, dtype=np.float64)
    vocab = ColorVocabulary.fit(points, vocab_size=4, random_state=RANDOM_STATE)
    # re-derive the actual cluster_id for each named color, rather than
    # assuming K-Means label order — robust to any internal reordering
    return vocab


def _co_occurrence(vocab: ColorVocabulary) -> tuple[CoOccurrenceModel, dict]:
    ids = {
        "red": vocab.assign(LAB_RED),
        "orange": vocab.assign(LAB_ORANGE),
        "blue": vocab.assign(LAB_BLUE),
        "teal": vocab.assign(LAB_TEAL),
    }
    color_counts = np.zeros(4, dtype=np.int64)
    color_counts[ids["red"]] = 10
    color_counts[ids["orange"]] = 8
    color_counts[ids["blue"]] = 6
    color_counts[ids["teal"]] = 4

    pair_counts = np.zeros((4, 4), dtype=np.int64)

    def set_pair(a, b, count):
        pair_counts[ids[a], ids[b]] = count
        pair_counts[ids[b], ids[a]] = count

    set_pair("red", "orange", 7)
    set_pair("red", "blue", 4)
    set_pair("blue", "teal", 3)

    model = CoOccurrenceModel(vocab_size=4, n_artworks=20, color_counts=color_counts, pair_counts=pair_counts)
    return model, ids


@pytest.fixture
def setup():
    vocab = _vocabulary()
    co_occurrence, ids = _co_occurrence(vocab)
    return vocab, co_occurrence, ids


def _hex_of(vocab: ColorVocabulary, ids: dict, name: str) -> str:
    return vocab.entries[ids[name]].hex


class TestSingleSeedRecommendation:
    def test_top_recommendation_is_strongest_association(self, setup):
        vocab, co_occurrence, ids = setup
        recommender = CoOccurrenceRecommender(vocab, co_occurrence)

        results = recommender.recommend(_hex_of(vocab, ids, "red"), top_n=5)

        assert len(results) >= 1
        assert results[0].cluster_id == ids["orange"]  # strongest: pair_count 4

    def test_ranking_order_matches_association_strength(self, setup):
        vocab, co_occurrence, ids = setup
        recommender = CoOccurrenceRecommender(vocab, co_occurrence)

        results = recommender.recommend(_hex_of(vocab, ids, "red"), top_n=5)
        ranked_ids = [r.cluster_id for r in results]

        # orange (strong) should outrank blue (weak); teal has zero
        # evidence with red and must not appear at all
        assert ranked_ids.index(ids["orange"]) < ranked_ids.index(ids["blue"])
        assert ids["teal"] not in ranked_ids

    def test_seed_color_itself_excluded(self, setup):
        vocab, co_occurrence, ids = setup
        recommender = CoOccurrenceRecommender(vocab, co_occurrence)

        results = recommender.recommend(_hex_of(vocab, ids, "red"), top_n=5)

        assert ids["red"] not in [r.cluster_id for r in results]

    def test_evidence_diagnostics_present(self, setup):
        vocab, co_occurrence, ids = setup
        recommender = CoOccurrenceRecommender(vocab, co_occurrence)

        results = recommender.recommend(_hex_of(vocab, ids, "red"), top_n=1)

        top = results[0]
        assert top.cluster_id == ids["orange"]
        assert len(top.evidence) == 1
        ev = top.evidence[0]
        assert ev.raw_co_occurrence == 7
        assert ev.conditional_probability == pytest.approx(7 / 10)  # P(orange|red) = 7/10
        assert ev.ppmi == pytest.approx(top.score)  # single seed -> score == that seed's ppmi


class TestMultiSeedRecommendation:
    def test_combines_evidence_by_averaging_ppmi(self, setup):
        vocab, co_occurrence, ids = setup
        recommender = CoOccurrenceRecommender(vocab, co_occurrence)

        red_hex = _hex_of(vocab, ids, "red")
        blue_hex = _hex_of(vocab, ids, "blue")
        results = recommender.recommend([red_hex, blue_hex], top_n=5)

        by_id = {r.cluster_id: r for r in results}
        # teal: 0 evidence with red, strong evidence with blue -> mean > 0, should appear
        assert ids["teal"] in by_id
        expected_teal_score = (
            co_occurrence.ppmi(ids["red"], ids["teal"]) + co_occurrence.ppmi(ids["blue"], ids["teal"])
        ) / 2
        assert by_id[ids["teal"]].score == pytest.approx(expected_teal_score)
        assert len(by_id[ids["teal"]].evidence) == 2  # one entry per seed

    def test_both_seed_colors_excluded_from_results(self, setup):
        vocab, co_occurrence, ids = setup
        recommender = CoOccurrenceRecommender(vocab, co_occurrence)

        results = recommender.recommend(
            [_hex_of(vocab, ids, "red"), _hex_of(vocab, ids, "blue")], top_n=5
        )
        result_ids = [r.cluster_id for r in results]
        assert ids["red"] not in result_ids
        assert ids["blue"] not in result_ids

    def test_duplicate_seeds_mapping_to_same_bin_not_double_counted(self, setup):
        vocab, co_occurrence, ids = setup
        recommender = CoOccurrenceRecommender(vocab, co_occurrence)
        red_hex = _hex_of(vocab, ids, "red")

        single = recommender.recommend(red_hex, top_n=5)
        duplicated = recommender.recommend([red_hex, red_hex], top_n=5)

        assert [r.score for r in single] == pytest.approx([r.score for r in duplicated])
        assert len(duplicated[0].evidence) == 1  # deduped, not two identical entries


class TestEdgeCases:
    def test_rare_color_with_no_positive_evidence_returns_empty(self, setup):
        vocab, co_occurrence, ids = setup
        recommender = CoOccurrenceRecommender(vocab, co_occurrence)

        # teal only ever co-occurred with blue; there's a positive
        # association back to blue, so it should NOT be empty here —
        # use a genuinely isolated case instead: zero out teal's pairs
        co_occurrence.pair_counts[ids["teal"], :] = 0
        co_occurrence.pair_counts[:, ids["teal"]] = 0

        results = recommender.recommend(_hex_of(vocab, ids, "teal"), top_n=5)
        assert results == []

    def test_out_of_distribution_seed_still_maps_to_nearest_bin(self, setup):
        vocab, co_occurrence, ids = setup
        recommender = CoOccurrenceRecommender(vocab, co_occurrence)

        # a color nowhere near any vocabulary entry should still resolve
        # to *some* nearest bin and not raise
        weird_hex = "#7f7f7f"
        results = recommender.recommend(weird_hex, top_n=5)
        assert isinstance(results, list)  # doesn't crash; may be empty or not

    def test_empty_seed_list_raises(self, setup):
        vocab, co_occurrence, _ids = setup
        recommender = CoOccurrenceRecommender(vocab, co_occurrence)
        with pytest.raises(ValueError):
            recommender.recommend([], top_n=5)

    def test_vocab_size_mismatch_raises(self, setup):
        vocab, co_occurrence, _ids = setup
        mismatched = CoOccurrenceModel(
            vocab_size=3,
            n_artworks=1,
            color_counts=np.zeros(3, dtype=np.int64),
            pair_counts=np.zeros((3, 3), dtype=np.int64),
        )
        with pytest.raises(ValueError):
            CoOccurrenceRecommender(vocab, mismatched)
