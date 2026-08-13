"""Tests for co-occurrence statistics (modeling/co_occurrence.py).

Hand-computable synthetic data throughout: 4 paintings over a 3-color
vocabulary {0, 1, 2}, so every expected count/probability/PMI value
below is verifiable by hand.

Paintings:
  A: {0, 1}
  B: {0, 1}
  C: {0, 2}
  D: {1}       (color 1 alone, no pairing)

Expected color_counts: 0 -> 3 (A,B,C), 1 -> 3 (A,B,D), 2 -> 1 (C)
Expected pair_counts:  (0,1) -> 2 (A,B), (0,2) -> 1 (C), (1,2) -> 0
"""

import math

import pytest

from paletteml.modeling.co_occurrence import CoOccurrenceModel

ARTWORKS = [{0, 1}, {0, 1}, {0, 2}, {1}]
VOCAB_SIZE = 3


@pytest.fixture
def model() -> CoOccurrenceModel:
    return CoOccurrenceModel.fit(ARTWORKS, vocab_size=VOCAB_SIZE)


class TestFit:
    def test_color_counts(self, model):
        assert model.color_counts.tolist() == [3, 3, 1]

    def test_pair_counts_symmetric_and_correct(self, model):
        assert model.raw_count(0, 1) == 2
        assert model.raw_count(1, 0) == 2  # symmetric
        assert model.raw_count(0, 2) == 1
        assert model.raw_count(1, 2) == 0

    def test_diagonal_is_zero(self, model):
        for i in range(VOCAB_SIZE):
            assert model.raw_count(i, i) == 0

    def test_n_artworks(self, model):
        assert model.n_artworks == 4


class TestConditionalProbability:
    def test_matches_hand_computed_value(self, model):
        # P(1 | 0) = pair_counts[0,1] / color_counts[0] = 2/3
        assert model.conditional_probability(0, 1) == pytest.approx(2 / 3)
        # P(0 | 1) = 2/3 too here, but not generally symmetric
        assert model.conditional_probability(1, 0) == pytest.approx(2 / 3)
        # P(2 | 0) = 1/3
        assert model.conditional_probability(0, 2) == pytest.approx(1 / 3)

    def test_asymmetric_when_counts_differ(self):
        # color 0 in 4 paintings, color 1 in 2, co-occurring in both of color 1's
        artworks = [{0, 1}, {0, 1}, {0}, {0}]
        m = CoOccurrenceModel.fit(artworks, vocab_size=2)
        assert m.conditional_probability(1, 0) == pytest.approx(1.0)  # P(0|1) = 2/2
        assert m.conditional_probability(0, 1) == pytest.approx(0.5)  # P(1|0) = 2/4

    def test_zero_when_color_never_occurs(self, model):
        never = CoOccurrenceModel.fit(ARTWORKS, vocab_size=4)  # id 3 never appears
        assert never.conditional_probability(3, 0) == 0.0


class TestPmiAndPpmi:
    def test_pmi_matches_hand_computed_value(self, model):
        # P(0)=3/4, P(1)=3/4, P(0,1)=2/4 -> pmi = ln(0.5 / (0.75*0.75))
        expected = math.log(0.5 / (0.75 * 0.75))
        assert model.pmi(0, 1) == pytest.approx(expected)

    def test_pmi_is_negative_infinity_for_unobserved_pair(self, model):
        assert model.pmi(1, 2) == float("-inf")

    def test_ppmi_clips_to_zero(self, model):
        assert model.ppmi(1, 2) == 0.0  # would be -inf as raw pmi (never co-occur)
        # pmi(0, 1) is actually negative here: colors 0 and 1 each occur
        # in 3/4 paintings (P=0.75 each) but only co-occur in 2/4 (0.5),
        # which is *below* the 0.5625 expected under independence — a
        # real below-chance pairing, not a bug. Use (0, 2), which is
        # genuinely positive, to check ppmi passes positive pmi through
        # unchanged.
        assert model.pmi(0, 1) < 0
        assert model.ppmi(0, 1) == 0.0
        assert model.pmi(0, 2) > 0
        assert model.ppmi(0, 2) == pytest.approx(model.pmi(0, 2))

    def test_ppmi_never_negative(self, model):
        for i in range(VOCAB_SIZE):
            for j in range(VOCAB_SIZE):
                if i != j:
                    assert model.ppmi(i, j) >= 0.0


class TestSaveLoad:
    def test_round_trip(self, tmp_path, model):
        path = tmp_path / "co_occurrence.json"
        model.save(path)
        loaded = CoOccurrenceModel.load(path)

        assert loaded.vocab_size == model.vocab_size
        assert loaded.n_artworks == model.n_artworks
        assert loaded.color_counts.tolist() == model.color_counts.tolist()
        assert loaded.pair_counts.tolist() == model.pair_counts.tolist()
        assert loaded.ppmi(0, 1) == pytest.approx(model.ppmi(0, 1))
