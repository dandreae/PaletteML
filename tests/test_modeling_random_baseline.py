"""Tests for the random baseline (modeling/random_baseline.py)."""

import numpy as np
import pytest

from paletteml.modeling.random_baseline import RandomBaseline
from paletteml.modeling.vocabulary import ColorVocabulary

RANDOM_STATE = 42


@pytest.fixture
def vocabulary() -> ColorVocabulary:
    points = np.random.default_rng(0).normal(size=(30, 3)) * 20 + [50, 0, 0]
    return ColorVocabulary.fit(points, vocab_size=10, random_state=RANDOM_STATE)


class TestRandomBaseline:
    def test_excludes_given_seeds(self, vocabulary):
        baseline = RandomBaseline(vocabulary, random_state=RANDOM_STATE)
        results = baseline.recommend(exclude={0, 1, 2}, top_n=5)
        assert not (set(results) & {0, 1, 2})

    def test_respects_top_n(self, vocabulary):
        baseline = RandomBaseline(vocabulary, random_state=RANDOM_STATE)
        assert len(baseline.recommend(exclude=set(), top_n=4)) == 4

    def test_caps_at_available_candidates(self, vocabulary):
        baseline = RandomBaseline(vocabulary, random_state=RANDOM_STATE)
        # vocab size 10, exclude 8 -> only 2 candidates left even though top_n=5
        exclude = set(range(8))
        results = baseline.recommend(exclude=exclude, top_n=5)
        assert len(results) == 2

    def test_no_duplicates_within_one_call(self, vocabulary):
        baseline = RandomBaseline(vocabulary, random_state=RANDOM_STATE)
        results = baseline.recommend(exclude=set(), top_n=10)
        assert len(results) == len(set(results))

    def test_empty_when_all_excluded(self, vocabulary):
        baseline = RandomBaseline(vocabulary, random_state=RANDOM_STATE)
        results = baseline.recommend(exclude=set(range(vocabulary.size)), top_n=5)
        assert results == []


class TestReproducibility:
    def test_fresh_instances_with_same_seed_produce_identical_sequences(self, vocabulary):
        b1 = RandomBaseline(vocabulary, random_state=RANDOM_STATE)
        b2 = RandomBaseline(vocabulary, random_state=RANDOM_STATE)

        # simulate a sequence of calls, as an evaluation run would make
        seq1 = [b1.recommend(exclude=set(), top_n=3) for _ in range(5)]
        seq2 = [b2.recommend(exclude=set(), top_n=3) for _ in range(5)]

        assert seq1 == seq2

    def test_different_seeds_diverge(self, vocabulary):
        b1 = RandomBaseline(vocabulary, random_state=1)
        b2 = RandomBaseline(vocabulary, random_state=2)
        seq1 = [b1.recommend(exclude=set(), top_n=3) for _ in range(5)]
        seq2 = [b2.recommend(exclude=set(), top_n=3) for _ in range(5)]
        assert seq1 != seq2

    def test_successive_calls_on_same_instance_differ(self, vocabulary):
        # not a hard guarantee for every possible seed, but true for
        # this fixture/seed, and documents the intended "consumes from
        # a shared stream" behavior
        baseline = RandomBaseline(vocabulary, random_state=RANDOM_STATE)
        first = baseline.recommend(exclude=set(), top_n=5)
        second = baseline.recommend(exclude=set(), top_n=5)
        assert first != second
