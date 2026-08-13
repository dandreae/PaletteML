"""Tests for the top-level evaluation orchestration (evaluation/harness.py).

These are the tests that directly check the two things this whole
stage exists to guarantee: no train/test leakage, and an identical
evaluation methodology across all three recommenders.
"""

from unittest.mock import patch

import numpy as np
import pytest

from paletteml.evaluation.harness import run_full_evaluation
from paletteml.modeling.vocabulary import ColorVocabulary
from tests.factories import make_artwork

RANDOM_STATE = 42


def _many_artworks(n: int, seed: int = 0) -> list:
    """n synthetic artworks, each with 2-3 colors drawn from a small
    shared set of Lab points, so there's real co-occurrence structure
    (not just noise) and every color repeats across many paintings.
    """
    rng = np.random.default_rng(seed)
    palette_points = {
        "a": (40.0, 55.0, 35.0),
        "b": (55.0, -40.0, 30.0),
        "c": (30.0, 10.0, -55.0),
        "d": (60.0, 20.0, 45.0),
    }
    artworks = []
    for i in range(n):
        n_colors = rng.integers(2, 4)
        chosen = rng.choice(list(palette_points.keys()), size=n_colors, replace=False)
        colors = [(f"#{c}", palette_points[c], 1.0 / n_colors) for c in chosen]
        artworks.append(make_artwork(f"synthetic:{i}", colors))
    return artworks


class TestNoLeakage:
    def test_vocabulary_is_fit_only_on_train_colors(self):
        train = _many_artworks(20, seed=1)
        test = _many_artworks(10, seed=2)
        n_train_colors = sum(len(a.palette) for a in train)

        with patch.object(ColorVocabulary, "fit", wraps=ColorVocabulary.fit) as spy:
            run_full_evaluation(train, test, vocab_size=4, random_state=RANDOM_STATE)

        assert spy.call_count == 1
        lab_colors_arg = spy.call_args[0][0]
        assert len(lab_colors_arg) == n_train_colors  # never train+test combined

    def test_co_occurrence_n_artworks_equals_train_size_only(self):
        train = _many_artworks(15, seed=3)
        test = _many_artworks(7, seed=4)

        run = run_full_evaluation(train, test, vocab_size=4, random_state=RANDOM_STATE)

        assert run.co_occurrence.n_artworks == len(train)

    def test_test_only_color_does_not_appear_in_vocabulary_fit_input(self):
        # a Lab point that exists ONLY in the test set
        test_only_lab = (95.0, 95.0, 95.0)
        train = _many_artworks(10, seed=5)
        test = [make_artwork("t:special", [("#special", test_only_lab, 1.0)])] + _many_artworks(5, seed=6)

        with patch.object(ColorVocabulary, "fit", wraps=ColorVocabulary.fit) as spy:
            run_full_evaluation(train, test, vocab_size=4, random_state=RANDOM_STATE)

        lab_colors_arg = spy.call_args[0][0]
        assert not any(np.allclose(row, test_only_lab) for row in lab_colors_arg)


class TestFairComparison:
    def test_all_three_recommenders_evaluated_on_identical_case_count(self):
        train = _many_artworks(20, seed=1)
        test = _many_artworks(12, seed=2)

        run = run_full_evaluation(train, test, vocab_size=4, random_state=RANDOM_STATE)

        n_cases = {name: r.n_cases for name, r in run.results.items()}
        assert n_cases["co_occurrence"] == n_cases["popularity"] == n_cases["random"]
        assert n_cases["co_occurrence"] == len(run.case_report.cases)

    def test_all_three_recommenders_see_identical_cases_in_identical_order(self):
        train = _many_artworks(20, seed=1)
        test = _many_artworks(12, seed=2)

        run = run_full_evaluation(train, test, vocab_size=4, random_state=RANDOM_STATE)

        co_cases = [cr.case for cr in run.results["co_occurrence"].case_results]
        pop_cases = [cr.case for cr in run.results["popularity"].case_results]
        rand_cases = [cr.case for cr in run.results["random"].case_results]
        assert co_cases == pop_cases == rand_cases

    def test_reproducible_end_to_end(self):
        train = _many_artworks(20, seed=1)
        test = _many_artworks(12, seed=2)

        run1 = run_full_evaluation(train, test, vocab_size=4, random_state=RANDOM_STATE)
        run2 = run_full_evaluation(train, test, vocab_size=4, random_state=RANDOM_STATE)

        for name in ("co_occurrence", "popularity", "random"):
            assert run1.results[name].hit_rate_at_5 == pytest.approx(run2.results[name].hit_rate_at_5)
            assert run1.results[name].mrr == pytest.approx(run2.results[name].mrr)


class TestReturnedStructure:
    def test_vocab_size_reflected_in_run(self):
        train = _many_artworks(20, seed=1)
        test = _many_artworks(10, seed=2)
        run = run_full_evaluation(train, test, vocab_size=4, random_state=RANDOM_STATE)
        assert run.vocab_size == 4
        assert run.vocabulary.size == 4
