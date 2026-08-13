"""Tests for train/test splitting (evaluation/split.py)."""

import pytest

from paletteml.evaluation.split import train_test_split_artworks
from tests.factories import make_artwork

LAB_A = (40.0, 55.0, 35.0)


def _artworks(n: int) -> list:
    return [make_artwork(f"a:{i}", [("#ff0000", LAB_A, 1.0)]) for i in range(n)]


class TestSplitSizesAndCoverage:
    def test_sizes_roughly_match_fraction(self):
        train, test = train_test_split_artworks(_artworks(100), test_fraction=0.2, random_state=42)
        assert len(test) == 20
        assert len(train) == 80

    def test_no_overlap(self):
        train, test = train_test_split_artworks(_artworks(50), test_fraction=0.3, random_state=1)
        train_ids = {a.artwork_id for a in train}
        test_ids = {a.artwork_id for a in test}
        assert train_ids.isdisjoint(test_ids)

    def test_full_coverage(self):
        artworks = _artworks(50)
        train, test = train_test_split_artworks(artworks, test_fraction=0.3, random_state=1)
        recovered_ids = {a.artwork_id for a in train} | {a.artwork_id for a in test}
        assert recovered_ids == {a.artwork_id for a in artworks}

    def test_both_sides_nonempty_for_extreme_fractions(self):
        train, test = train_test_split_artworks(_artworks(5), test_fraction=0.01, random_state=1)
        assert len(test) >= 1
        assert len(train) >= 1


class TestDeterminism:
    def test_same_seed_gives_identical_split(self):
        artworks = _artworks(60)
        train1, test1 = train_test_split_artworks(artworks, test_fraction=0.25, random_state=7)
        train2, test2 = train_test_split_artworks(artworks, test_fraction=0.25, random_state=7)
        assert [a.artwork_id for a in train1] == [a.artwork_id for a in train2]
        assert [a.artwork_id for a in test1] == [a.artwork_id for a in test2]

    def test_different_seed_gives_different_split(self):
        artworks = _artworks(60)
        _, test1 = train_test_split_artworks(artworks, test_fraction=0.25, random_state=1)
        _, test2 = train_test_split_artworks(artworks, test_fraction=0.25, random_state=2)
        assert {a.artwork_id for a in test1} != {a.artwork_id for a in test2}


class TestValidation:
    def test_rejects_out_of_range_fraction(self):
        with pytest.raises(ValueError):
            train_test_split_artworks(_artworks(10), test_fraction=1.5)
        with pytest.raises(ValueError):
            train_test_split_artworks(_artworks(10), test_fraction=0.0)

    def test_rejects_too_few_artworks(self):
        with pytest.raises(ValueError):
            train_test_split_artworks(_artworks(1), test_fraction=0.5)
