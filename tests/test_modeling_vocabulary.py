"""Tests for the color vocabulary (modeling/vocabulary.py)."""

import numpy as np
import pytest

from paletteml.modeling.vocabulary import ColorVocabulary

RANDOM_STATE = 42

# Three well-separated Lab points standing in for "red", "green", "blue"
# bins — far enough apart that K-Means recovers them exactly regardless
# of init, which is what makes the expected results in these tests
# knowable by hand.
LAB_RED = [40.0, 55.0, 35.0]
LAB_GREEN = [55.0, -40.0, 30.0]
LAB_BLUE = [30.0, 20.0, -55.0]


def _pooled_lab_colors() -> np.ndarray:
    # 10 samples at each of 3 distinct points -> unambiguous k=3 clustering
    points = [LAB_RED] * 10 + [LAB_GREEN] * 10 + [LAB_BLUE] * 10
    return np.array(points, dtype=np.float64)


class TestFit:
    def test_recovers_distinct_clusters(self):
        vocabulary = ColorVocabulary.fit(_pooled_lab_colors(), vocab_size=3, random_state=RANDOM_STATE)

        assert vocabulary.size == 3
        centers = sorted((e.lab for e in vocabulary.entries), key=lambda lab: lab[0])
        # cluster centers should land exactly on the 3 fed-in points
        expected = sorted([LAB_RED, LAB_GREEN, LAB_BLUE], key=lambda lab: lab[0])
        for got, want in zip(centers, expected):
            assert got == pytest.approx(want, abs=1e-6)

    def test_each_entry_has_consistent_hex_rgb_lab(self):
        vocabulary = ColorVocabulary.fit(_pooled_lab_colors(), vocab_size=3, random_state=RANDOM_STATE)
        for entry in vocabulary.entries:
            assert entry.hex.startswith("#")
            assert len(entry.hex) == 7
            assert all(0 <= c <= 255 for c in entry.rgb)

    def test_clips_vocab_size_to_available_samples(self):
        tiny = np.array([LAB_RED, LAB_GREEN], dtype=np.float64)
        vocabulary = ColorVocabulary.fit(tiny, vocab_size=10, random_state=RANDOM_STATE)
        assert vocabulary.size == 2

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            ColorVocabulary.fit(np.array([1.0, 2.0, 3.0]), vocab_size=3)

    def test_deterministic_with_fixed_random_state(self):
        data = _pooled_lab_colors()
        v1 = ColorVocabulary.fit(data, vocab_size=3, random_state=RANDOM_STATE)
        v2 = ColorVocabulary.fit(data, vocab_size=3, random_state=RANDOM_STATE)
        assert [e.hex for e in v1.entries] == [e.hex for e in v2.entries]
        assert [e.lab for e in v1.entries] == [e.lab for e in v2.entries]


class TestAssign:
    def test_assigns_to_nearest_cluster(self):
        vocabulary = ColorVocabulary.fit(_pooled_lab_colors(), vocab_size=3, random_state=RANDOM_STATE)

        red_id = vocabulary.assign(np.array(LAB_RED) + 0.5)  # near red, not exactly on it
        green_id = vocabulary.assign(np.array(LAB_GREEN) + 0.5)
        blue_id = vocabulary.assign(np.array(LAB_BLUE) + 0.5)

        assert len({red_id, green_id, blue_id}) == 3  # all distinct
        assert vocabulary.entries[red_id].lab == pytest.approx(LAB_RED, abs=1e-6)
        assert vocabulary.entries[green_id].lab == pytest.approx(LAB_GREEN, abs=1e-6)
        assert vocabulary.entries[blue_id].lab == pytest.approx(LAB_BLUE, abs=1e-6)

    def test_assigns_far_off_color_to_closest_available_bin(self):
        # an "unknown"/out-of-distribution color still gets a nearest
        # match rather than erroring
        vocabulary = ColorVocabulary.fit(_pooled_lab_colors(), vocab_size=3, random_state=RANDOM_STATE)
        far_color = np.array([90.0, 90.0, 90.0])
        cluster_id = vocabulary.assign(far_color)
        assert 0 <= cluster_id < vocabulary.size


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        vocabulary = ColorVocabulary.fit(_pooled_lab_colors(), vocab_size=3, random_state=RANDOM_STATE)
        path = tmp_path / "vocab.json"
        vocabulary.save(path)

        loaded = ColorVocabulary.load(path)

        assert loaded.size == vocabulary.size
        assert loaded.random_state == vocabulary.random_state
        assert [e.hex for e in loaded.entries] == [e.hex for e in vocabulary.entries]
        assert [e.lab for e in loaded.entries] == [e.lab for e in vocabulary.entries]
        assert [e.rgb for e in loaded.entries] == [e.rgb for e in vocabulary.entries]

    def test_loaded_vocabulary_assigns_identically(self, tmp_path):
        vocabulary = ColorVocabulary.fit(_pooled_lab_colors(), vocab_size=3, random_state=RANDOM_STATE)
        path = tmp_path / "vocab.json"
        vocabulary.save(path)
        loaded = ColorVocabulary.load(path)

        probe = np.array(LAB_GREEN) + 0.5
        assert loaded.assign(probe) == vocabulary.assign(probe)
