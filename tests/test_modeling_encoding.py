"""Tests for palette -> vocabulary encoding (modeling/encoding.py)."""

import numpy as np
import pytest

from paletteml.color.extraction import DominantColor
from paletteml.modeling.encoding import encode_palette
from paletteml.modeling.vocabulary import ColorVocabulary

RANDOM_STATE = 42
LAB_A = [40.0, 55.0, 35.0]
LAB_B = [55.0, -40.0, 30.0]


def _two_color_vocabulary() -> ColorVocabulary:
    data = np.array([LAB_A] * 5 + [LAB_B] * 5, dtype=np.float64)
    return ColorVocabulary.fit(data, vocab_size=2, random_state=RANDOM_STATE)


class TestEncodePalette:
    def test_merges_colors_mapped_to_the_same_bin(self):
        vocabulary = _two_color_vocabulary()
        # two distinct extracted colors, both near LAB_A -> same vocab bin
        colors = [
            DominantColor(hex="#aaaaaa", rgb=(1, 1, 1), lab=tuple(np.array(LAB_A) + 0.1), proportion=0.3),
            DominantColor(hex="#bbbbbb", rgb=(2, 2, 2), lab=tuple(np.array(LAB_A) - 0.1), proportion=0.2),
        ]

        weights = encode_palette(colors, vocabulary)

        assert len(weights) == 1  # merged into a single bin, not two
        ((cluster_id, weight),) = weights.items()
        assert weight == pytest.approx(0.5)  # 0.3 + 0.2, summed not overwritten

    def test_keeps_distinct_bins_separate(self):
        vocabulary = _two_color_vocabulary()
        colors = [
            DominantColor(hex="#aaaaaa", rgb=(1, 1, 1), lab=tuple(LAB_A), proportion=0.6),
            DominantColor(hex="#bbbbbb", rgb=(2, 2, 2), lab=tuple(LAB_B), proportion=0.4),
        ]

        weights = encode_palette(colors, vocabulary)

        assert len(weights) == 2
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_empty_palette_returns_empty_dict(self):
        vocabulary = _two_color_vocabulary()
        assert encode_palette([], vocabulary) == {}
