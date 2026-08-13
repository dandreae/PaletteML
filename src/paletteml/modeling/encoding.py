"""Convert one painting's extracted palette into vocabulary-space.

Each extracted dominant color gets mapped to its nearest vocabulary
bin. A painting's palette often has more than one extracted color
land on the same bin (e.g. two similar shades of red both being the
nearest match to the same vocabulary "red") — those are merged, not
double-counted, so a bin's presence in a painting is unambiguous
when co-occurrence is counted downstream.
"""

from __future__ import annotations

from collections.abc import Sequence

from paletteml.color.extraction import DominantColor
from paletteml.modeling.vocabulary import ColorVocabulary


def encode_palette(
    colors: Sequence[DominantColor], vocabulary: ColorVocabulary
) -> dict[int, float]:
    """Map a painting's dominant colors onto vocabulary cluster_ids.

    Returns a dict of {cluster_id: weight}, where weight is the sum
    of proportions of every extracted color that mapped to that
    cluster_id (so a bin hit by two extracted colors gets their
    combined proportion, not two separate entries). Weights are not
    renormalized — they still sum to (approximately) the painting's
    total palette proportion, i.e. ~1.0.
    """
    weights: dict[int, float] = {}
    for color in colors:
        cluster_id = vocabulary.assign(color.lab)
        weights[cluster_id] = weights.get(cluster_id, 0.0) + color.proportion
    return weights
