"""Small shared test-data builders for the evaluation test suite.

Not a conftest.py — these take parameters, so they're plain importable
functions rather than fixtures.
"""

from __future__ import annotations

from paletteml.color.extraction import DominantColor
from paletteml.data.dataset import LoadedArtwork


def make_artwork(
    artwork_id: str, colors: list[tuple[str, tuple[float, float, float], float]]
) -> LoadedArtwork:
    """Build a LoadedArtwork from (hex, lab, proportion) tuples.

    `rgb` is filled with a placeholder — nothing in the evaluation
    pipeline reads DominantColor.rgb, only .lab and .proportion.
    """
    palette = [
        DominantColor(hex=hex_, rgb=(0, 0, 0), lab=lab, proportion=proportion)
        for hex_, lab, proportion in colors
    ]
    return LoadedArtwork(artwork_id=artwork_id, title=artwork_id, palette=palette)
