"""Train/test split of artworks for evaluation.

Splitting happens by *painting* (artwork_id), never by individual
color: every dominant color extracted from one painting stays
together on whichever side of the split that painting lands on.
Splitting by color instead would leak information — colors from the
same painting share brushwork, lighting, and a single artist's
choices, so seeing some of a painting's colors during training while
holding out others "from the same painting" would let a model do well
by learning painting-specific correlations, not genuine
cross-painting color relationships.

This is the only split boundary in the whole evaluation pipeline, and
here is exactly how leakage is prevented from that point on:
  - the color VOCABULARY (modeling/vocabulary.py) is fit only on Lab
    colors pooled from train_artworks — see evaluation/harness.py
  - the CO-OCCURRENCE model (modeling/co_occurrence.py) is fit only on
    train_artworks' palettes, encoded with that train-only vocabulary
  - test_artworks are touched for the first time only at evaluation
    (evaluation/cases.py): their palettes are *encoded* against the
    already-fitted vocabulary. Encoding is a fixed nearest-neighbor
    lookup against centers that were already frozen — no fitting
    happens, so no information from test artworks flows back into the
    vocabulary or the co-occurrence counts.
"""

from __future__ import annotations

import numpy as np

from paletteml.config import RANDOM_SEED
from paletteml.data.dataset import LoadedArtwork


def train_test_split_artworks(
    artworks: list[LoadedArtwork],
    test_fraction: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> tuple[list[LoadedArtwork], list[LoadedArtwork]]:
    """Split artworks into (train, test), by whole painting, deterministically.

    Same `artworks` + `random_state` always produces the same
    partition — every evaluation number downstream depends on this
    being reproducible.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if len(artworks) < 2:
        raise ValueError("need at least 2 artworks to split")

    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(artworks))
    n_test = round(len(artworks) * test_fraction)
    n_test = max(1, min(n_test, len(artworks) - 1))  # keep both sides non-empty

    test_indices = set(order[:n_test].tolist())

    train, test = [], []
    for i, artwork in enumerate(artworks):
        (test if i in test_indices else train).append(artwork)
    return train, test
