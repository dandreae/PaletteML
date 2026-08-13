"""Reproducible random baseline.

Uniformly samples candidate vocabulary colors, excluding the given
seeds. Exists purely as a lower sanity-check bound for evaluation: if
the learned co-occurrence model can't beat this, it isn't
demonstrating it learned anything from real color relationships —
this is what a recommender with zero knowledge looks like.
"""

from __future__ import annotations

from collections.abc import Collection

import numpy as np

from paletteml.config import RANDOM_SEED
from paletteml.modeling.vocabulary import ColorVocabulary


class RandomBaseline:
    """Ranks candidate vocabulary colors uniformly at random."""

    def __init__(self, vocabulary: ColorVocabulary, random_state: int = RANDOM_SEED):
        self.vocabulary = vocabulary
        self._rng = np.random.default_rng(random_state)

    def recommend(self, exclude: Collection[int], top_n: int) -> list[int]:
        """Return up to `top_n` cluster_ids, uniformly at random, excluding `exclude`.

        Consumes from this instance's own random generator, so
        repeated calls on the same RandomBaseline produce a
        deterministic *sequence* of different-looking draws — for an
        evaluation run to be reproducible end-to-end, construct a
        fresh RandomBaseline(random_state=...) at the start of that
        run and call it in the same order every time (which
        evaluation/harness.py does, by iterating a fixed case list).
        """
        excluded = set(exclude)
        candidates = [cid for cid in range(self.vocabulary.size) if cid not in excluded]
        n = min(top_n, len(candidates))
        if n == 0:
            return []
        chosen_indices = self._rng.choice(len(candidates), size=n, replace=False)
        return [candidates[i] for i in chosen_indices]
