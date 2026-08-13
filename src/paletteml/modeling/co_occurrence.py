"""Interpretable color co-occurrence statistics.

Counts, for each pair of vocabulary colors, how many training
paintings contain both — the "straightforward co-occurrence matrix"
this project's first recommender is built on. Deliberately not SVD /
an embedding: every number here traces back to a raw count you can
recompute by hand, which is the point at this stage.

Presence is binary per painting: a vocabulary color either appears in
a painting's encoded palette (see encoding.py, which already merges
duplicate bin hits) or it doesn't. Co-occurrence counts "did colors i
and j both appear somewhere in this painting", not how much area they
covered — proportions are available upstream in encode_palette() if a
future version wants a weighted variant, but starting unweighted
keeps the statistics easiest to sanity-check.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CoOccurrenceModel:
    """Pairwise co-occurrence counts over a fixed-size color vocabulary."""

    vocab_size: int
    n_artworks: int
    color_counts: np.ndarray  # shape (V,) — # artworks containing this color
    pair_counts: np.ndarray  # shape (V, V), symmetric, zero diagonal

    @classmethod
    def fit(cls, artwork_vocab_ids: list[set[int]], vocab_size: int) -> CoOccurrenceModel:
        """Build co-occurrence stats from each painting's set of vocab ids.

        Each element of `artwork_vocab_ids` is the *distinct* set of
        vocabulary cluster_ids present in one painting (see
        encoding.py) — using a set here is what prevents a painting
        with two colors mapped to the same bin from inflating that
        bin's counts.
        """
        color_counts = np.zeros(vocab_size, dtype=np.int64)
        pair_counts = np.zeros((vocab_size, vocab_size), dtype=np.int64)

        for ids in artwork_vocab_ids:
            ids = sorted(ids)
            for cid in ids:
                color_counts[cid] += 1
            for a_idx in range(len(ids)):
                for b_idx in range(a_idx + 1, len(ids)):
                    a, b = ids[a_idx], ids[b_idx]
                    pair_counts[a, b] += 1
                    pair_counts[b, a] += 1

        return cls(
            vocab_size=vocab_size,
            n_artworks=len(artwork_vocab_ids),
            color_counts=color_counts,
            pair_counts=pair_counts,
        )

    # --- pairwise statistics ---

    def raw_count(self, i: int, j: int) -> int:
        """Number of training paintings containing both color i and color j."""
        return int(self.pair_counts[i, j])

    def conditional_probability(self, i: int, j: int) -> float:
        """P(j present | i present) = pair_counts[i,j] / color_counts[i]."""
        if self.color_counts[i] == 0:
            return 0.0
        return float(self.pair_counts[i, j] / self.color_counts[i])

    def pmi(self, i: int, j: int) -> float:
        """Pointwise mutual information between colors i and j.

        log( P(i,j) / (P(i) * P(j)) ), estimated from counts over
        n_artworks. Returns -inf if i and j were never observed
        together (undefined/no evidence) or either never occurred.
        """
        if self.pair_counts[i, j] == 0 or self.color_counts[i] == 0 or self.color_counts[j] == 0:
            return float("-inf")
        p_i = self.color_counts[i] / self.n_artworks
        p_j = self.color_counts[j] / self.n_artworks
        p_ij = self.pair_counts[i, j] / self.n_artworks
        return math.log(p_ij / (p_i * p_j))

    def ppmi(self, i: int, j: int) -> float:
        """Positive PMI: max(pmi, 0).

        Raw PMI is unstable for rare pairs on a modest dataset — a
        pair observed once can produce a huge PMI, and an unobserved
        pair produces -inf, which isn't a meaningful "these colors
        avoid each other" signal, just an absence of evidence. PPMI
        (standard in distributional semantics for this exact reason)
        clips that noise to 0, so ranking is driven by real positive
        associations rather than by -inf/extreme-count artifacts.
        """
        if self.pair_counts[i, j] == 0:
            return 0.0
        return max(0.0, self.pmi(i, j))

    def build_ppmi_matrix(self) -> np.ndarray:
        """Dense (vocab_size, vocab_size) matrix of ppmi(i,j) for every pair, zero diagonal.

        This is the matrix modeling/embedding.py factorizes via SVD.
        Built by calling the same .ppmi() used for direct pairwise
        ranking (not a separate computation), so the embedding and the
        direct co-occurrence recommender are guaranteed to agree on
        every pairwise association value — only how each aggregates
        that information into a recommendation differs.
        """
        v = self.vocab_size
        matrix = np.zeros((v, v), dtype=np.float64)
        for i in range(v):
            for j in range(i + 1, v):
                value = self.ppmi(i, j)
                matrix[i, j] = value
                matrix[j, i] = value
        return matrix

    # --- persistence ---

    def save(self, path: Path) -> None:
        payload = {
            "vocab_size": self.vocab_size,
            "n_artworks": self.n_artworks,
            "color_counts": self.color_counts.tolist(),
            "pair_counts": self.pair_counts.tolist(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CoOccurrenceModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            vocab_size=payload["vocab_size"],
            n_artworks=payload["n_artworks"],
            color_counts=np.array(payload["color_counts"], dtype=np.int64),
            pair_counts=np.array(payload["pair_counts"], dtype=np.int64),
        )
