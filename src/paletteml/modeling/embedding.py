"""Low-dimensional color embedding via truncated SVD on the PPMI matrix.

Mathematical intuition
-----------------------
The direct co-occurrence recommender (modeling/recommend.py) scores a
candidate purely by its own observed PPMI with each seed — a single
cell of the co-occurrence matrix. That's maximally literal: if two
colors never happened to co-occur in the training paintings, PPMI is
exactly 0, and no amount of *indirect* evidence (e.g. "this candidate
behaves just like several colors the seed DOES pair well with") can
change that score. This project's training set is modest (a few
hundred paintings), so a lot of vocabulary-color pairs simply never
co-occur — the evaluation report from the previous stage found the
PPMI matrix well under half-dense at most vocabulary sizes.

Truncated SVD treats the whole VxV PPMI matrix as a description of
each color, via its row: "here is how strongly this color associates
with every other color in the vocabulary." Factoring PPMI ~= U_k
Sigma_k V_k^T and keeping only the k largest singular values/vectors
finds the k directions of variation that explain the most of that
structure, and represents each color as its coordinates along those
directions (this module uses U_k * sqrt(Sigma_k) as the embedding —
standard practice, since it makes the dot product of two embeddings
approximate the corresponding entry of the low-rank PPMI
reconstruction). Two colors end up with similar embedding vectors if
they have *similar PPMI relationships to the rest of the vocabulary*,
whether or not they were ever directly observed together. This is the
same idea behind classic distributional word embeddings — Levy &
Goldberg (2014) showed skip-gram word2vec is implicitly factorizing a
shifted PMI matrix in almost exactly this way — applied here to color
co-occurrence instead of word co-occurrence.

Practically, this is a compression + denoising step: a 64x64 (or
96x96) PPMI matrix estimated from a few hundred paintings is noisy —
individual cells can be dominated by one or two coincidental
co-occurrences. Keeping only the top-k singular vectors discards the
lower-variance, more sample-noise-dominated structure and keeps the
dominant signal, at the real cost of also discarding rare-but-genuine
associations. Whether that trade helps is an empirical question for
the evaluation harness, not something assumed here.

Ranking difference from direct PPMI (see modeling/svd_recommend.py):
score(seed, candidate) becomes cosine similarity of their embedding
vectors instead of a direct PPMI lookup. Every candidate gets *some*
score this way — including pairs with zero observed co-occurrence —
so the SVD recommender always returns a full top_n ranking, unlike the
direct recommender, which can legitimately return fewer results (even
zero) when a seed has no positive PPMI evidence for anything.

Worth stating precisely, because it's easy to assume otherwise: cosine
similarity does NOT always preserve direct PPMI's pairwise ordering,
even at full rank (verified directly in tests/test_modeling_svd_recommend.py).
Cosine similarity normalizes by each color's embedding norm, which
reflects how strongly-and-broadly connected that color is across the
*whole* vocabulary — so a candidate with slightly lower raw PPMI to a
seed can still rank higher by cosine similarity if its overall
connectivity profile is more "concentrated" in the seed's direction.
This is a real property of the method, not a bug, and it's exactly the
kind of behavior difference this experiment needs to evaluate
empirically rather than assume is an improvement.

Implementation note: matrices here are small (vocab_size is at most a
few hundred), so this uses a full dense SVD via numpy.linalg.svd
rather than sklearn's randomized TruncatedSVD — exact and fully
deterministic with no algorithm/random_state subtlety, at negligible
compute cost for this size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paletteml.modeling.co_occurrence import CoOccurrenceModel


def _compute_full_svd(ppmi_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full SVD of the PPMI matrix. Returns (U, singular_values) — both full-rank."""
    u, singular_values, _vt = np.linalg.svd(ppmi_matrix, full_matrices=False)
    return u, singular_values


@dataclass(frozen=True)
class ColorEmbedding:
    """A k-dimensional embedding of every vocabulary color, from SVD of its PPMI matrix."""

    vocab_size: int
    n_components: int
    vectors: np.ndarray  # shape (vocab_size, n_components)
    singular_values: np.ndarray  # shape (n_components,) — the kept ones, for diagnostics
    explained_variance_ratio: float  # sum(kept singular values^2) / sum(all singular values^2)

    @classmethod
    def fit(cls, co_occurrence: CoOccurrenceModel, n_components: int) -> ColorEmbedding:
        """Fit a single embedding at one dimension. See fit_multiple() to fit
        several dimensions from one underlying SVD computation efficiently."""
        ppmi_matrix = co_occurrence.build_ppmi_matrix()
        u, s = _compute_full_svd(ppmi_matrix)
        return cls._from_full_svd(co_occurrence.vocab_size, u, s, n_components)

    @classmethod
    def _from_full_svd(cls, vocab_size: int, u: np.ndarray, s: np.ndarray, n_components: int) -> ColorEmbedding:
        # capped only at the matrix's actual rank (len(s)) — a caller
        # asking for n_components == vocab_size gets the full, lossless
        # (no-compression) embedding rather than an arbitrarily-blocked
        # request; there's no compression benefit to that choice, but
        # nothing mathematically wrong with allowing it either
        effective_k = max(1, min(n_components, len(s)))
        total_energy = float(np.sum(s**2))
        kept_energy = float(np.sum(s[:effective_k] ** 2))
        ratio = (kept_energy / total_energy) if total_energy > 0 else 0.0
        vectors = u[:, :effective_k] * np.sqrt(s[:effective_k])
        return cls(
            vocab_size=vocab_size,
            n_components=effective_k,
            vectors=vectors,
            singular_values=s[:effective_k],
            explained_variance_ratio=ratio,
        )

    def similarity(self, i: int, j: int) -> float:
        """Cosine similarity between vocabulary colors i and j's embeddings."""
        vi, vj = self.vectors[i], self.vectors[j]
        denom = np.linalg.norm(vi) * np.linalg.norm(vj)
        if denom == 0:
            return 0.0
        return float(np.dot(vi, vj) / denom)

    def save(self, path: Path) -> None:
        payload = {
            "vocab_size": self.vocab_size,
            "n_components": self.n_components,
            "vectors": self.vectors.tolist(),
            "singular_values": self.singular_values.tolist(),
            "explained_variance_ratio": self.explained_variance_ratio,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ColorEmbedding:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            vocab_size=payload["vocab_size"],
            n_components=payload["n_components"],
            vectors=np.array(payload["vectors"], dtype=np.float64),
            singular_values=np.array(payload["singular_values"], dtype=np.float64),
            explained_variance_ratio=payload["explained_variance_ratio"],
        )


def fit_multiple(co_occurrence: CoOccurrenceModel, n_components_list: list[int]) -> dict[int, ColorEmbedding]:
    """Fit embeddings at several dimensions from ONE underlying SVD computation.

    Truncating a single SVD at different k gives genuinely nested
    embeddings — the first k columns of the d=32 embedding's vectors
    are identical to the full d=16 embedding's vectors — and avoids
    repeating the factorization once per dimension tested.
    """
    ppmi_matrix = co_occurrence.build_ppmi_matrix()
    u, s = _compute_full_svd(ppmi_matrix)
    return {k: ColorEmbedding._from_full_svd(co_occurrence.vocab_size, u, s, k) for k in n_components_list}
