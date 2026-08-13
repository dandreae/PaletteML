"""Global color vocabulary: a shared set of representative colors.

Extracted painting palettes contain continuous Lab colors — no two
paintings' "red" are numerically identical. To count how often colors
appear together across many paintings, they first need to be mapped
onto a shared, finite vocabulary. This clusters every dominant color
pooled from the training paintings (in CIELAB, for the same
perceptual-uniformity reason extraction itself clusters in Lab) into
`vocab_size` representative bins via K-Means, and lets any new Lab
color be assigned to its nearest bin.

See config.py's DEFAULT_VOCAB_SIZE for the sizing rationale.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from paletteml.color.space import lab_to_rgb, rgb_to_hex
from paletteml.config import DEFAULT_VOCAB_SIZE, RANDOM_SEED


@dataclass(frozen=True)
class VocabEntry:
    """One vocabulary color: a K-Means cluster center in three representations."""

    cluster_id: int
    lab: tuple[float, float, float]
    rgb: tuple[int, int, int]
    hex: str


class ColorVocabulary:
    """A fitted set of representative colors, and nearest-bin lookup.

    Only the cluster centers are persisted (as plain Lab coordinates,
    not a pickled sklearn estimator) — nearest-neighbor assignment at
    inference time is a small, explicit brute-force lookup over
    `vocab_size` points, which is both fast enough and avoids tying
    saved artifacts to a specific sklearn version.
    """

    def __init__(self, entries: list[VocabEntry], random_state: int):
        if not entries:
            raise ValueError("ColorVocabulary requires at least one entry")
        self.entries = entries
        self.random_state = random_state
        self._centers_lab = np.array([e.lab for e in entries], dtype=np.float64)

    @property
    def size(self) -> int:
        return len(self.entries)

    @classmethod
    def fit(
        cls,
        lab_colors: np.ndarray,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        random_state: int = RANDOM_SEED,
    ) -> ColorVocabulary:
        """Cluster pooled Lab colors into `vocab_size` representative bins.

        Parameters
        ----------
        lab_colors : array-like, shape (N, 3)
            Every dominant color's Lab value, pooled across all
            training paintings (i.e. flattened, not one array per
            painting).
        vocab_size : int
            Target number of vocabulary colors. Clipped down if fewer
            distinct samples are available.
        random_state : int
            Fixed for reproducible, deterministic vocabulary fitting.
        """
        lab_colors = np.asarray(lab_colors, dtype=np.float64)
        if lab_colors.ndim != 2 or lab_colors.shape[1] != 3:
            raise ValueError(f"Expected shape (N, 3), got {lab_colors.shape}")

        effective_k = max(1, min(vocab_size, len(lab_colors)))
        kmeans = KMeans(n_clusters=effective_k, n_init=10, random_state=random_state)
        kmeans.fit(lab_colors)

        entries = []
        for cluster_id, center in enumerate(kmeans.cluster_centers_):
            rgb = lab_to_rgb(center)
            entries.append(
                VocabEntry(
                    cluster_id=cluster_id,
                    lab=tuple(float(c) for c in center),
                    rgb=tuple(int(c) for c in rgb),
                    hex=rgb_to_hex(rgb.astype(np.float64)),
                )
            )
        return cls(entries, random_state=random_state)

    def assign(self, lab_color: np.ndarray) -> int:
        """Return the cluster_id of the nearest vocabulary color to `lab_color`."""
        lab_color = np.asarray(lab_color, dtype=np.float64)
        distances = np.sum((self._centers_lab - lab_color) ** 2, axis=1)
        return int(np.argmin(distances))

    def save(self, path: Path) -> None:
        payload = {
            "random_state": self.random_state,
            "entries": [asdict(e) for e in self.entries],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ColorVocabulary:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = [
            VocabEntry(
                cluster_id=e["cluster_id"],
                lab=tuple(e["lab"]),
                rgb=tuple(e["rgb"]),
                hex=e["hex"],
            )
            for e in payload["entries"]
        ]
        return cls(entries, random_state=payload["random_state"])
