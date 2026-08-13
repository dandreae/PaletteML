"""Typed read-back access to the processed artwork dataset.

Reads `data/processed/palettes.jsonl` (written by
`data/ingest.py`/`scripts/build_dataset.py`) back into
`paletteml.color.extraction.DominantColor` objects, so downstream
code (the vocabulary/co-occurrence training pipeline) works with the
same typed representation extraction produces, rather than raw dicts.

Train/val/test splitting belongs to the evaluation stage, not here —
this module only knows how to read the dataset back in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from paletteml.color.extraction import DominantColor


@dataclass(frozen=True)
class LoadedArtwork:
    """One row of the processed dataset, with its palette as typed DominantColors."""

    artwork_id: str
    title: str
    palette: list[DominantColor]


def load_processed_artworks(path: Path) -> list[LoadedArtwork]:
    """Read every row of a palettes.jsonl file into LoadedArtwork objects."""
    artworks = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            palette = [
                DominantColor(
                    hex=c["hex"],
                    rgb=tuple(c["rgb"]),
                    lab=tuple(c["lab"]),
                    proportion=c["proportion"],
                )
                for c in row["palette"]
            ]
            artworks.append(
                LoadedArtwork(artwork_id=row["artwork_id"], title=row["title"], palette=palette)
            )
    return artworks
