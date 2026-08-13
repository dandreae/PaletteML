"""Source-agnostic schema for the processed artwork dataset.

`ArtworkMetadata` is the stable identity/attribution record any
dataset source produces. `ArtworkRecord` composes that metadata with
the output of the existing color-extraction pipeline
(`paletteml.color.extraction`) plus where the image is cached
locally, and knows how to flatten itself into one JSONL row. No
color-conversion or clustering logic lives here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from paletteml.color.extraction import PaletteExtractionResult


@dataclass(frozen=True)
class ArtworkMetadata:
    """Stable, source-agnostic identity and attribution for one artwork."""

    artwork_id: str  # namespaced by source, e.g. "artic:11" — stable across re-runs
    title: str
    artist: str | None
    year_display: str | None  # human-readable, e.g. "1878" or "c. 1620/30"
    year_start: int | None
    year_end: int | None
    source: str  # short source id, e.g. "artic"
    source_url: str  # link back to the object page, for attribution
    image_url: str  # URL the image was actually fetched from
    license: str  # e.g. "CC0"


@dataclass(frozen=True)
class ArtworkRecord:
    """One fully processed dataset row: metadata + local image + palette."""

    metadata: ArtworkMetadata
    image_path: str  # local path the image was cached to (not committed to git)
    palette: PaletteExtractionResult

    def to_json_dict(self) -> dict:
        """Flatten into a single JSON-serializable dict (one JSONL row)."""
        row = asdict(self.metadata)
        row["image_path"] = self.image_path
        row["image_width"], row["image_height"] = self.palette.image_size
        row["n_pixels_sampled"] = self.palette.n_pixels
        row["palette"] = [asdict(c) for c in self.palette.colors]
        return row