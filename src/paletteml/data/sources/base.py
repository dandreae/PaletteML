"""Abstract interface a dataset source must implement.

Keeps source-specific concerns (API shape, query params, pagination,
licensing filters, rate-limiting, retries) out of the generic
ingestion pipeline in data/ingest.py, so a second source can be
added later without touching orchestration logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from paletteml.data.schema import ArtworkMetadata


class ArtworkSource(ABC):
    """Something that can list candidate artworks and download their images."""

    name: str

    @abstractmethod
    def iter_candidates(self, limit: int) -> Iterator[ArtworkMetadata]:
        """Yield up to `limit` artworks that have a usable, licensed image.

        Implementations are responsible for their own filtering
        (licensing, artwork type, image availability) and pagination.
        """

    @abstractmethod
    def download_image(self, metadata: ArtworkMetadata, dest_path: Path) -> None:
        """Download the artwork's image to dest_path. Raises on failure."""