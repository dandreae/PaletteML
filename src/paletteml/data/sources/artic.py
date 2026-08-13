"""Art Institute of Chicago (api.artic.edu) dataset source.

Chosen because it needs no API key, documents a courtesy
`AIC-User-Agent` header instead of hard rate limits, and — for
artworks flagged `is_public_domain: true` — serves both metadata and
images under CC0. See README.md's "Dataset" section for the full
licensing/access writeup.

Query note: AIC's documented structured query syntax
(`query[bool][must][...]`) proved unreliable in practice (returned 0
results for filters that worked moments earlier via other params).
Instead we use the plain full-text search `q=painting`, which is
consistently reliable, and do the actual filtering — artwork type,
public-domain status, image availability — ourselves in
`_is_valid_candidate`. This is also easier to unit test.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests

from paletteml.data.schema import ArtworkMetadata
from paletteml.data.sources.base import ArtworkSource

logger = logging.getLogger(__name__)

API_BASE = "https://api.artic.edu/api/v1"
IIIF_BASE = "https://www.artic.edu/iiif/2"
SEARCH_QUERY = "painting"
SEARCH_FIELDS = (
    "id,title,artist_display,date_display,date_start,date_end,"
    "image_id,is_public_domain,artwork_type_title"
)
PAGE_SIZE = 100
# Safety cap on how many search pages we'll page through for one
# build, so a very large --limit with a low candidate-hit-rate can't
# spin forever / hammer the API.
MAX_PAGES = 50
IMAGE_WIDTH = 600  # IIIF request width; extraction downsamples further anyway
MAX_DOWNLOAD_RETRIES = 3
USER_AGENT = "PaletteML/0.1 (student portfolio project; no production traffic)"


class ImageDownloadError(Exception):
    """Raised when an artwork's image could not be downloaded."""


class AicSource(ArtworkSource):
    """ArtworkSource backed by the Art Institute of Chicago public API."""

    name = "artic"

    def __init__(self, session: requests.Session | None = None, request_delay: float = 0.2):
        self.session = session or requests.Session()
        # Minimum pause between outbound requests (search pages and
        # image downloads alike) — polite default, not a hard API
        # requirement.
        self.request_delay = request_delay

    def iter_candidates(self, limit: int) -> Iterator[ArtworkMetadata]:
        seen_ids: set[str] = set()
        yielded = 0
        page = 1

        while yielded < limit and page <= MAX_PAGES:
            if page > 1:
                time.sleep(self.request_delay)
            data = self._search_page(page)
            items = data.get("data", [])
            if not items:
                logger.info("artic: source exhausted at page %d", page)
                break

            for item in items:
                if yielded >= limit:
                    break
                if not self._is_valid_candidate(item):
                    continue
                metadata = self._to_metadata(item)
                if metadata.artwork_id in seen_ids:
                    continue
                seen_ids.add(metadata.artwork_id)
                yielded += 1
                yield metadata

            page += 1

    def download_image(self, metadata: ArtworkMetadata, dest_path: Path) -> None:
        last_exc: Exception | None = None
        for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
            try:
                response = self.session.get(
                    metadata.image_url,
                    headers={"AIC-User-Agent": USER_AGENT},
                    timeout=30,
                )
                response.raise_for_status()
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_bytes(response.content)
                time.sleep(self.request_delay)
                return
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < MAX_DOWNLOAD_RETRIES:
                    time.sleep(self.request_delay * attempt)
        raise ImageDownloadError(
            f"Failed to download {metadata.image_url} after {MAX_DOWNLOAD_RETRIES} attempts: {last_exc}"
        ) from last_exc

    # --- internals ---

    def _search_page(self, page: int) -> dict:
        response = self.session.get(
            f"{API_BASE}/artworks/search",
            params={
                "q": SEARCH_QUERY,
                "fields": SEARCH_FIELDS,
                "limit": PAGE_SIZE,
                "page": page,
            },
            headers={"AIC-User-Agent": USER_AGENT},
            timeout=20,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _is_valid_candidate(item: dict[str, Any]) -> bool:
        return (
            item.get("artwork_type_title") == "Painting"
            and item.get("is_public_domain") is True
            and bool(item.get("image_id"))
        )

    def _to_metadata(self, item: dict[str, Any]) -> ArtworkMetadata:
        return ArtworkMetadata(
            artwork_id=f"{self.name}:{item['id']}",
            title=item.get("title") or "Untitled",
            artist=item.get("artist_display"),
            year_display=item.get("date_display"),
            year_start=item.get("date_start"),
            year_end=item.get("date_end"),
            source=self.name,
            source_url=f"https://www.artic.edu/artworks/{item['id']}",
            image_url=f"{IIIF_BASE}/{item['image_id']}/full/{IMAGE_WIDTH},/0/default.jpg",
            license="CC0",
        )
