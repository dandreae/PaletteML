"""Generic dataset-ingestion pipeline: source -> cached images -> palettes -> JSONL.

Deliberately source-agnostic: everything here operates against the
ArtworkSource interface (sources/base.py), so a second dataset source
can be added later without touching this file. Source-specific
query/pagination/licensing details live in sources/*.py. Color
extraction is delegated entirely to the existing
`paletteml.color.extraction.extract_palette` — no color logic is
duplicated here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from paletteml.color.extraction import extract_palette
from paletteml.config import N_DOMINANT_COLORS, PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR
from paletteml.data.schema import ArtworkRecord
from paletteml.data.sources.base import ArtworkSource

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = PROCESSED_DATA_DIR / "palettes.jsonl"


@dataclass
class BuildSummary:
    """Result of one build_dataset() run, for CLI reporting and tests."""

    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    output_path: Path | None = None
    failures: list[tuple[str, str]] = field(default_factory=list)  # (artwork_id, reason)


def _local_image_path(raw_dir: Path, artwork_id: str) -> Path:
    """Deterministic local cache path for an artwork's image.

    "artic:11" -> raw_dir/artic_11.jpg
    """
    safe_name = artwork_id.replace(":", "_").replace("/", "_")
    return raw_dir / f"{safe_name}.jpg"


def _portable_path(path: Path) -> str:
    """Render a path relative to PROJECT_ROOT when possible, so the
    processed dataset records a portable path rather than one baked
    to this machine's absolute filesystem layout. Falls back to an
    absolute (but slash-normalized) path if `path` isn't under
    PROJECT_ROOT (e.g. a custom raw_dir outside the repo, as in tests).
    """
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_dataset(
    source: ArtworkSource,
    limit: int,
    raw_dir: Path = RAW_DATA_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    n_colors: int = N_DOMINANT_COLORS,
) -> BuildSummary:
    """Fetch up to `limit` artworks from `source`, extract their dominant-
    color palettes, and write one JSON record per line to `output_path`.

    - Images already present under raw_dir are reused, not re-downloaded.
    - A failure on any single artwork (network error, corrupt/undecodable
      image, unexpected metadata) is logged and skipped — it never aborts
      the whole run. Its cached image (if any) is removed so a later
      re-run retries the download instead of getting stuck on a bad file.
    - Duplicate artwork_ids are skipped defensively, even though sources
      are expected to already dedupe.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = BuildSummary(output_path=output_path)
    seen_ids: set[str] = set()
    records: list[ArtworkRecord] = []

    for metadata in source.iter_candidates(limit):
        if metadata.artwork_id in seen_ids:
            logger.debug("%s: duplicate candidate, skipping", metadata.artwork_id)
            continue
        seen_ids.add(metadata.artwork_id)
        summary.attempted += 1

        image_path = _local_image_path(raw_dir, metadata.artwork_id)
        try:
            if image_path.exists() and image_path.stat().st_size > 0:
                logger.info("[%d] %s: using cached image", summary.attempted, metadata.artwork_id)
            else:
                logger.info("[%d] %s: downloading image", summary.attempted, metadata.artwork_id)
                source.download_image(metadata, image_path)

            palette = extract_palette(image_path, n_colors=n_colors)
            records.append(
                ArtworkRecord(
                    metadata=metadata, image_path=_portable_path(image_path), palette=palette
                )
            )
            summary.succeeded += 1
            width, height = palette.image_size
            logger.info(
                "[%d] %s: OK (%d colors, %dx%d)",
                summary.attempted,
                metadata.artwork_id,
                len(palette.colors),
                width,
                height,
            )
        except Exception as exc:  # noqa: BLE001 - one bad artwork must not kill the run
            summary.failed += 1
            reason = f"{type(exc).__name__}: {exc}"
            summary.failures.append((metadata.artwork_id, reason))
            logger.warning("[%d] %s: FAILED (%s)", summary.attempted, metadata.artwork_id, reason)
            if image_path.exists():
                try:
                    image_path.unlink()
                except OSError:
                    pass
            continue

    _write_jsonl(records, output_path)
    return summary


def _write_jsonl(records: list[ArtworkRecord], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")
