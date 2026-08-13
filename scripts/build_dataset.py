#!/usr/bin/env python
"""CLI: download paintings and extract their dominant-color palettes.

Source: Art Institute of Chicago public API (CC0 metadata + images
for public-domain-flagged artworks). See README.md's "Dataset"
section for licensing/access details.

Usage:
    python scripts/build_dataset.py --limit 100
    python scripts/build_dataset.py --limit 20 --n-colors 6
    python scripts/build_dataset.py --limit 100 --output data/processed/palettes.jsonl -v

Re-running with the same --raw-dir reuses already-downloaded images
instead of re-fetching them.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from paletteml.config import N_DOMINANT_COLORS, PROCESSED_DATA_DIR, RAW_DATA_DIR
from paletteml.data.ingest import build_dataset
from paletteml.data.sources.artic import AicSource


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--limit", type=int, default=100, help="Number of artworks to process")
    parser.add_argument("--n-colors", type=int, default=N_DOMINANT_COLORS)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_DIR / "palettes.jsonl")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    source = AicSource()
    summary = build_dataset(
        source=source,
        limit=args.limit,
        raw_dir=args.raw_dir,
        output_path=args.output,
        n_colors=args.n_colors,
    )

    print()
    print("=" * 60)
    print("Dataset build summary")
    print("=" * 60)
    print(f"Attempted:   {summary.attempted}")
    print(f"Succeeded:   {summary.succeeded}")
    print(f"Failed:      {summary.failed}")
    if summary.failures:
        print("Failures:")
        for artwork_id, reason in summary.failures:
            print(f"  - {artwork_id}: {reason}")
    print(f"Output:      {summary.output_path}")


if __name__ == "__main__":
    main()
