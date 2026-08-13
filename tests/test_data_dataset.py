"""Tests for reading back the processed dataset (data/dataset.py)."""

import json

from paletteml.color.extraction import DominantColor
from paletteml.data.dataset import load_processed_artworks


def test_loads_rows_into_typed_dominant_colors(tmp_path):
    path = tmp_path / "palettes.jsonl"
    rows = [
        {
            "artwork_id": "fake:1",
            "title": "Test Artwork",
            "palette": [
                {"hex": "#ff0000", "rgb": [255, 0, 0], "lab": [53.2, 80.1, 67.2], "proportion": 0.6},
                {"hex": "#0000ff", "rgb": [0, 0, 255], "lab": [32.3, 79.2, -107.9], "proportion": 0.4},
            ],
        }
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    artworks = load_processed_artworks(path)

    assert len(artworks) == 1
    artwork = artworks[0]
    assert artwork.artwork_id == "fake:1"
    assert artwork.title == "Test Artwork"
    assert len(artwork.palette) == 2
    assert isinstance(artwork.palette[0], DominantColor)
    assert artwork.palette[0].rgb == (255, 0, 0)
    assert artwork.palette[0].lab == (53.2, 80.1, 67.2)
    assert artwork.palette[0].proportion == 0.6


def test_skips_blank_lines(tmp_path):
    path = tmp_path / "palettes.jsonl"
    row = {"artwork_id": "fake:1", "title": "T", "palette": []}
    path.write_text(f"\n{json.dumps(row)}\n\n", encoding="utf-8")

    artworks = load_processed_artworks(path)

    assert len(artworks) == 1
    assert artworks[0].palette == []
