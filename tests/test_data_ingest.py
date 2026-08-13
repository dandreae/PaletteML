"""Tests for the generic ingestion pipeline (data/ingest.py).

Uses an in-memory fake ArtworkSource instead of the real AIC source,
so these exercise the real download-cache -> extract_palette ->
JSONL-write pipeline end-to-end without any network access, and
without mocking color extraction itself.
"""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from paletteml.data.ingest import build_dataset
from paletteml.data.schema import ArtworkMetadata
from paletteml.data.sources.base import ArtworkSource

RANDOM_STATE = 42


def _solid_color_jpeg_bytes(rgb: tuple[int, int, int], size=(20, 20)) -> bytes:
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = rgb
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG")
    return buf.getvalue()


def _metadata(artwork_id: str, title: str = "Test Artwork") -> ArtworkMetadata:
    return ArtworkMetadata(
        artwork_id=artwork_id,
        title=title,
        artist="Test Artist",
        year_display="1900",
        year_start=1900,
        year_end=1900,
        source="fake",
        source_url=f"https://example.invalid/{artwork_id}",
        image_url=f"https://example.invalid/{artwork_id}.jpg",
        license="CC0",
    )


class FakeArtworkSource(ArtworkSource):
    """In-memory ArtworkSource: candidates and image bytes are fixed upfront."""

    name = "fake"

    def __init__(self, candidates: list[ArtworkMetadata], images: dict[str, bytes | Exception]):
        self._candidates = candidates
        self._images = images
        self.download_calls: list[str] = []

    def iter_candidates(self, limit: int):
        yield from self._candidates[:limit]

    def download_image(self, metadata: ArtworkMetadata, dest_path: Path) -> None:
        self.download_calls.append(metadata.artwork_id)
        payload = self._images[metadata.artwork_id]
        if isinstance(payload, Exception):
            raise payload
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(payload)


class TestBuildDatasetHappyPath:
    def test_writes_jsonl_with_expected_fields(self, tmp_path):
        candidates = [_metadata("fake:1"), _metadata("fake:2"), _metadata("fake:3")]
        images = {
            "fake:1": _solid_color_jpeg_bytes((220, 20, 20)),
            "fake:2": _solid_color_jpeg_bytes((20, 220, 20)),
            "fake:3": _solid_color_jpeg_bytes((20, 20, 220)),
        }
        source = FakeArtworkSource(candidates, images)
        output_path = tmp_path / "out.jsonl"

        summary = build_dataset(
            source=source,
            limit=3,
            raw_dir=tmp_path / "raw",
            output_path=output_path,
            n_colors=2,
        )

        assert summary.attempted == 3
        assert summary.succeeded == 3
        assert summary.failed == 0
        assert summary.failures == []

        lines = output_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3

        row = json.loads(lines[0])
        assert row["artwork_id"] == "fake:1"
        assert row["title"] == "Test Artwork"
        assert row["source"] == "fake"
        assert row["license"] == "CC0"
        assert Path(row["image_path"]).exists()
        assert row["image_width"] == 20 and row["image_height"] == 20
        assert len(row["palette"]) == 2
        for color in row["palette"]:
            assert set(color.keys()) == {"hex", "rgb", "lab", "proportion"}
            assert color["hex"].startswith("#")
        assert sum(c["proportion"] for c in row["palette"]) == pytest.approx(1.0, abs=1e-6)


class TestBuildDatasetFailureHandling:
    def test_corrupt_and_missing_images_do_not_crash_the_run(self, tmp_path):
        candidates = [_metadata("fake:1"), _metadata("fake:2"), _metadata("fake:3")]
        images = {
            "fake:1": _solid_color_jpeg_bytes((220, 20, 20)),
            "fake:2": b"this is not a valid image file",  # corrupt
            "fake:3": ConnectionError("simulated network failure"),  # missing/failed download
        }
        source = FakeArtworkSource(candidates, images)
        output_path = tmp_path / "out.jsonl"

        summary = build_dataset(
            source=source, limit=3, raw_dir=tmp_path / "raw", output_path=output_path, n_colors=2
        )

        assert summary.attempted == 3
        assert summary.succeeded == 1
        assert summary.failed == 2
        failed_ids = {artwork_id for artwork_id, _reason in summary.failures}
        assert failed_ids == {"fake:2", "fake:3"}

        lines = output_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["artwork_id"] == "fake:1"

        # the corrupt cached file should have been removed, not left behind
        assert not (tmp_path / "raw" / "fake_2.jpg").exists()

    def test_dedupes_repeated_artwork_ids(self, tmp_path):
        dup = _metadata("fake:1")
        source = FakeArtworkSource(
            [dup, dup], {"fake:1": _solid_color_jpeg_bytes((100, 100, 100))}
        )
        output_path = tmp_path / "out.jsonl"

        summary = build_dataset(
            source=source, limit=2, raw_dir=tmp_path / "raw", output_path=output_path, n_colors=1
        )

        assert summary.attempted == 1
        assert summary.succeeded == 1
        assert source.download_calls == ["fake:1"]
        assert len(output_path.read_text(encoding="utf-8").splitlines()) == 1


class TestBuildDatasetCaching:
    def test_rerun_uses_cached_image_and_does_not_redownload(self, tmp_path):
        candidates = [_metadata("fake:1")]
        images = {"fake:1": _solid_color_jpeg_bytes((50, 60, 70))}
        raw_dir = tmp_path / "raw"

        source1 = FakeArtworkSource(candidates, images)
        build_dataset(
            source=source1, limit=1, raw_dir=raw_dir, output_path=tmp_path / "out1.jsonl", n_colors=1
        )
        assert source1.download_calls == ["fake:1"]

        # second run: image bytes deliberately set to an exception —
        # if this were invoked, the run would fail. It shouldn't be
        # invoked at all because the cached file from run 1 is reused.
        source2 = FakeArtworkSource(candidates, {"fake:1": AssertionError("should not download")})
        summary2 = build_dataset(
            source=source2, limit=1, raw_dir=raw_dir, output_path=tmp_path / "out2.jsonl", n_colors=1
        )

        assert source2.download_calls == []
        assert summary2.succeeded == 1
        assert summary2.failed == 0


class TestBuildDatasetDeterminism:
    def test_identical_output_across_independent_runs(self, tmp_path):
        candidates = [_metadata("fake:1"), _metadata("fake:2")]
        images = {
            "fake:1": _solid_color_jpeg_bytes((180, 90, 40)),
            "fake:2": _solid_color_jpeg_bytes((30, 140, 200)),
        }

        out1 = tmp_path / "run1" / "out.jsonl"
        build_dataset(
            source=FakeArtworkSource(candidates, images),
            limit=2,
            raw_dir=tmp_path / "run1_raw",
            output_path=out1,
            n_colors=3,
        )

        out2 = tmp_path / "run2" / "out.jsonl"
        build_dataset(
            source=FakeArtworkSource(candidates, images),
            limit=2,
            raw_dir=tmp_path / "run2_raw",
            output_path=out2,
            n_colors=3,
        )

        def normalize(path: Path) -> list[dict]:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for row in rows:
                row.pop("image_path")  # differs between the two raw_dirs by design
            return rows

        assert normalize(out1) == normalize(out2)
