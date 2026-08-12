"""Tests for dominant-color extraction (color/extraction.py).

Synthetic images with known, exact color proportions are used so
assertions can check real numbers rather than "it didn't crash".
K-Means is run with a fixed random_state throughout for
reproducibility.
"""

import numpy as np
import pytest
from PIL import Image

from paletteml.color.extraction import (
    DominantColor,
    ImageLoadError,
    PaletteExtractionResult,
    extract_palette,
    load_image,
    normalize_image,
)

RANDOM_STATE = 42


def _solid_blocks_image() -> Image.Image:
    """A 10x10 RGB image: 50% pure red, 30% pure green, 20% pure blue.

    Small enough to stay under the default resize threshold, so no
    resampling blurs the block edges and proportions stay exact.
    """
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[:, 0:5] = [255, 0, 0]  # 5 columns x 10 rows = 50 px
    arr[:, 5:8] = [0, 255, 0]  # 3 columns x 10 rows = 30 px
    arr[:, 8:10] = [0, 0, 255]  # 2 columns x 10 rows = 20 px
    return Image.fromarray(arr, mode="RGB")


class TestExtractPaletteSolidBlocks:
    def test_returns_expected_structure(self):
        result = extract_palette(_solid_blocks_image(), n_colors=3, random_state=RANDOM_STATE)
        assert isinstance(result, PaletteExtractionResult)
        assert len(result.colors) == 3
        assert all(isinstance(c, DominantColor) for c in result.colors)
        assert result.n_pixels == 100
        assert result.image_size == (10, 10)

    def test_recovers_known_colors_and_proportions(self):
        result = extract_palette(_solid_blocks_image(), n_colors=3, random_state=RANDOM_STATE)
        # sorted descending by proportion: red (0.5), green (0.3), blue (0.2)
        proportions = [c.proportion for c in result.colors]
        assert proportions == pytest.approx([0.5, 0.3, 0.2], abs=1e-6)

        red, green, blue = result.colors
        assert red.rgb == pytest.approx((255, 0, 0), abs=2)
        assert green.rgb == pytest.approx((0, 255, 0), abs=2)
        assert blue.rgb == pytest.approx((0, 0, 255), abs=2)
        assert red.hex.startswith("#")

    def test_proportions_sum_to_one(self):
        result = extract_palette(_solid_blocks_image(), n_colors=3, random_state=RANDOM_STATE)
        assert sum(c.proportion for c in result.colors) == pytest.approx(1.0, abs=1e-9)

    def test_deterministic_with_fixed_random_state(self):
        r1 = extract_palette(_solid_blocks_image(), n_colors=3, random_state=RANDOM_STATE)
        r2 = extract_palette(_solid_blocks_image(), n_colors=3, random_state=RANDOM_STATE)
        assert [c.hex for c in r1.colors] == [c.hex for c in r2.colors]
        assert [c.proportion for c in r1.colors] == [c.proportion for c in r2.colors]


class TestGrayscaleImage:
    def test_extraction_succeeds_and_stays_neutral(self):
        arr = np.zeros((20, 20), dtype=np.uint8)
        arr[:, :10] = 40
        arr[:, 10:] = 220
        image = Image.fromarray(arr, mode="L")

        result = extract_palette(image, n_colors=2, random_state=RANDOM_STATE)

        assert len(result.colors) == 2
        assert sum(c.proportion for c in result.colors) == pytest.approx(1.0, abs=1e-9)
        for color in result.colors:
            r, g, b = color.rgb
            # grayscale in -> roughly gray out (channels close to equal)
            assert max(abs(r - g), abs(g - b), abs(r - b)) <= 2


class TestRgbaImage:
    def test_transparent_region_flattens_to_white(self):
        arr = np.zeros((10, 10, 4), dtype=np.uint8)
        arr[:, :5] = [200, 30, 30, 255]  # opaque red, left half
        arr[:, 5:] = [0, 0, 0, 0]  # fully transparent, right half
        image = Image.fromarray(arr, mode="RGBA")

        normalized = normalize_image(image)
        assert normalized.mode == "RGB"
        normalized_arr = np.asarray(normalized)
        assert normalized_arr[0, 9].tolist() == [255, 255, 255]  # transparent -> white
        assert normalized_arr[0, 0].tolist() == [200, 30, 30]  # opaque preserved

    def test_extraction_succeeds_on_rgba(self):
        arr = np.zeros((10, 10, 4), dtype=np.uint8)
        arr[:, :5] = [200, 30, 30, 255]
        arr[:, 5:] = [0, 0, 0, 0]
        image = Image.fromarray(arr, mode="RGBA")

        result = extract_palette(image, n_colors=2, random_state=RANDOM_STATE)
        assert len(result.colors) == 2
        assert sum(c.proportion for c in result.colors) == pytest.approx(1.0, abs=1e-9)


class TestEdgeCases:
    def test_fewer_pixels_than_requested_colors_is_clipped_not_a_crash(self):
        image = Image.fromarray(np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8), mode="RGB")
        result = extract_palette(image, n_colors=5, random_state=RANDOM_STATE)
        assert result.n_pixels == 2
        assert len(result.colors) <= 2

    def test_resizes_large_images(self):
        arr = np.random.default_rng(0).integers(0, 255, size=(500, 400, 3), dtype=np.uint8)
        image = Image.fromarray(arr, mode="RGB")
        result = extract_palette(image, n_colors=5, max_dimension=100, random_state=RANDOM_STATE)
        assert max(result.image_size) <= 100


class TestInvalidImageInput:
    def test_missing_file_raises_image_load_error(self, tmp_path):
        missing = tmp_path / "does_not_exist.png"
        with pytest.raises(ImageLoadError):
            load_image(missing)

    def test_corrupted_file_raises_image_load_error(self, tmp_path):
        corrupted = tmp_path / "corrupted.png"
        corrupted.write_bytes(b"not actually an image, just some bytes")
        with pytest.raises(ImageLoadError):
            load_image(corrupted)

    def test_extract_palette_propagates_image_load_error(self, tmp_path):
        corrupted = tmp_path / "corrupted.jpg"
        corrupted.write_bytes(b"\x00\x01\x02garbage")
        with pytest.raises(ImageLoadError):
            extract_palette(corrupted)
