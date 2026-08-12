"""Tests for RGB <-> Lab and RGB <-> hex conversion (color/space.py)."""

import numpy as np
import pytest

from paletteml.color.space import hex_to_rgb, lab_to_rgb, rgb_to_hex, rgb_to_lab


class TestRgbToLab:
    def test_black_is_zero(self):
        lab = rgb_to_lab(np.array([0, 0, 0]))
        assert lab == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)

    def test_white_is_l100_a0_b0(self):
        lab = rgb_to_lab(np.array([255, 255, 255]))
        # skimage's D65 white-point conversion leaves a tiny a/b
        # residual (~0.005) for pure white; not a bug, just float/
        # illuminant rounding, so the tolerance is a bit looser here
        # than elsewhere.
        assert lab == pytest.approx([100.0, 0.0, 0.0], abs=1e-2)

    def test_rejects_wrong_last_dimension(self):
        with pytest.raises(ValueError):
            rgb_to_lab(np.array([255, 255]))

    def test_batch_of_colors(self):
        rgb = np.array([[0, 0, 0], [255, 255, 255]])
        lab = rgb_to_lab(rgb)
        assert lab.shape == (2, 3)
        assert lab[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
        assert lab[1] == pytest.approx([100.0, 0.0, 0.0], abs=1e-2)


class TestLabToRgb:
    @pytest.mark.parametrize(
        "rgb",
        [
            (0, 0, 0),
            (255, 255, 255),
            (255, 0, 0),
            (12, 200, 90),
        ],
    )
    def test_round_trip(self, rgb):
        lab = rgb_to_lab(np.array(rgb, dtype=np.float64))
        recovered = lab_to_rgb(lab)
        # uint8 rounding through Lab can be off by a shade
        assert recovered.tolist() == pytest.approx(list(rgb), abs=1)

    def test_output_is_uint8_and_clipped(self):
        # An out-of-gamut Lab color should clip into [0, 255], not wrap/crash.
        rgb = lab_to_rgb(np.array([50.0, 200.0, 200.0]))
        assert rgb.dtype == np.uint8
        assert rgb.min() >= 0 and rgb.max() <= 255

    def test_rejects_wrong_last_dimension(self):
        with pytest.raises(ValueError):
            lab_to_rgb(np.array([50.0, 0.0]))


class TestHexConversion:
    @pytest.mark.parametrize(
        "rgb,expected",
        [
            ((255, 0, 0), "#ff0000"),
            ((0, 255, 0), "#00ff00"),
            ((0, 0, 255), "#0000ff"),
            ((0, 0, 0), "#000000"),
            ((255, 255, 255), "#ffffff"),
        ],
    )
    def test_rgb_to_hex(self, rgb, expected):
        assert rgb_to_hex(np.array(rgb, dtype=np.float64)) == expected

    def test_rgb_to_hex_clips_out_of_range(self):
        assert rgb_to_hex(np.array([-5.0, 300.0, 128.4])) == "#00ff80"

    @pytest.mark.parametrize(
        "hex_str,expected",
        [
            ("#ff0000", (255, 0, 0)),
            ("ff0000", (255, 0, 0)),
            ("#000000", (0, 0, 0)),
            ("#FFFFFF", (255, 255, 255)),
        ],
    )
    def test_hex_to_rgb(self, hex_str, expected):
        assert hex_to_rgb(hex_str).tolist() == list(expected)

    @pytest.mark.parametrize("bad_hex", ["#fff", "#ff00ff00", "not-a-color", "#gg0000"])
    def test_hex_to_rgb_rejects_invalid(self, bad_hex):
        with pytest.raises(ValueError):
            hex_to_rgb(bad_hex)

    def test_hex_round_trip(self):
        original = np.array([12, 200, 90], dtype=np.uint8)
        assert hex_to_rgb(rgb_to_hex(original.astype(np.float64))).tolist() == original.tolist()
