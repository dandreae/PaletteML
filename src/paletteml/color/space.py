"""Perceptual color-space conversions and hex helpers.

RGB is not perceptually uniform: equal numeric distances in RGB do
not correspond to equal perceived differences. All clustering and
distance calculations in this project operate in CIELAB, where
Euclidean distance approximates human-perceived color difference.
These are thin, well-tested wrappers so the rest of the codebase
never touches skimage.color directly.
"""

from __future__ import annotations

import numpy as np
from skimage.color import lab2rgb, rgb2lab


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB color(s) to CIELAB.

    Parameters
    ----------
    rgb : array-like, shape (..., 3)
        sRGB values in the 0-255 range (uint8 or float). Any leading
        shape is supported: a single color (3,), a list of colors
        (N, 3), or a full image (H, W, 3).

    Returns
    -------
    np.ndarray, same leading shape, dtype float64
        CIELAB values: L in [0, 100], a/b roughly in [-128, 127].
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.shape[-1] != 3:
        raise ValueError(f"Expected last dimension of size 3 (RGB), got shape {rgb.shape}")
    return rgb2lab(rgb / 255.0)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB color(s) back to sRGB.

    Parameters
    ----------
    lab : array-like, shape (..., 3)
        CIELAB values (as produced by rgb_to_lab).

    Returns
    -------
    np.ndarray, same leading shape, dtype uint8
        sRGB values in [0, 255], out-of-gamut results clipped.
    """
    lab = np.asarray(lab, dtype=np.float64)
    if lab.shape[-1] != 3:
        raise ValueError(f"Expected last dimension of size 3 (Lab), got shape {lab.shape}")
    rgb = lab2rgb(lab)
    rgb_255 = np.clip(rgb * 255.0, 0, 255)
    return np.round(rgb_255).astype(np.uint8)


def rgb_to_hex(rgb: np.ndarray) -> str:
    """Convert a single sRGB color (3,) to a "#rrggbb" hex string.

    Values are rounded and clipped to [0, 255] so this also accepts
    raw (unclipped) float output from color-space round-trips.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.shape != (3,):
        raise ValueError(f"Expected a single RGB color of shape (3,), got shape {rgb.shape}")
    r, g, b = (int(np.clip(round(c), 0, 255)) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_str: str) -> np.ndarray:
    """Convert a "#rrggbb" (or "rrggbb") hex string to an RGB array."""
    hex_str = hex_str.lstrip("#")
    if len(hex_str) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got {hex_str!r}")
    try:
        r, g, b = (int(hex_str[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError as exc:
        raise ValueError(f"Invalid hex color: {hex_str!r}") from exc
    return np.array([r, g, b], dtype=np.uint8)
