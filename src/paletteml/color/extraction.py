"""Dominant-color extraction from a single painting.

This is the first genuine ML step in the pipeline: cluster a
painting's pixels (in CIELAB) with K-Means and take the cluster
centers, weighted by cluster size, as that painting's palette.
See color/space.py for why Lab is used instead of RGB.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.cluster import KMeans

from paletteml.color.space import lab_to_rgb, rgb_to_hex, rgb_to_lab
from paletteml.config import EXTRACTION_MAX_DIMENSION, N_DOMINANT_COLORS, RANDOM_SEED

ImageSource = Union[str, Path, BinaryIO, Image.Image]


class ImageLoadError(Exception):
    """Raised when an image cannot be safely opened or decoded."""


@dataclass(frozen=True)
class DominantColor:
    """A single dominant color extracted from a painting."""

    hex: str
    rgb: tuple[int, int, int]
    lab: tuple[float, float, float]
    proportion: float


@dataclass(frozen=True)
class PaletteExtractionResult:
    """Structured result of extracting a painting's dominant-color palette."""

    colors: list[DominantColor]
    image_size: tuple[int, int]  # (width, height) after normalization
    n_pixels: int  # number of pixels clustered over

    def as_dict(self) -> dict:
        return asdict(self)


def load_image(source: ImageSource) -> Image.Image:
    """Safely open and fully decode an image.

    Accepts a filesystem path, a file-like object, or an already-open
    PIL Image. Pillow opens images lazily, so a truncated/corrupted
    file can pass `Image.open` and only fail once pixel data is
    actually read — `.load()` forces that now, at a single,
    predictable point, instead of failing later inside extraction.

    Raises
    ------
    ImageLoadError
        If the source cannot be opened or decoded as an image.
    """
    if isinstance(source, Image.Image):
        image = source
    else:
        try:
            image = Image.open(source)
        except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
            raise ImageLoadError(f"Could not open image {source!r}: {exc}") from exc

    try:
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"Could not decode image {source!r}: {exc}") from exc

    return image


def normalize_image(
    image: Image.Image, max_dimension: int = EXTRACTION_MAX_DIMENSION
) -> Image.Image:
    """Return an RGB copy of `image`, ready for pixel clustering.

    - Transparency (RGBA/LA, or palette images with a transparency
      entry) is flattened onto a white background rather than simply
      dropped, so a fully transparent pixel becomes white instead of
      whatever arbitrary RGB it stored underneath.
    - Any other mode (L, P, CMYK, ...) is converted to RGB.
    - The image is downscaled (never upscaled) so its longest side is
      at most `max_dimension`, since clustering doesn't benefit from
      full resolution and this keeps extraction fast.
    """
    has_transparency = image.mode in ("RGBA", "LA") or "transparency" in image.info
    if has_transparency:
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    longest_side = max(width, height)
    if longest_side > max_dimension:
        scale = max_dimension / longest_side
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    return image


def extract_palette(
    source: ImageSource,
    n_colors: int = N_DOMINANT_COLORS,
    max_dimension: int = EXTRACTION_MAX_DIMENSION,
    random_state: int = RANDOM_SEED,
) -> PaletteExtractionResult:
    """Extract the dominant colors of a painting via K-Means in CIELAB.

    Parameters
    ----------
    source : path, file-like, or PIL Image
        The painting to process.
    n_colors : int
        Target number of dominant colors (clipped down if the image
        has fewer pixels than this, which only matters for tiny
        images).
    max_dimension : int
        Longest side, in pixels, to downscale to before clustering.
    random_state : int
        Fixed for reproducible cluster assignments across runs.

    Returns
    -------
    PaletteExtractionResult
        Dominant colors sorted by descending proportion of the image
        they represent, plus the pixel count and normalized image
        size the extraction ran on.
    """
    image = load_image(source)
    image = normalize_image(image, max_dimension=max_dimension)

    pixels_rgb = np.asarray(image, dtype=np.uint8).reshape(-1, 3)
    n_pixels = pixels_rgb.shape[0]
    effective_k = max(1, min(n_colors, n_pixels))

    pixels_lab = rgb_to_lab(pixels_rgb)

    kmeans = KMeans(n_clusters=effective_k, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(pixels_lab)

    counts = np.bincount(labels, minlength=effective_k)
    proportions = counts / n_pixels

    colors = []
    for i in range(effective_k):
        lab_center = kmeans.cluster_centers_[i]
        rgb_center = lab_to_rgb(lab_center)
        colors.append(
            DominantColor(
                hex=rgb_to_hex(rgb_center.astype(np.float64)),
                rgb=tuple(int(c) for c in rgb_center),
                lab=tuple(float(c) for c in lab_center),
                proportion=float(proportions[i]),
            )
        )

    colors.sort(key=lambda c: c.proportion, reverse=True)

    return PaletteExtractionResult(
        colors=colors, image_size=image.size, n_pixels=n_pixels
    )
