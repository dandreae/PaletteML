"""Dominant-color extraction from a single painting.

This is the first genuine ML step in the pipeline: cluster a
painting's pixels (in Lab space) with K-Means and take the cluster
centers, weighted by cluster size, as that painting's palette.

TODO: implement `extract_palette`, including:
  - downsampling large images before clustering (speed)
  - choosing k (fixed N_DOMINANT_COLORS for v1; consider silhouette-
    based k selection as a stretch goal)
  - returning both the Lab centers and their relative pixel weights
"""

from paletteml.config import N_DOMINANT_COLORS


def extract_palette(image_path: str, n_colors: int = N_DOMINANT_COLORS):
    """Extract the n_colors most dominant colors from an image.

    Returns (colors_lab, weights) — TODO.
    """
    raise NotImplementedError
