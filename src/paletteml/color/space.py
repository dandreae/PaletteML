"""Perceptual color-space conversions.

RGB is not perceptually uniform: equal numeric distances in RGB do
not correspond to equal perceived differences. All clustering and
nearest-neighbor logic downstream should operate in CIELAB, where
Euclidean distance approximates human-perceived color difference
(Delta E).

TODO: implement thin wrappers around `skimage.color.rgb2lab` /
`lab2rgb` (and `deltaE_ciede2000` for evaluation) with the array
shapes this project standardizes on.
"""


def rgb_to_lab(rgb):
    """Convert an array of sRGB colors (0-255 or 0-1) to CIELAB.

    TODO: implement using skimage.color.rgb2lab.
    """
    raise NotImplementedError


def lab_to_rgb(lab):
    """Convert an array of CIELAB colors back to sRGB (0-255).

    TODO: implement using skimage.color.lab2rgb.
    """
    raise NotImplementedError
