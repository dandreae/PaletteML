"""PaletteML: learns color relationships from real paintings and
recommends colors that work well with a user-provided color or
partial palette.

Subpackages
-----------
data        Dataset acquisition and loading (paintings + metadata).
color       Dominant-color extraction and perceptual color-space conversion.
modeling    Unsupervised color-relationship model (embedding + clustering)
            and the recommendation logic built on top of it.
evaluation  Quantitative evaluation of recommendation quality.
api         FastAPI application exposing the trained model.
"""

__version__ = "0.1.0"
