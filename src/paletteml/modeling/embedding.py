"""Learn a color-relationship embedding from co-occurring palette colors.

Core idea: quantize the Lab color space into a fixed set of bins,
then treat each painting's extracted palette as a small "bag of
colors". Build a color-by-color co-occurrence matrix across the
whole dataset (colors that appear in the same painting's palette
co-occur), reweight it (e.g. PMI), and factorize it (truncated SVD)
to get a low-dimensional embedding per color bin.

Colors that real paintings actually combine end up close together
in this learned space — the model's notion of "what works well
together" is entirely data-driven, not rule-based.

TODO: implement
  - color quantization (bin Lab space)
  - co-occurrence matrix construction across the training split
  - PMI reweighting + truncated SVD -> embedding matrix
  - clustering of the embedding (K-Means/GMM) into palette archetypes
  - persistence to `models/`
"""


def fit_color_embedding(training_palettes, n_components: int = 32):
    """Fit the co-occurrence embedding on a set of training palettes.

    TODO.
    """
    raise NotImplementedError
