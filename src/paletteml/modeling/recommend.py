"""Palette recommendation built on top of the learned color embedding.

Given a user-provided color (or partial palette), locate the
nearest quantized bin(s) in the learned embedding space and return
the nearest-neighbor colors/palettes — a retrieval grounded in the
fitted model, not a hardcoded color-wheel rule.

TODO: implement
  - map an arbitrary input RGB/Lab color to its nearest learned bin
  - k-NN lookup in embedding space for companion colors
  - optional: retrieve and surface the real painting(s) a
    recommendation came from, for interpretability in the UI
"""


def recommend_palette(seed_colors, n_suggestions: int = 4):
    """Recommend companion colors for a seed color or partial palette.

    TODO.
    """
    raise NotImplementedError
