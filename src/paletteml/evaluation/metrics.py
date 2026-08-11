"""Held-out color prediction: the project's core evaluation task.

For each painting in the held-out test split: remove one color from
its extracted palette, give the model the rest, and ask it to
recommend companion colors. Score whether the held-out color is
recovered.

Metrics (TODO):
  - top-k hit rate: is a suggestion within a Delta E threshold of
    the held-out color, for k in {1, 3, 5}
  - mean Delta E (CIEDE2000) from the held-out color to the nearest
    suggestion
  - compare against baselines: (a) random color from the training
    palette distribution, (b) classic color-wheel complementary /
    analogous rule applied to the remaining colors

Reporting: write a metrics table + a couple of qualitative example
figures (real palette vs. model suggestions) to `reports/`.
"""


def evaluate_held_out_color(model, test_palettes, k=(1, 3, 5)):
    """Run the held-out color-recovery evaluation.

    TODO.
    """
    raise NotImplementedError


def baseline_random(train_palettes):
    """Baseline: recommend colors drawn from the training color distribution.

    TODO.
    """
    raise NotImplementedError


def baseline_color_wheel(seed_colors):
    """Baseline: classic complementary/analogous color-wheel rule.

    TODO.
    """
    raise NotImplementedError
