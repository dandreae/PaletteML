#!/usr/bin/env python
"""CLI: fit the color-relationship embedding on the training split.

Usage (once implemented):
    python scripts/train.py

TODO: load data/processed/palettes.parquet, split by painting into
train/val/test, call paletteml.modeling.embedding.fit_color_embedding
on the training split, and persist the fitted model to models/.
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
