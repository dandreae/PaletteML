"""Typed access to the painting manifest and extracted-palette table.

TODO: implement a small class (or pandas-backed helpers) that loads
`data/processed/manifest.csv` and `data/processed/palettes.parquet`
and exposes train/val/test splits by painting (not by color, to
avoid leakage between splits).
"""
