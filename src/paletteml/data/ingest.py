"""Download / assemble the raw painting dataset.

Plan (not yet implemented): pull a manageable subset of a public
painting dataset (e.g. WikiArt via the Hugging Face `datasets` hub,
or the Kaggle "Painter by Numbers" set), save images under
`data/raw/`, and write a manifest CSV (image path, artist, style,
genre if available) to `data/processed/manifest.csv`.

Keep the subset small enough (a few thousand images) to extract
palettes and train the relationship model within the project's
one-week timebox.
"""

from paletteml.config import RAW_DATA_DIR


def download_dataset(limit: int | None = None) -> None:
    """Download raw painting images into RAW_DATA_DIR.

    TODO: implement dataset source, download, and manifest generation.
    """
    raise NotImplementedError
