#!/usr/bin/env python
"""Dev helper: visualize dominant-color extraction for one local image.

Not part of the production pipeline or API — this is for eyeballing
that color/extraction.py is producing sane palettes while developing
it, nothing more.

Usage:
    python scripts/visualize_extraction.py path/to/painting.jpg
    python scripts/visualize_extraction.py path/to/painting.jpg --n-colors 6
    python scripts/visualize_extraction.py path/to/painting.jpg --out somewhere/else.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from paletteml.color.extraction import extract_palette
from paletteml.config import N_DOMINANT_COLORS, REPORTS_DIR


def visualize(image_path: Path, n_colors: int, out_path: Path) -> Path:
    result = extract_palette(image_path, n_colors=n_colors)
    original = Image.open(image_path).convert("RGB")

    fig, (ax_img, ax_palette) = plt.subplots(
        1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.2, 1]}
    )

    ax_img.imshow(original)
    ax_img.set_title(image_path.name)
    ax_img.axis("off")

    n = len(result.colors)
    ax_palette.set_xlim(0, 1)
    ax_palette.set_ylim(0, n)
    ax_palette.set_title(f"{n} dominant colors ({result.n_pixels} px sampled)")
    ax_palette.axis("off")

    for i, color in enumerate(result.colors):
        y = n - i - 1
        ax_palette.add_patch(
            plt.Rectangle((0, y), 1, 0.85, color=[c / 255 for c in color.rgb])
        )
        ax_palette.text(
            1.08,
            y + 0.42,
            f"{color.hex}   {color.proportion * 100:5.1f}%",
            va="center",
            fontsize=11,
            family="monospace",
            clip_on=False,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a local image file")
    parser.add_argument("--n-colors", type=int, default=N_DOMINANT_COLORS)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: reports/figures/<image-stem>_palette.png)",
    )
    args = parser.parse_args()

    out_path = args.out or (REPORTS_DIR / "figures" / f"{args.image.stem}_palette.png")
    saved_to = visualize(args.image, args.n_colors, out_path)
    print(f"Saved visualization to {saved_to}")


if __name__ == "__main__":
    main()
