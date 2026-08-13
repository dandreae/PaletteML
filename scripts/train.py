#!/usr/bin/env python
"""CLI: fit the color vocabulary + co-occurrence model on the processed
artwork dataset, and persist artifacts to models/.

Usage:
    python scripts/train.py
    python scripts/train.py --vocab-size 64 --input data/processed/palettes.jsonl

Produces:
    models/color_vocabulary.json   ColorVocabulary (see modeling/vocabulary.py)
    models/co_occurrence.json      CoOccurrenceModel (see modeling/co_occurrence.py)

Inference (scripts/recommend.py or library use) loads these artifacts
rather than retraining.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from paletteml.config import DEFAULT_VOCAB_SIZE, MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED
from paletteml.data.dataset import load_processed_artworks
from paletteml.modeling.baseline import PopularityBaseline
from paletteml.modeling.co_occurrence import CoOccurrenceModel
from paletteml.modeling.encoding import encode_palette
from paletteml.modeling.vocabulary import ColorVocabulary


def train(
    input_path: Path,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    models_dir: Path = MODELS_DIR,
    random_state: int = RANDOM_SEED,
) -> None:
    artworks = load_processed_artworks(input_path)
    if not artworks:
        raise SystemExit(f"No artworks found in {input_path} — run scripts/build_dataset.py first.")

    all_lab = np.array([color.lab for artwork in artworks for color in artwork.palette])
    print(f"Loaded {len(artworks)} artworks, {len(all_lab)} extracted dominant colors total.")

    print(f"Fitting color vocabulary (vocab_size={vocab_size})...")
    vocabulary = ColorVocabulary.fit(all_lab, vocab_size=vocab_size, random_state=random_state)
    print(f"  -> {vocabulary.size} vocabulary colors")

    print("Encoding artwork palettes into vocabulary space...")
    artwork_vocab_ids = []
    for artwork in artworks:
        weights = encode_palette(artwork.palette, vocabulary)
        artwork_vocab_ids.append(set(weights.keys()))
    avg_bins = sum(len(ids) for ids in artwork_vocab_ids) / len(artwork_vocab_ids)
    print(f"  -> average {avg_bins:.2f} distinct vocabulary colors per painting")

    print("Building co-occurrence statistics...")
    co_occurrence = CoOccurrenceModel.fit(artwork_vocab_ids, vocab_size=vocabulary.size)

    n_pairs = vocabulary.size * (vocabulary.size - 1) // 2
    observed_pairs = int(np.sum(co_occurrence.pair_counts > 0) // 2)
    density = observed_pairs / n_pairs if n_pairs else 0.0

    vocab_path = models_dir / "color_vocabulary.json"
    co_path = models_dir / "co_occurrence.json"
    vocabulary.save(vocab_path)
    co_occurrence.save(co_path)

    print()
    print("=" * 60)
    print("Training summary")
    print("=" * 60)
    print(f"Artworks used:            {co_occurrence.n_artworks}")
    print(f"Vocabulary size:          {vocabulary.size}")
    print(f"Avg vocab colors/painting: {avg_bins:.2f}")
    print(f"Observed color pairs:     {observed_pairs} / {n_pairs} ({density:.1%} of all pairs)")

    baseline = PopularityBaseline(vocabulary, co_occurrence)
    print()
    print("Most popular vocabulary colors (popularity baseline):")
    for rec in baseline.recommend(top_n=10):
        print(
            f"  {rec.hex}  in {rec.supporting_artworks:>4d}/{co_occurrence.n_artworks} "
            f"paintings ({rec.score:.1%})"
        )

    print()
    print(f"Saved: {vocab_path}")
    print(f"Saved: {co_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_DIR / "palettes.jsonl")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    train(input_path=args.input, vocab_size=args.vocab_size, models_dir=args.models_dir)


if __name__ == "__main__":
    main()
