#!/usr/bin/env python
"""Dev CLI: get color recommendations from the trained co-occurrence model.

Not the API (that's a later stage) — a thin wrapper for manually
exercising the trained model from the command line.

Usage:
    python scripts/recommend.py "#b23a2f"
    python scripts/recommend.py "#b23a2f" "#2f6b8e" --top-n 8
    python scripts/recommend.py "#b23a2f" --baseline   # popularity baseline instead

Requires models/color_vocabulary.json and models/co_occurrence.json
(run scripts/train.py first).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from paletteml.config import MODELS_DIR
from paletteml.modeling.baseline import PopularityBaseline
from paletteml.modeling.co_occurrence import CoOccurrenceModel
from paletteml.modeling.recommend import CoOccurrenceRecommender
from paletteml.modeling.vocabulary import ColorVocabulary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seed_colors", nargs="+", help="One or more hex colors, e.g. #b23a2f")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument(
        "--baseline", action="store_true", help="Show the popularity baseline instead (ignores seed colors)"
    )
    args = parser.parse_args()

    vocabulary = ColorVocabulary.load(args.models_dir / "color_vocabulary.json")
    co_occurrence = CoOccurrenceModel.load(args.models_dir / "co_occurrence.json")

    if args.baseline:
        baseline = PopularityBaseline(vocabulary, co_occurrence)
        print(f"Popularity baseline (seed colors {args.seed_colors} ignored):")
        for rec in baseline.recommend(top_n=args.top_n):
            print(f"  {rec.hex}  score={rec.score:.3f}  supporting_artworks={rec.supporting_artworks}")
        return

    recommender = CoOccurrenceRecommender(vocabulary, co_occurrence)
    recommendations = recommender.recommend(args.seed_colors, top_n=args.top_n)

    print(f"Seeds: {', '.join(args.seed_colors)}")
    if not recommendations:
        print("  (no recommendations — insufficient co-occurrence evidence for this seed)")
    for rec in recommendations:
        print(f"  {rec.hex}  score={rec.score:.3f}  supporting_artworks={rec.supporting_artworks}")
        for ev in rec.evidence:
            print(
                f"      vs {ev.seed_hex}: raw_count={ev.raw_co_occurrence}  "
                f"cond_prob={ev.conditional_probability:.3f}  ppmi={ev.ppmi:.3f}"
            )


if __name__ == "__main__":
    main()
