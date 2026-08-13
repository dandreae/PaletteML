"""Trivial popularity baseline.

Recommends whichever vocabulary colors appear in the most training
paintings, entirely ignoring any seed color. This exists purely as a
simple point of comparison for the learned co-occurrence recommender
during evaluation (next stage) — it is not used by
CoOccurrenceRecommender and doesn't share any code path with it
beyond reading the same already-computed color_counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paletteml.modeling.co_occurrence import CoOccurrenceModel
from paletteml.modeling.vocabulary import ColorVocabulary


@dataclass(frozen=True)
class PopularityRecommendation:
    hex: str
    lab: tuple[float, float, float]
    cluster_id: int
    score: float  # fraction of training paintings containing this color
    supporting_artworks: int


class PopularityBaseline:
    """Ranks vocabulary colors by raw popularity across training paintings."""

    def __init__(self, vocabulary: ColorVocabulary, co_occurrence: CoOccurrenceModel):
        if vocabulary.size != co_occurrence.vocab_size:
            raise ValueError(
                f"vocabulary size ({vocabulary.size}) does not match "
                f"co_occurrence vocab_size ({co_occurrence.vocab_size})"
            )
        self.vocabulary = vocabulary
        self.co_occurrence = co_occurrence

    def recommend(self, top_n: int = 5) -> list[PopularityRecommendation]:
        order = np.argsort(-self.co_occurrence.color_counts)
        n_artworks = self.co_occurrence.n_artworks

        results = []
        for cluster_id in order[:top_n]:
            cluster_id = int(cluster_id)
            entry = self.vocabulary.entries[cluster_id]
            count = int(self.co_occurrence.color_counts[cluster_id])
            results.append(
                PopularityRecommendation(
                    hex=entry.hex,
                    lab=entry.lab,
                    cluster_id=cluster_id,
                    score=(count / n_artworks) if n_artworks else 0.0,
                    supporting_artworks=count,
                )
            )
        return results
