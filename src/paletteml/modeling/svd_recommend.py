"""Palette recommendation via cosine similarity in the SVD color embedding.

Same seed-resolution and multi-seed-combination approach as the direct
co-occurrence recommender (modeling/recommend.py) — average the score
across seeds, always exclude the seed colors themselves — but
candidates are scored by cosine similarity in the learned embedding
(modeling/embedding.py) instead of by reading PPMI off the
co-occurrence matrix directly. See embedding.py's module docstring for
why that can differ, in particular: this recommender never filters out
candidates with no direct co-occurrence evidence, so it always returns
a full top_n ranking (the direct recommender can legitimately return
fewer results, even zero).
"""

from __future__ import annotations

from dataclasses import dataclass

from paletteml.color.space import hex_to_lab
from paletteml.modeling.embedding import ColorEmbedding
from paletteml.modeling.vocabulary import ColorVocabulary


@dataclass(frozen=True)
class SvdSeedEvidence:
    """Diagnostic detail for how one seed color contributed to a recommendation's score."""

    seed_hex: str
    seed_cluster_id: int
    cosine_similarity: float


@dataclass(frozen=True)
class SvdRecommendation:
    """One recommended color, with enough detail to see why it ranked highly."""

    hex: str
    lab: tuple[float, float, float]
    cluster_id: int
    score: float  # mean cosine similarity across seeds
    evidence: list[SvdSeedEvidence]


class SvdEmbeddingRecommender:
    """Recommends companion colors using cosine similarity in a learned SVD embedding."""

    def __init__(self, vocabulary: ColorVocabulary, embedding: ColorEmbedding):
        if vocabulary.size != embedding.vocab_size:
            raise ValueError(
                f"vocabulary size ({vocabulary.size}) does not match "
                f"embedding vocab_size ({embedding.vocab_size})"
            )
        self.vocabulary = vocabulary
        self.embedding = embedding

    def recommend(self, seed_colors: str | list[str], top_n: int = 5) -> list[SvdRecommendation]:
        """Recommend up to `top_n` companion colors for one or more seed hex colors.

        Scoring: cosine similarity between each candidate's embedding
        and each seed's embedding, averaged across seeds — identical
        combination rule to the direct co-occurrence recommender, for
        a like-for-like comparison between the two.
        """
        if isinstance(seed_colors, str):
            seed_colors = [seed_colors]
        if not seed_colors:
            raise ValueError("recommend() requires at least one seed color")

        seed_cluster_ids = sorted({self.vocabulary.assign(hex_to_lab(h)) for h in seed_colors})
        seed_hex_by_cluster: dict[int, str] = {}
        for h in seed_colors:
            cid = self.vocabulary.assign(hex_to_lab(h))
            seed_hex_by_cluster.setdefault(cid, h)

        return self.recommend_from_cluster_ids(
            seed_cluster_ids, top_n=top_n, seed_hex_by_cluster=seed_hex_by_cluster
        )

    def recommend_from_cluster_ids(
        self,
        seed_cluster_ids: list[int],
        top_n: int = 5,
        seed_hex_by_cluster: dict[int, str] | None = None,
    ) -> list[SvdRecommendation]:
        """Rank candidates directly from vocabulary cluster ids.

        Used internally by recommend(), and directly by the evaluation
        harness (evaluation/adapters.py) for the same reason
        CoOccurrenceRecommender exposes this: cases already have
        cluster ids from encode_palette(), so this skips a lossy
        cluster_id -> hex -> cluster_id round trip.
        """
        if not seed_cluster_ids:
            raise ValueError("recommend_from_cluster_ids() requires at least one seed cluster id")
        seed_cluster_ids = sorted(set(seed_cluster_ids))
        if seed_hex_by_cluster is None:
            seed_hex_by_cluster = {cid: self.vocabulary.entries[cid].hex for cid in seed_cluster_ids}

        scored: list[tuple[int, float, list[SvdSeedEvidence]]] = []
        for candidate_id in range(self.vocabulary.size):
            if candidate_id in seed_cluster_ids:
                continue

            evidence = []
            similarities = []
            for seed_id in seed_cluster_ids:
                sim = self.embedding.similarity(seed_id, candidate_id)
                similarities.append(sim)
                evidence.append(
                    SvdSeedEvidence(
                        seed_hex=seed_hex_by_cluster[seed_id],
                        seed_cluster_id=seed_id,
                        cosine_similarity=sim,
                    )
                )

            score = sum(similarities) / len(similarities)
            scored.append((candidate_id, score, evidence))

        scored.sort(key=lambda item: item[1], reverse=True)

        recommendations = []
        for candidate_id, score, evidence in scored[:top_n]:
            entry = self.vocabulary.entries[candidate_id]
            recommendations.append(
                SvdRecommendation(
                    hex=entry.hex, lab=entry.lab, cluster_id=candidate_id, score=score, evidence=evidence
                )
            )
        return recommendations
