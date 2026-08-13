"""Palette recommendation from learned color co-occurrence.

Given one or more seed colors, map each to its nearest vocabulary
color, then rank every other vocabulary color by how strongly the
co-occurrence model associates it with the seed(s) — a retrieval
grounded in counts from real paintings, not a color-wheel rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from paletteml.color.space import hex_to_lab
from paletteml.modeling.co_occurrence import CoOccurrenceModel
from paletteml.modeling.vocabulary import ColorVocabulary


@dataclass(frozen=True)
class SeedEvidence:
    """Diagnostic detail for how one seed color contributed to a recommendation's score."""

    seed_hex: str
    seed_cluster_id: int
    raw_co_occurrence: int
    conditional_probability: float
    ppmi: float


@dataclass(frozen=True)
class Recommendation:
    """One recommended color, with enough detail to see why it ranked highly."""

    hex: str
    lab: tuple[float, float, float]
    cluster_id: int
    score: float  # mean PPMI across seeds — see recommend() docstring
    supporting_artworks: int  # # training paintings this color appears in at all
    evidence: list[SeedEvidence]  # per-seed breakdown behind `score`


class CoOccurrenceRecommender:
    """Recommends companion colors using learned co-occurrence statistics."""

    def __init__(self, vocabulary: ColorVocabulary, co_occurrence: CoOccurrenceModel):
        if vocabulary.size != co_occurrence.vocab_size:
            raise ValueError(
                f"vocabulary size ({vocabulary.size}) does not match "
                f"co_occurrence vocab_size ({co_occurrence.vocab_size})"
            )
        self.vocabulary = vocabulary
        self.co_occurrence = co_occurrence

    def recommend(self, seed_colors: str | list[str], top_n: int = 5) -> list[Recommendation]:
        """Recommend up to `top_n` companion colors for one or more seed hex colors.

        Scoring: each candidate vocabulary color is scored against
        every seed independently with PPMI (see co_occurrence.py),
        then those per-seed scores are averaged. Averaging — rather
        than summing or requiring agreement on every seed (min) — is
        the simplest way to combine multiple seeds: it rewards a
        candidate that associates well with the seeds actually given,
        keeps the score on a comparable scale regardless of how many
        seeds were passed, and doesn't let one strong seed pairing get
        diluted to zero just because another seed has no observed
        history with the candidate.

        A candidate is only returned if its averaged score is > 0,
        i.e. it had *some* observed positive association with at
        least one seed — candidates with zero evidence for every seed
        are excluded rather than arbitrarily ordered by tie-breaking.
        Seed colors' own vocabulary bins are always excluded from
        results. This means recommend() can legitimately return fewer
        than top_n results (even zero) when the seed color's bin has
        little or no co-occurrence history — see README for why the
        popularity baseline exists as a fallback comparison for this.
        """
        if isinstance(seed_colors, str):
            seed_colors = [seed_colors]
        if not seed_colors:
            raise ValueError("recommend() requires at least one seed color")

        seed_cluster_ids = sorted({self.vocabulary.assign(hex_to_lab(h)) for h in seed_colors})
        # keep one representative seed hex per cluster id, for diagnostics
        seed_hex_by_cluster: dict[int, str] = {}
        for h in seed_colors:
            cid = self.vocabulary.assign(hex_to_lab(h))
            seed_hex_by_cluster.setdefault(cid, h)

        scored: list[tuple[int, float, list[SeedEvidence]]] = []
        for candidate_id in range(self.vocabulary.size):
            if candidate_id in seed_cluster_ids:
                continue

            evidence = []
            ppmi_scores = []
            for seed_id in seed_cluster_ids:
                ppmi = self.co_occurrence.ppmi(seed_id, candidate_id)
                ppmi_scores.append(ppmi)
                evidence.append(
                    SeedEvidence(
                        seed_hex=seed_hex_by_cluster[seed_id],
                        seed_cluster_id=seed_id,
                        raw_co_occurrence=self.co_occurrence.raw_count(seed_id, candidate_id),
                        conditional_probability=self.co_occurrence.conditional_probability(
                            seed_id, candidate_id
                        ),
                        ppmi=ppmi,
                    )
                )

            score = sum(ppmi_scores) / len(ppmi_scores)
            if score <= 0:
                continue
            scored.append((candidate_id, score, evidence))

        scored.sort(key=lambda item: item[1], reverse=True)

        recommendations = []
        for candidate_id, score, evidence in scored[:top_n]:
            entry = self.vocabulary.entries[candidate_id]
            recommendations.append(
                Recommendation(
                    hex=entry.hex,
                    lab=entry.lab,
                    cluster_id=candidate_id,
                    score=score,
                    supporting_artworks=int(self.co_occurrence.color_counts[candidate_id]),
                    evidence=evidence,
                )
            )
        return recommendations
