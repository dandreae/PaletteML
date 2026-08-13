"""Ranking metrics for the held-out color-prediction evaluation, and
the orchestration loop that scores one recommender against a fixed
list of evaluation cases (evaluation/cases.py).

Metrics, in plain English:

  Hit Rate @ K
    Across all evaluation cases, what fraction had the true held-out
    color somewhere in the top K recommendations? A single case
    scores 1 (found within the top K) or 0 (didn't), and the metric
    is just the average of those 0/1 scores. Answers "if a user only
    ever looks at the top K suggestions, how often would the right
    color actually be there?"

  MRR (Mean Reciprocal Rank)
    Per case: 1 / rank if the held-out color was found anywhere in the
    ranking (rank 1 -> 1.0, rank 2 -> 0.5, rank 4 -> 0.25), else 0.
    Averaged across all cases. Unlike Hit@K, MRR doesn't need one
    arbitrary cutoff — it rewards getting the right color near the top
    more than getting it into the list at all, and a single number
    summarizes ranking quality across the whole list.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from paletteml.evaluation.cases import EvalCase

# (seed_cluster_ids, top_n) -> ranked candidate cluster_ids, best first
RankFn = Callable[[list[int], int], list[int]]


@dataclass(frozen=True)
class EvalCaseResult:
    """One case's outcome: the full ranking produced, and where (if anywhere) the target landed."""

    case: EvalCase
    ranked_candidates: list[int]
    rank: int | None  # 1-based rank of the hidden color, or None if not found at all
    reciprocal_rank: float

    def hit_at(self, k: int) -> bool:
        return self.rank is not None and self.rank <= k


@dataclass
class EvaluationResult:
    name: str
    n_cases: int
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float
    mrr: float
    case_results: list[EvalCaseResult] = field(default_factory=list)


def hit_rate_at_k(ranks: list[int | None], k: int) -> float:
    """Fraction of cases whose true rank is <= k. `None` means never found."""
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """Average of 1/rank per case (0.0 for cases where rank is None)."""
    if not ranks:
        return 0.0
    return sum((1.0 / r) if r is not None else 0.0 for r in ranks) / len(ranks)


def evaluate_recommender(
    name: str,
    rank_fn: RankFn,
    cases: list[EvalCase],
    full_rank_top_n: int,
) -> EvaluationResult:
    """Run `rank_fn` over every case and compute Hit@1/3/5 + MRR.

    `rank_fn` is called once per case with a generous `full_rank_top_n`
    (the caller passes the vocabulary size) so Hit@1/3/5 and MRR can
    all be read off one ranking per case — each metric just looks
    further down the same list rather than re-querying the recommender
    per K. A recommender is free to return fewer candidates than
    requested (the co-occurrence recommender does exactly this when it
    has no positive evidence for more candidates) — that shorter list
    is scored as-is, since it's exactly what a real user would see.
    """
    case_results = []
    for case in cases:
        ranked = rank_fn(list(case.seed_cluster_ids), full_rank_top_n)
        try:
            rank = ranked.index(case.hidden_cluster_id) + 1
        except ValueError:
            rank = None
        reciprocal_rank = (1.0 / rank) if rank is not None else 0.0
        case_results.append(
            EvalCaseResult(case=case, ranked_candidates=ranked, rank=rank, reciprocal_rank=reciprocal_rank)
        )

    ranks = [cr.rank for cr in case_results]
    return EvaluationResult(
        name=name,
        n_cases=len(cases),
        hit_rate_at_1=hit_rate_at_k(ranks, 1),
        hit_rate_at_3=hit_rate_at_k(ranks, 3),
        hit_rate_at_5=hit_rate_at_k(ranks, 5),
        mrr=mean_reciprocal_rank(ranks),
        case_results=case_results,
    )
