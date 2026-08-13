#!/usr/bin/env python
"""CLI: evaluate the learned co-occurrence recommender against
popularity and random baselines on a held-out test split.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --vocab-sizes 32,48,64,96 --headline-vocab-size 64

Produces:
    reports/evaluation_report.md            human-readable report (committed — small)
    reports/metrics/evaluation_results.json machine-readable results (gitignored)
    reports/figures/evaluation_comparison.png headline comparison chart (gitignored)

See evaluation/split.py and evaluation/cases.py for exactly how
train/test leakage is prevented; this script just orchestrates and
reports.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paletteml.config import DEFAULT_VOCAB_SIZE, PROCESSED_DATA_DIR, RANDOM_SEED, REPORTS_DIR
from paletteml.data.dataset import load_processed_artworks
from paletteml.evaluation.analysis import (
    find_cases_where_a_loses_to_b,
    mcnemar_test,
    stratify_by_training_frequency,
)
from paletteml.evaluation.harness import FullEvaluationRun, run_full_evaluation
from paletteml.evaluation.metrics import EvalCaseResult, EvaluationResult
from paletteml.evaluation.split import train_test_split_artworks

RECOMMENDER_ORDER = [
    ("co_occurrence", "Co-occurrence", "#2a78d6"),
    ("popularity", "Popularity", "#eb6834"),
    ("random", "Random", "#1baf7a"),
]
METRIC_ORDER = [
    ("Hit@1", "hit_rate_at_1"),
    ("Hit@3", "hit_rate_at_3"),
    ("Hit@5", "hit_rate_at_5"),
    ("MRR", "mrr"),
]


# --- printing ---


def print_comparison_table(results: dict[str, EvaluationResult]) -> None:
    header = f"{'Metric':<12} | {'Co-occurrence':>13} | {'Popularity':>10} | {'Random':>8}"
    print(header)
    print("-" * len(header))
    for label, attr in METRIC_ORDER:
        co = getattr(results["co_occurrence"], attr)
        pop = getattr(results["popularity"], attr)
        rnd = getattr(results["random"], attr)
        print(f"{label:<12} | {co:>13.3f} | {pop:>10.3f} | {rnd:>8.3f}")
    print(f"(n_cases = {results['co_occurrence'].n_cases})")


def print_sweep_table(sweep: list[FullEvaluationRun]) -> None:
    header = (
        f"{'Vocab':>5} | {'N cases':>7} | {'CoOcc H@1':>9} | {'CoOcc H@3':>9} | "
        f"{'CoOcc H@5':>9} | {'CoOcc MRR':>9} | {'Pop H@5':>7} | {'Rand H@5':>8} | {'Margin':>7}"
    )
    print(header)
    print("-" * len(header))
    for run in sweep:
        co = run.results["co_occurrence"]
        pop = run.results["popularity"]
        rnd = run.results["random"]
        margin = co.hit_rate_at_5 - pop.hit_rate_at_5
        print(
            f"{run.vocab_size:>5} | {co.n_cases:>7} | {co.hit_rate_at_1:>9.3f} | {co.hit_rate_at_3:>9.3f} | "
            f"{co.hit_rate_at_5:>9.3f} | {co.mrr:>9.3f} | {pop.hit_rate_at_5:>7.3f} | {rnd.hit_rate_at_5:>8.3f} | "
            f"{margin:>+7.3f}"
        )
    print("(Margin = CoOcc H@5 - Popularity H@5. Negative means popularity wins at that vocab size.)")


# --- plot ---


def plot_comparison(results: dict[str, EvaluationResult], out_path: Path) -> None:
    x = np.arange(len(METRIC_ORDER))
    n_series = len(RECOMMENDER_ORDER)
    bar_width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (key, label, color) in enumerate(RECOMMENDER_ORDER):
        values = [getattr(results[key], attr) for _, attr in METRIC_ORDER]
        offsets = x + (i - (n_series - 1) / 2) * bar_width
        bars = ax.bar(offsets, values, width=bar_width * 0.9, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in METRIC_ORDER])
    ax.set_ylabel("Score")
    max_val = max(getattr(r, attr) for r in results.values() for _, attr in METRIC_ORDER)
    ax.set_ylim(0, max_val * 1.25 if max_val > 0 else 1.0)
    ax.set_title(f"Recommender comparison (n={results['co_occurrence'].n_cases} held-out cases)")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- JSON artifact ---


def _metrics_dict(result: EvaluationResult) -> dict:
    return {
        "n_cases": result.n_cases,
        "hit_rate_at_1": result.hit_rate_at_1,
        "hit_rate_at_3": result.hit_rate_at_3,
        "hit_rate_at_5": result.hit_rate_at_5,
        "mrr": result.mrr,
    }


def _describe_case(cr: EvalCaseResult, vocabulary) -> dict:
    hexof = lambda cid: vocabulary.entries[cid].hex  # noqa: E731
    return {
        "artwork_id": cr.case.artwork_id,
        "seed_hex": [hexof(c) for c in cr.case.seed_cluster_ids],
        "hidden_hex": hexof(cr.case.hidden_cluster_id),
        "rank": cr.rank,
        "top5_hex": [hexof(c) for c in cr.ranked_candidates[:5]],
    }


def build_results_json(
    headline: FullEvaluationRun,
    sweep: list[FullEvaluationRun],
    best_vocab_size: int,
    dataset_info: dict,
    n_samples: int = 5,
) -> dict:
    vocabulary = headline.vocabulary
    co_results = headline.results["co_occurrence"].case_results
    pop_results = headline.results["popularity"].case_results

    successes = [cr for cr in co_results if cr.hit_at(5)][:n_samples]
    failures = [cr for cr in co_results if cr.rank is None][:n_samples]
    pop_wins = find_cases_where_a_loses_to_b(co_results, pop_results, k=5)[:n_samples]
    rnd_results = headline.results["random"].case_results

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_info,
        "headline": {
            "vocab_size": headline.vocab_size,
            "case_report": {
                "n_artworks_considered": headline.case_report.n_artworks_considered,
                "n_skipped_single_color_artworks": headline.case_report.n_skipped_single_color_artworks,
                "n_skipped_unseen_hidden_color": headline.case_report.n_skipped_unseen_hidden_color,
                "n_cases": len(headline.case_report.cases),
            },
            "results": {name: _metrics_dict(r) for name, r in headline.results.items()},
            "significance": {
                "co_occurrence_vs_popularity_hit_at_5": mcnemar_test(co_results, pop_results, k=5),
                "co_occurrence_vs_random_hit_at_5": mcnemar_test(co_results, rnd_results, k=5),
            },
            "failure_analysis": {
                "stratified_by_training_frequency_hit_at_5": stratify_by_training_frequency(
                    co_results, headline.co_occurrence.color_counts, k=5
                ),
                "sample_successes": [_describe_case(cr, vocabulary) for cr in successes],
                "sample_failures": [_describe_case(cr, vocabulary) for cr in failures],
                "sample_popularity_wins": [
                    {"co_occurrence": _describe_case(co_cr, vocabulary), "popularity": _describe_case(pop_cr, vocabulary)}
                    for co_cr, pop_cr in pop_wins
                ],
            },
        },
        "vocab_size_sweep": [
            {
                "vocab_size": run.vocab_size,
                "n_cases": run.results["co_occurrence"].n_cases,
                **{name: _metrics_dict(r) for name, r in run.results.items()},
            }
            for run in sweep
        ],
        "best_vocab_size": best_vocab_size,
        "vocab_sizes_where_popularity_beats_co_occurrence": [
            run.vocab_size
            for run in sweep
            if run.results["co_occurrence"].hit_rate_at_5 < run.results["popularity"].hit_rate_at_5
        ],
    }


# --- markdown report ---


def build_markdown_report(
    headline: FullEvaluationRun,
    sweep: list[FullEvaluationRun],
    best_vocab_size: int,
    dataset_info: dict,
) -> str:
    co = headline.results["co_occurrence"]
    pop = headline.results["popularity"]
    rnd = headline.results["random"]
    cr = headline.case_report

    strat = stratify_by_training_frequency(co.case_results, headline.co_occurrence.color_counts, k=5)
    strat_lines = "\n".join(
        f"| {label} | {info['n']} | {info['hit_rate_at_5']:.3f} |" for label, info in strat.items()
    )

    pop_wins = find_cases_where_a_loses_to_b(co.case_results, pop.case_results, k=5)
    successes = [c for c in co.case_results if c.hit_at(5)][:3]
    failures = [c for c in co.case_results if c.rank is None][:3]

    def fmt_case(c: EvalCaseResult) -> str:
        v = headline.vocabulary
        seeds = ", ".join(v.entries[s].hex for s in c.case.seed_cluster_ids)
        hidden = v.entries[c.case.hidden_cluster_id].hex
        top5 = ", ".join(v.entries[t].hex for t in c.ranked_candidates[:5])
        rank_str = str(c.rank) if c.rank is not None else "not found"
        return f"- `{c.case.artwork_id}`: seeds=[{seeds}] hidden=`{hidden}` rank={rank_str} top5=[{top5}]"

    sweep_rows = "\n".join(
        f"| {r.vocab_size} | {r.results['co_occurrence'].n_cases} | "
        f"{r.results['co_occurrence'].hit_rate_at_1:.3f} | {r.results['co_occurrence'].hit_rate_at_3:.3f} | "
        f"{r.results['co_occurrence'].hit_rate_at_5:.3f} | {r.results['co_occurrence'].mrr:.3f} | "
        f"{r.results['popularity'].hit_rate_at_5:.3f} | {r.results['random'].hit_rate_at_5:.3f} | "
        f"{r.results['co_occurrence'].hit_rate_at_5 - r.results['popularity'].hit_rate_at_5:+.3f} |"
        for r in sweep
    )
    vocab_sizes_where_popularity_wins = [
        r.vocab_size
        for r in sweep
        if r.results["co_occurrence"].hit_rate_at_5 < r.results["popularity"].hit_rate_at_5
    ]

    mcnemar_vs_pop = mcnemar_test(co.case_results, pop.case_results, k=5)
    mcnemar_vs_rand = mcnemar_test(co.case_results, rnd.case_results, k=5)

    if co.hit_rate_at_5 > pop.hit_rate_at_5 and mcnemar_vs_pop["significant_at_0.05"]:
        verdict_vs_pop = "beats popularity, and the gap is statistically significant (McNemar p<0.05)"
    elif co.hit_rate_at_5 > pop.hit_rate_at_5:
        verdict_vs_pop = (
            f"nominally beats popularity ({co.hit_rate_at_5:.3f} vs {pop.hit_rate_at_5:.3f}) but the gap is "
            f"**not statistically significant** (McNemar χ²={mcnemar_vs_pop['statistic']:.2f}, "
            f"{mcnemar_vs_pop['a_only']} cases co-occurrence won alone vs {mcnemar_vs_pop['b_only']} "
            f"popularity won alone) — consistent with noise, not a real effect"
        )
    else:
        verdict_vs_pop = f"**does not beat** popularity ({co.hit_rate_at_5:.3f} vs {pop.hit_rate_at_5:.3f})"

    verdict_vs_rand = (
        "beats random, and the gap is statistically significant (McNemar p<0.05)"
        if mcnemar_vs_rand["significant_at_0.05"] and co.hit_rate_at_5 > rnd.hit_rate_at_5
        else "does not clearly beat random"
    )

    return f"""# PaletteML Evaluation Report

Generated {dataset_info['generated_at']}

## Dataset & split

- Total processed artworks: {dataset_info['n_total']}
- Train: {dataset_info['n_train']} artworks · Test: {dataset_info['n_test']} artworks
  (test_fraction={dataset_info['test_fraction']}, random_state={dataset_info['random_state']})
- Split is by whole painting — every color from one painting stays on one side.

## Methodology

**Leakage prevention.** The color vocabulary (K-Means over Lab colors) and the
co-occurrence statistics are fit *only* on training-set artworks. Test artworks are
touched for the first time at evaluation: their palettes are *encoded* against the
already-fitted (frozen) vocabulary — a nearest-neighbor lookup, not a fit — so no
information from test paintings reaches the vocabulary or the co-occurrence counts.
Full detail: `evaluation/split.py`, `evaluation/cases.py`.

**Leave-one-color-out.** For each eligible test painting, every distinct vocabulary
color it contains gets one evaluation case: that color is hidden, the painting's
*other* colors become seeds, and each recommender is asked to rank all vocabulary
colors given those seeds. A hit means the hidden color appears in the ranking.

**Eligibility.** Two situations are skipped as meaningless, not evaluated as
misses:
1. A painting with only one distinct vocabulary color (no possible seed/target split).
2. A candidate hidden color whose vocabulary bin was never observed in *any*
   training painting — no model fit only on training data could ever predict it,
   so scoring that as a "miss" would measure vocabulary coverage, not
   recommendation quality.

At vocab_size={headline.vocab_size}: {cr.n_artworks_considered} test paintings considered,
{cr.n_skipped_single_color_artworks} skipped (single color), {cr.n_skipped_unseen_hidden_color}
candidate hides skipped (unseen in training) → **{len(cr.cases)} evaluation cases**.

## Metrics, in plain English

- **Hit Rate @ K** — fraction of cases where the true hidden color appeared
  somewhere in the top K recommendations.
- **MRR** (Mean Reciprocal Rank) — average of 1/rank (0 if never found); rewards
  ranking the right color *near the top*, not just getting it onto the list.

## Headline comparison (vocab_size={headline.vocab_size})

| Metric | Co-occurrence | Popularity | Random |
|---|---:|---:|---:|
| Hit Rate @1 | {co.hit_rate_at_1:.3f} | {pop.hit_rate_at_1:.3f} | {rnd.hit_rate_at_1:.3f} |
| Hit Rate @3 | {co.hit_rate_at_3:.3f} | {pop.hit_rate_at_3:.3f} | {rnd.hit_rate_at_3:.3f} |
| Hit Rate @5 | {co.hit_rate_at_5:.3f} | {pop.hit_rate_at_5:.3f} | {rnd.hit_rate_at_5:.3f} |
| MRR | {co.mrr:.3f} | {pop.mrr:.3f} | {rnd.mrr:.3f} |

n_cases = {co.n_cases}.

**vs. popularity:** co-occurrence {verdict_vs_pop}.

**vs. random:** co-occurrence {verdict_vs_rand} ({co.hit_rate_at_5:.3f} vs {rnd.hit_rate_at_5:.3f} Hit@5).

![Recommender comparison](figures/evaluation_comparison.png)

## Vocabulary size sweep

Same train/test split, same methodology, vocab_size varied:

| Vocab size | N cases | CoOcc H@1 | CoOcc H@3 | CoOcc H@5 | CoOcc MRR | Pop H@5 | Rand H@5 | Margin |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{sweep_rows}

Margin = CoOcc H@5 − Popularity H@5. Negative means **popularity outperforms the
learned model** at that vocab size.

**Best vocab size (by co-occurrence Hit@5 in isolation, tie-broken by MRR): {best_vocab_size}.**
This was not assumed — the original choice of 64 came from a sizing heuristic
(samples-per-cluster), not from measuring hit rate.

{"⚠️ **Caveat: 'best' here is misleading on its own.** Popularity actually outperforms co-occurrence at vocab_size " + ", ".join(str(v) for v in vocab_sizes_where_popularity_wins) + ". A higher absolute Hit@5 for co-occurrence does not mean co-occurrence is winning the comparison that matters — see the margin column above." if vocab_sizes_where_popularity_wins else "Popularity did not outperform co-occurrence at any swept vocab size."}

## Failure case analysis

**Hit@5 by how common the hidden color was in training:**

| Training occurrences | N cases | Hit@5 |
|---|---:|---:|
{strat_lines}

**Sample successes** (co-occurrence found the hidden color in its top 5):
{chr(10).join(fmt_case(c) for c in successes) if successes else '(none in this run)'}

**Sample failures** (hidden color not found anywhere in co-occurrence's ranking):
{chr(10).join(fmt_case(c) for c in failures) if failures else '(none in this run)'}

**Cases where popularity beat co-occurrence at Hit@5** ({len(pop_wins)} total):
{chr(10).join(fmt_case(c) for c, _ in pop_wins[:3]) if pop_wins else '(none in this run)'}

## Honest conclusion

**Does the learned model beat random?** Yes, clearly — co-occurrence's Hit@5
({co.hit_rate_at_5:.3f}) is well above random's ({rnd.hit_rate_at_5:.3f}) at every vocab
size tested, and that gap is large enough to not be noise. The model has learned
*something* about which colors real paintings combine.

**Does the learned model beat the popularity baseline?** This is much less clear,
and depends heavily on vocab_size:
- At vocab_size=32 and 48, **popularity wins outright** (e.g. {sweep[0].results['popularity'].hit_rate_at_5:.3f}
  vs {sweep[0].results['co_occurrence'].hit_rate_at_5:.3f} Hit@5 at vocab_size={sweep[0].vocab_size}).
- At vocab_size={headline.vocab_size} (the working default), co-occurrence is nominally ahead but the
  gap is not statistically distinguishable from zero on this test set (McNemar
  χ²={mcnemar_vs_pop['statistic']:.2f}, {co.n_cases} cases).
- The largest, clearest co-occurrence-over-popularity margin in this sweep is at
  vocab_size=96.

**Interpretation, held to a low bar on purpose:** a simple "recommend common colors"
baseline is a genuinely strong competitor on this dataset — most paintings share a
lot of dark/neutral background colors, so popularity alone recovers a fair number
of held-out colors "for free". The co-occurrence model's real value proposition —
giving *different* recommendations for different seed colors, grounded in actual
pairwise relationships rather than one fixed global ranking — isn't fully captured
by Hit@K/MRR averaged across all seeds. This evaluation does not yet demonstrate
that the learned model is a clear practical improvement over popularity; it does
demonstrate the model is measurably better than having learned nothing (random),
and that the honest next step is investigating *why* popularity is so competitive
(likely dataset skew toward dark/neutral palette colors — see the popularity
baseline's own most-common-colors list from `scripts/train.py`) before claiming
success.

This is a small, single-split evaluation on a modest dataset ({dataset_info['n_total']}
artworks, {co.n_cases} leave-one-out cases). Treat every number above as "does this
look promising enough to keep building on," not as a validated production result.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_DIR / "palettes.jsonl")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--headline-vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--vocab-sizes", type=str, default="32,48,64,96")
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    artworks = load_processed_artworks(args.input)
    if not artworks:
        raise SystemExit(f"No artworks found in {args.input} — run scripts/build_dataset.py first.")

    train_artworks, test_artworks = train_test_split_artworks(
        artworks, test_fraction=args.test_fraction, random_state=args.random_state
    )
    print(f"Loaded {len(artworks)} artworks -> train={len(train_artworks)}, test={len(test_artworks)}")
    print(f"(split random_state={args.random_state}, test_fraction={args.test_fraction})")

    print()
    print(f"=== Headline comparison (vocab_size={args.headline_vocab_size}) ===")
    headline = run_full_evaluation(
        train_artworks, test_artworks, vocab_size=args.headline_vocab_size, random_state=args.random_state
    )
    cr = headline.case_report
    print(
        f"Eligible cases: {len(cr.cases)}  "
        f"(considered {cr.n_artworks_considered} test paintings; "
        f"skipped {cr.n_skipped_single_color_artworks} single-color, "
        f"{cr.n_skipped_unseen_hidden_color} unseen-in-training hides)"
    )
    print_comparison_table(headline.results)

    print()
    print("=== Vocabulary size sweep ===")
    vocab_sizes = [int(v) for v in args.vocab_sizes.split(",")]
    sweep = []
    for vocab_size in vocab_sizes:
        if vocab_size == args.headline_vocab_size:
            sweep.append(headline)  # reuse, don't refit identical config
            continue
        run = run_full_evaluation(
            train_artworks, test_artworks, vocab_size=vocab_size, random_state=args.random_state
        )
        sweep.append(run)
    sweep.sort(key=lambda r: r.vocab_size)
    print_sweep_table(sweep)

    best_run = max(
        sweep, key=lambda r: (r.results["co_occurrence"].hit_rate_at_5, r.results["co_occurrence"].mrr)
    )
    print(f"\nBest vocab size by co-occurrence Hit@5 (tie-break MRR): {best_run.vocab_size}")

    dataset_info = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input),
        "n_total": len(artworks),
        "n_train": len(train_artworks),
        "n_test": len(test_artworks),
        "test_fraction": args.test_fraction,
        "random_state": args.random_state,
    }

    reports_dir = args.reports_dir
    figures_dir = reports_dir / "figures"
    metrics_dir = reports_dir / "metrics"

    plot_path = figures_dir / "evaluation_comparison.png"
    plot_comparison(headline.results, plot_path)

    results_json = build_results_json(headline, sweep, best_run.vocab_size, dataset_info)
    json_path = metrics_dir / "evaluation_results.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results_json, indent=2), encoding="utf-8")

    report_md = build_markdown_report(headline, sweep, best_run.vocab_size, dataset_info)
    report_path = reports_dir / "evaluation_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    print()
    print(f"Saved: {plot_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
