# PaletteML Evaluation Report

Generated 2026-08-13T09:56:11.421633+00:00

## Dataset & split

- Total processed artworks: 300
- Train: 240 artworks · Test: 60 artworks
  (test_fraction=0.2, random_state=42)
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

At vocab_size=64: 60 test paintings considered,
0 skipped (single color), 0
candidate hides skipped (unseen in training) → **293 evaluation cases**.

## Metrics, in plain English

- **Hit Rate @ K** — fraction of cases where the true hidden color appeared
  somewhere in the top K recommendations.
- **MRR** (Mean Reciprocal Rank) — average of 1/rank (0 if never found); rewards
  ranking the right color *near the top*, not just getting it onto the list.

## Headline comparison (vocab_size=64)

| Metric | Co-occurrence | Popularity | Random |
|---|---:|---:|---:|
| Hit Rate @1 | 0.020 | 0.051 | 0.034 |
| Hit Rate @3 | 0.119 | 0.109 | 0.058 |
| Hit Rate @5 | 0.174 | 0.171 | 0.099 |
| MRR | 0.122 | 0.139 | 0.095 |

n_cases = 293.

**vs. popularity:** co-occurrence nominally beats popularity (0.174 vs 0.171) but the gap is **not statistically significant** (McNemar χ²=0.00, 40 cases co-occurrence won alone vs 39 popularity won alone) — consistent with noise, not a real effect.

**vs. random:** co-occurrence beats random, and the gap is statistically significant (McNemar p<0.05) (0.174 vs 0.099 Hit@5).

![Recommender comparison](figures/evaluation_comparison.png)

## Vocabulary size sweep

Same train/test split, same methodology, vocab_size varied:

| Vocab size | N cases | CoOcc H@1 | CoOcc H@3 | CoOcc H@5 | CoOcc MRR | Pop H@5 | Rand H@5 | Margin |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 280 | 0.036 | 0.168 | 0.304 | 0.177 | 0.382 | 0.232 | -0.079 |
| 48 | 286 | 0.035 | 0.147 | 0.213 | 0.147 | 0.217 | 0.098 | -0.003 |
| 64 | 293 | 0.020 | 0.119 | 0.174 | 0.122 | 0.171 | 0.099 | +0.003 |
| 96 | 292 | 0.041 | 0.092 | 0.154 | 0.120 | 0.127 | 0.062 | +0.027 |

Margin = CoOcc H@5 − Popularity H@5. Negative means **popularity outperforms the
learned model** at that vocab size.

**Best vocab size (by co-occurrence Hit@5 in isolation, tie-broken by MRR): 32.**
This was not assumed — the original choice of 64 came from a sizing heuristic
(samples-per-cluster), not from measuring hit rate.

⚠️ **Caveat: 'best' here is misleading on its own.** Popularity actually outperforms co-occurrence at vocab_size 32, 48. A higher absolute Hit@5 for co-occurrence does not mean co-occurrence is winning the comparison that matters — see the margin column above.

## Failure case analysis

**Hit@5 by how common the hidden color was in training:**

| Training occurrences | N cases | Hit@5 |
|---|---:|---:|
| 1-4 | 3 | 0.333 |
| 5-19 | 103 | 0.223 |
| 20+ | 187 | 0.144 |

**Sample successes** (co-occurrence found the hidden color in its top 5):
- `artic:11`: seeds=[#1a1612, #452416, #6f5335] hidden=`#2b2016` rank=5 top5=[#b59e7b, #9f3a26, #7f311c, #875039, #2b2016]
- `artic:11`: seeds=[#2b2016, #452416, #6f5335] hidden=`#1a1612` rank=3 top5=[#7f311c, #633d16, #1a1612, #9f3a26, #8e5628]
- `artic:11`: seeds=[#2b2016, #1a1612, #6f5335] hidden=`#452416` rank=1 top5=[#452416, #7f311c, #463729, #988163, #b59e7b]

**Sample failures** (hidden color not found anywhere in co-occurrence's ranking):
- `artic:87479`: seeds=[#b59e7b, #78664c, #463729, #452416] hidden=`#43687d` rank=not found top5=[#1a1612, #dbc582, #9f3a26, #7f311c, #875039]
- `artic:20199`: seeds=[#9a95a9, #733a2e, #717656, #afa28c] hidden=`#394c51` rank=not found top5=[#9a756f, #48622a, #21337d, #7384a9, #9b7632]
- `artic:20199`: seeds=[#394c51, #733a2e, #717656, #afa28c] hidden=`#9a95a9` rank=not found top5=[#48622a, #4c7158, #2e4e6e, #3e5141, #21337d]

**Cases where popularity beat co-occurrence at Hit@5** (39 total):
- `artic:20684`: seeds=[#c8c6b5, #394c51, #98a1a2, #213344] hidden=`#7a7466` rank=41 top5=[#688a8a, #9a756f, #566a71, #9f3a26, #43687d]
- `artic:9`: seeds=[#633d16, #84643c, #9d8250, #352918] hidden=`#1a1612` rank=27 top5=[#513b22, #584519, #8e5628, #c49a62, #cfbc99]
- `artic:87479`: seeds=[#b59e7b, #43687d, #78664c, #452416] hidden=`#463729` rank=32 top5=[#213344, #688a8a, #9f3a26, #7f311c, #875039]

## Honest conclusion

**Does the learned model beat random?** Yes, clearly — co-occurrence's Hit@5
(0.174) is well above random's (0.099) at every vocab
size tested, and that gap is large enough to not be noise. The model has learned
*something* about which colors real paintings combine.

**Does the learned model beat the popularity baseline?** This is much less clear,
and depends heavily on vocab_size:
- At vocab_size=32 and 48, **popularity wins outright** (e.g. 0.382
  vs 0.304 Hit@5 at vocab_size=32).
- At vocab_size=64 (the working default), co-occurrence is nominally ahead but the
  gap is not statistically distinguishable from zero on this test set (McNemar
  χ²=0.00, 293 cases).
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

This is a small, single-split evaluation on a modest dataset (300
artworks, 293 leave-one-out cases). Treat every number above as "does this
look promising enough to keep building on," not as a validated production result.
