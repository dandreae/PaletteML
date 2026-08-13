# PaletteML Evaluation Report: SVD Embedding vs. PPMI Co-occurrence

Generated 2026-08-13T10:24:09.827627+00:00

## Dataset & split

- Total processed artworks: 300
- Train: 240 artworks · Test: 60 artworks
  (test_fraction=0.2, random_state=42)
- Split is by whole painting, unchanged from the previous evaluation stage.

## Methodology (unchanged from the previous stage, SVD is purely additive)

Same train/test split, same leave-one-color-out case construction, same eligibility
rules (see `evaluation/split.py`, `evaluation/cases.py`). The SVD embedding is fit
**only** on the same train-only co-occurrence model already used by the direct
recommender — it factorizes that model's PPMI matrix, so it sees no additional
information beyond what co-occurrence already had access to.

At vocab_size=64: 60 test paintings considered,
0 skipped (single color), 0
candidate hides skipped (unseen in training) → **293 evaluation cases**,
identical across all four recommenders below.

## Headline comparison (vocab_size=64, SVD dimension=16)

| Metric | Co-occurrence | SVD embedding | Popularity | Random |
|---|---:|---:|---:|---:|
| Hit Rate @1 | 0.020 | 0.065 | 0.051 | 0.034 |
| Hit Rate @3 | 0.119 | 0.154 | 0.109 | 0.058 |
| Hit Rate @5 | 0.174 | 0.225 | 0.171 | 0.099 |
| MRR | 0.122 | 0.170 | 0.139 | 0.095 |

n_cases = 293.

**SVD vs. co-occurrence:** SVD nominally beats co-occurrence (0.225 vs 0.174) but the gap is **not statistically significant** (McNemar χ²=2.42, 48 cases SVD won alone vs 33 co-occurrence won alone) — consistent with noise.

**SVD vs. popularity:** SVD nominally beats popularity (0.225 vs 0.171) but the gap is **not statistically significant** (McNemar χ²=2.68, 50 cases SVD won alone vs 34 popularity won alone) — consistent with noise.

**SVD vs. random:** SVD beats random, and the gap is statistically significant (McNemar p<0.05).

**Co-occurrence vs. popularity** (carried over from the previous stage, same split):
co-occurrence nominally beats popularity (0.174 vs 0.171) but the gap is **not statistically significant** (McNemar χ²=0.00, 40 cases co-occurrence won alone vs 39 popularity won alone) — consistent with noise.

![Recommender comparison](figures/evaluation_comparison.png)

## Embedding dimension sweep (fixed vocab_size=64)

Same cases, same co_occurrence model, only the SVD truncation dimension varies:

| Dimension | Hit@1 | Hit@3 | Hit@5 | MRR | Explained variance | vs. Co-occurrence H@5 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.048 | 0.099 | 0.215 | 0.145 | 0.605 | +0.041 |
| 16 | 0.065 | 0.154 | 0.225 | 0.170 | 0.785 | +0.051 |
| 32 | 0.058 | 0.143 | 0.205 | 0.157 | 0.947 | +0.031 |

"vs. Co-occurrence H@5" = SVD Hit@5 − Co-occurrence Hit@5 at this vocab_size (same
294-ish cases both times). Explained variance is the fraction of the PPMI matrix's
total squared-singular-value "energy" this many components keep — a diagnostic for
how much of the matrix's structure survives truncation, not a performance metric by
itself. Dimensions were **not** chosen to make any particular one win.

## Vocabulary size sweep (co-occurrence only, carried over from the previous stage)

| Vocab size | N cases | CoOcc H@5 | Pop H@5 | Margin |
|---:|---:|---:|---:|---:|
| 32 | 280 | 0.304 | 0.382 | -0.079 |
| 48 | 286 | 0.213 | 0.217 | -0.003 |
| 64 | 293 | 0.174 | 0.171 | +0.003 |
| 96 | 292 | 0.154 | 0.127 | +0.027 |

Best vocab size by co-occurrence Hit@5 (tie-break MRR): 32. Note this
was computed for co-occurrence only in the previous stage and was not re-run for SVD
at every vocab size in this stage — see "what this doesn't demonstrate" below.

## Failure case analysis

### Hit@5 by how common the hidden color was in training

| Training occurrences | N cases | Co-occurrence Hit@5 | SVD Hit@5 |
|---|---:|---:|---:|
| 1-4 | 3 | 0.333 | 0.000 |
| 5-19 | 103 | 0.223 | 0.214 |
| 20+ | 187 | 0.144 | 0.235 |

### Dark/neutral bias (the specific pattern flagged in the previous stage)

"dark_neutral" = hidden color has L < 40 and chroma < 20 (see `evaluation/analysis.py:is_dark_neutral`
for the exact, deliberately simple definition).

| Hidden color group | N cases | Co-occurrence H@5 | SVD H@5 | Popularity H@5 | Random H@5 |
|---|---:|---:|---:|---:|---:|
| dark_neutral | 103 | 0.204 | 0.252 | 0.427 | 0.097 |
| other | 190 | 0.158 | 0.211 | 0.032 | 0.100 |

### Sample cases

**Co-occurrence succeeded** (hit @5):
- `artic:11`: seeds=[#1a1612, #452416, #6f5335] hidden=`#2b2016` rank=5 top5=[#b59e7b, #9f3a26, #7f311c, #875039, #2b2016]
- `artic:11`: seeds=[#2b2016, #452416, #6f5335] hidden=`#1a1612` rank=3 top5=[#7f311c, #633d16, #1a1612, #9f3a26, #8e5628]
- `artic:11`: seeds=[#2b2016, #1a1612, #6f5335] hidden=`#452416` rank=1 top5=[#452416, #7f311c, #463729, #988163, #b59e7b]

**Co-occurrence failed** (not found anywhere in its ranking):
- `artic:87479`: seeds=[#b59e7b, #78664c, #463729, #452416] hidden=`#43687d` rank=not found top5=[#1a1612, #dbc582, #9f3a26, #7f311c, #875039]
- `artic:20199`: seeds=[#9a95a9, #733a2e, #717656, #afa28c] hidden=`#394c51` rank=not found top5=[#9a756f, #48622a, #21337d, #7384a9, #9b7632]
- `artic:20199`: seeds=[#394c51, #733a2e, #717656, #afa28c] hidden=`#9a95a9` rank=not found top5=[#48622a, #4c7158, #2e4e6e, #3e5141, #21337d]

**Popularity beat co-occurrence** (39 total, first 3):
- `artic:20684`: seeds=[#c8c6b5, #394c51, #98a1a2, #213344] hidden=`#7a7466` rank=41 top5=[#688a8a, #9a756f, #566a71, #9f3a26, #43687d]
- `artic:9`: seeds=[#633d16, #84643c, #9d8250, #352918] hidden=`#1a1612` rank=27 top5=[#513b22, #584519, #8e5628, #c49a62, #cfbc99]
- `artic:87479`: seeds=[#b59e7b, #43687d, #78664c, #452416] hidden=`#463729` rank=32 top5=[#213344, #688a8a, #9f3a26, #7f311c, #875039]

**Popularity beat SVD** (34 total, first 3):
- `artic:11`: seeds=[#2b2016, #452416, #6f5335] hidden=`#1a1612` rank=10 top5=[#b59e7b, #463729, #84643c, #51493a, #633d16]
- `artic:20684`: seeds=[#c8c6b5, #394c51, #98a1a2, #213344] hidden=`#7a7466` rank=17 top5=[#566a71, #43687d, #423942, #2e4e6e, #b2866a]
- `artic:9`: seeds=[#633d16, #84643c, #9d8250, #352918] hidden=`#1a1612` rank=18 top5=[#513b22, #5f563e, #2b2016, #463729, #6f5335]

**SVD beat co-occurrence** (48 total, first 3):
- `artic:20684`: seeds=[#394c51, #7a7466, #98a1a2, #213344] hidden=`#c8c6b5` rank=15 top5=[#566a71, #688a8a, #908e61, #9f3a26, #9a756f]
- `artic:20684`: seeds=[#c8c6b5, #394c51, #7a7466, #213344] hidden=`#98a1a2` rank=21 top5=[#688a8a, #43687d, #566a71, #48622a, #9f3a26]
- `artic:9`: seeds=[#633d16, #1a1612, #9d8250, #352918] hidden=`#84643c` rank=26 top5=[#452416, #8e5628, #584519, #513b22, #cfbc99]

**Co-occurrence beat SVD** (33 total, first 3):
- `artic:11`: seeds=[#2b2016, #452416, #6f5335] hidden=`#1a1612` rank=10 top5=[#b59e7b, #463729, #84643c, #51493a, #633d16]
- `artic:11`: seeds=[#2b2016, #1a1612, #6f5335] hidden=`#452416` rank=11 top5=[#463729, #b59e7b, #5f563e, #633d16, #84643c]
- `artic:14655`: seeds=[#875039, #98a1a2, #213344, #717656] hidden=`#566a71` rank=13 top5=[#c8c6b5, #4c7158, #3e5141, #423942, #688a8a]

## What this experiment actually demonstrates

**Headline numbers favor SVD, consistently, but not (yet) significantly.** SVD had
the best Hit@5 and MRR of all four recommenders at every tested embedding dimension
(8, 16, 32) — it wasn't a lucky one-off pick. But McNemar's
test on the same 293 cases puts SVD-vs-co-occurrence at χ²=2.42
and SVD-vs-popularity at χ²=2.68, both below the 3.841
significance threshold. That's notably closer to significance than co-occurrence's own
χ²=0.00 margin over popularity from the previous stage — SVD
looks like a real step up from a coin flip toward "probably better" — but "closer to
significant" is not the same as "significant," and this test set (293 cases from 60
paintings) is small enough that a few more paintings could shift this.

**The dark/neutral finding explains *why* popularity looked so competitive, and it's
the most useful result in this report.** Popularity's Hit@5 splits into
42.7% on dark/neutral hidden colors (103 cases) vs. only
3.2% on everything else (190 cases) — over a 13x gap. Popularity isn't
good at recommending relevant colors; it's good at guessing that a painting contains a
dark background, which most of them do. Both learned models are far more balanced:
co-occurrence goes 20.4% → 15.8%, SVD goes
25.2% → 21.1%. On the "other" (chromatic, non-background) group
specifically — arguably the group that actually matters for a palette-recommendation
product, since nobody needs help finding "add a dark background" — **SVD
(21.1%) and co-occurrence (15.8%) both beat popularity
(3.2%) by roughly 6-7x.** The random baseline shows no such split
(9.7% vs 10.0%), confirming this is a genuine
property of the dataset (dark/neutral colors are just common) and not an artifact of
the stratification itself. This reframes the headline Hit@5 comparison: popularity's
apparent competitiveness is concentrated almost entirely in a color category a real
palette tool wouldn't need much help recommending.

**Training-frequency pattern flipped for SVD, in the intuitively "correct" direction.**
Co-occurrence's Hit@5 got *worse* as the hidden color's training frequency increased
(0.223 at 5-19 occurrences → 0.144 at 20+,
103 and 187 cases respectively) — a pattern flagged as a puzzle in the
previous stage. SVD shows the opposite, more intuitive direction
(0.214 → 0.235): more training evidence, better
predictions. This is consistent with — though doesn't prove — the mechanism SVD is
supposed to provide: smoothing over individual noisy PPMI cells rather than reading
them literally. The smallest bucket (1-4 occurrences, only 3 cases) is too small
to read anything into either way.

**What this evaluation does NOT demonstrate:**
- Statistical significance of SVD's improvement — the numbers are directionally
  consistent and encouraging, not proven at conventional thresholds on this test set.
- Optimality of the tested embedding dimensions (8, 16, 32) — a
  reasonable range relative to vocab_size=64, not exhaustively searched
  (dimension 16 won here; that's one data point, not a tuned optimum).
- Interaction between vocab_size and embedding dimension — the dimension sweep only
  ran at vocab_size=64; a different vocab_size could favor SVD differently,
  and that wasn't checked (a real scope limitation, not an oversight to hide).

This remains a small, single-split evaluation on a modest dataset (300
artworks, 293 leave-one-out cases). The dark/neutral breakdown is the most
actionable finding here — it says more about what to fix next (the dataset's color
distribution and what "success" should even mean for a palette tool) than the
headline Hit@K numbers do on their own.
