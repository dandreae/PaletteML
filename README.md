# PaletteML

Learns color relationships from real paintings and recommends colors
that work well with a user-provided color or partial palette.

This is a machine-learning / data-science portfolio project, not an
LLM wrapper. No OpenAI/Anthropic/other LLM APIs are used anywhere in
the pipeline. The recommendation engine is a from-scratch model
trained on colors extracted from painting images.

## Idea

A color-wheel calculator can tell you the "complementary" of a hue
using geometry. PaletteML instead learns what colors actually get
combined by looking at thousands of real paintings, and recommends
based on that learned structure. It never sees color-theory rules —
only pixels.

## Architecture

```
Paintings (images)
      |
      v
Dominant-color extraction (K-Means clustering in CIELAB)      <- unsupervised ML
      |
      v
Painting -> palette table (N colors per painting)
      |
      v
Color relationship model:
  co-occurrence of colors across real palettes -> embedding      <- unsupervised ML
  (PMI-reweighted truncated SVD, word2vec-style)
  + clustering of the embedding into palette "archetypes"         <- unsupervised ML
      |
      v
Recommendation = nearest neighbors in the learned embedding space
      |
      v
FastAPI backend  --->  small static HTML/JS UI
      |
      v
Quantitative evaluation: held-out color prediction vs. baselines
      |
      v
Docker -> AWS
```

### What is actually the machine learning here

1. **Dominant-color extraction** — K-Means clustering of a painting's
   pixels in CIELAB space. Genuine unsupervised learning, run once
   per image, to turn a painting into a short list of (color, weight)
   pairs.
2. **Perceptual color space** — RGB is not perceptually uniform,
   so extraction and every distance calculation downstream operate
   in CIELAB (`skimage.color.rgb2lab`), where Euclidean distance
   approximates human-perceived difference (Delta E). This is the
   feature representation the rest of the pipeline depends on.
3. **Color-relationship modeling** — the core model. Build a
   color-by-color co-occurrence matrix from real palettes (quantized
   Lab bins), reweight it (PMI), and factorize it (truncated SVD) to
   get a learned embedding: colors that real paintings actually
   combine end up close together in this space. Cluster the
   embedding to find recurring palette archetypes.
4. **Recommendation** — map a user's seed color(s) into the learned
   embedding and retrieve nearest neighbors. Retrieval grounded in a
   fitted model, not an if/else color-wheel rule.
5. **Quantitative evaluation** — the part that makes this testable
   rather than a nice-looking demo: on a held-out test split of
   paintings, remove one color from each real extracted palette, ask
   the model to recommend companions from the rest, and measure
   whether it recovers the held-out color (top-k hit rate within a
   Delta E threshold; mean Delta E to nearest suggestion). Compared
   against a random-color baseline and a classic color-wheel rule
   baseline, to demonstrate the learned model beats naive color
   theory.

Classical unsupervised ML (K-Means, PMI+SVD, k-NN retrieval) was
chosen deliberately over a neural embedding: same conceptual weight,
far less training-time and infra risk, appropriate for a dataset of a
few thousand paintings and a one-week timebox.

## Dataset

**Source:** [Art Institute of Chicago public API](https://api.artic.edu/docs/) (`api.artic.edu`). Chosen because it requires no API key (just a courtesy `AIC-User-Agent` header identifying the app instead of a hard rate limit), is well documented, and serves images through a IIIF endpoint that lets us request a fixed, modest resolution directly rather than downloading full-resolution masters. Metadata is CC0-licensed; for artworks flagged `is_public_domain: true`, the images themselves are also released under CC0 — free to reproduce for any purpose, including this project.

**What's downloaded:** for each candidate artwork — title, artist, display/start/end date, a stable id, a link back to the object page, and one JPEG image (~600px wide, via `https://www.artic.edu/iiif/2/{image_id}/full/600,/0/default.jpg`). Only artworks with `artwork_type_title == "Painting"`, `is_public_domain == true`, and an available image are kept; everything else is filtered out before download. (Note: AIC's documented structured query syntax for expressing that filter server-side proved unreliable in testing — it returned zero results for filters that worked moments earlier via other params — so filtering happens client-side in `data/sources/artic.py` against the plain full-text search, which was consistently reliable.)

**How palettes are generated:** each downloaded image is run through the existing `color.extraction.extract_palette` (K-Means in CIELAB, see above) — nothing dataset-specific about color extraction, it's the same function used for a single image. Metadata + palette are flattened into one JSON object per line in `data/processed/palettes.jsonl`.

**Reproducing the dataset locally:**

```bash
python scripts/build_dataset.py --limit 100
# or, for a quick smoke test:
python scripts/build_dataset.py --limit 10 -v
```

Re-running with the same `--raw-dir` (default `data/raw/`) reuses already-downloaded images instead of re-fetching them — only new artworks trigger a download. A failure on any single artwork (network error, corrupt/undecodable image) is logged and skipped; it doesn't abort the run, and its partial cache file (if any) is removed so a later re-run retries it.

**Why raw artwork files (and the processed JSONL) aren't committed to git:** both are fully reproducible from `scripts/build_dataset.py` plus the AIC API, so committing them would just bloat the repo with regenerable binary/derived data. `data/raw/` and `data/processed/` are gitignored; anyone cloning the repo regenerates the dataset locally with the command above.

## Repository layout

```
src/paletteml/
  config.py          Paths and constants
  data/               Dataset acquisition:
                        schema.py     source-agnostic ArtworkMetadata/ArtworkRecord
                        sources/      dataset-source-specific code (artic.py = AIC API)
                        ingest.py     generic pipeline: any source -> cached images -> palettes -> JSONL
                        dataset.py    typed read-back access (modeling stage, not yet implemented)
  color/              Dominant-color extraction (extraction.py) and
                      perceptual color-space conversion (space.py)
  modeling/           vocabulary.py (K-Means color bins), co_occurrence.py (PPMI),
                      recommend.py (direct PPMI recommender), embedding.py (SVD
                      embedding), svd_recommend.py (cosine-similarity recommender),
                      baseline.py / random_baseline.py (comparison baselines)
  evaluation/         Train/test split, leave-one-out cases, Hit@K/MRR metrics,
                      significance testing, failure-case analysis
  api/                FastAPI app (main.py), Pydantic schemas (schemas.py) —
                      see "Deploying to Render" below

scripts/              CLI entry points: build_dataset.py, train.py, evaluate.py, recommend.py
frontend/             Not yet built
data/                 raw/ processed/ external/  (gitignored, regenerable via scripts/build_dataset.py)
models/               Trained artifacts: color_vocabulary.json, co_occurrence.json,
                      color_embedding.json — small (tens of KB), committed to git,
                      required at runtime by the API (see "Deploying to Render")
reports/              evaluation_report.md (committed) + figures/metrics (gitignored, regenerable)
notebooks/            Exploratory analysis
tests/                pytest suite
docker/               Dockerfile / docker-compose.yml (stubs — not needed for Render, see below)
render.yaml           Render Blueprint (build/start commands, health check)
```

## Setup

Requires Python 3.11+ (developed against 3.14; see note below on
Windows).

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements-dev.txt   # includes requirements.txt + dev tools
```

Run the (currently trivial) test suite:

```bash
pytest
```

Run the API locally (requires trained artifacts under `models/` —
already committed, or regenerate with `python scripts/train.py`):

```bash
uvicorn paletteml.api.main:app --reload
# GET  http://127.0.0.1:8000/health    -> {"status": "ok", "model_loaded": true, ...}
# docs http://127.0.0.1:8000/docs      -> interactive OpenAPI UI
```

Try it:

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"colors": ["#b23a2f"], "top_n": 5}'

curl -X POST http://127.0.0.1:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"colors": ["#2f6b8e"], "top_n": 5}'
```

**Note on Python version:** this machine only has Python 3.14
installed. If any dependency (notably `scikit-image` /
`scikit-learn`) turns out not to have prebuilt wheels for 3.14 when
you run the install, the fastest fix is installing a 3.12 interpreter
alongside it (`py install 3.12` via the Python launcher) and creating
the venv with `py -3.12 -m venv .venv` instead. Not needed unless
`pip install` actually fails.

## Deploying to Render

The API is a plain FastAPI service — no Docker needed, Render's native
Python runtime builds and runs it directly. `render.yaml` at the repo
root defines this as a Blueprint; the same settings can also be entered
by hand in the Render dashboard when creating a new Web Service.

**Build command:**
```
pip install -r requirements.txt
```

**Start command:**
```
uvicorn paletteml.api.main:app --host 0.0.0.0 --port $PORT
```
Binding `0.0.0.0` (not `127.0.0.1`) and reading `$PORT` (not a
hardcoded port) are both required — Render assigns the port at
runtime and routes external traffic to it; a service bound to
`127.0.0.1` or a fixed port is unreachable from outside its container.

**Environment variables:** none are required at runtime — the API
never calls an external service (the Art Institute of Chicago API is
only used offline by `scripts/build_dataset.py`, not by the deployed
service). `PYTHON_VERSION=3.12.7` is set in `render.yaml` to pin the
build to a known-good interpreter version rather than whatever Render
defaults to; not required for correctness, just for reproducible
builds. No API keys or secrets are needed anywhere in this stack.

**Files that must be committed before deploying:** beyond the normal
source tree, specifically:
- `models/color_vocabulary.json`, `models/co_occurrence.json`,
  `models/color_embedding.json` — the trained artifacts the API loads
  at startup (see `api/main.py`'s `lifespan`). These are small (tens
  of KB total) and were deliberately committed rather than
  regenerated at build time — see the note in `.gitignore` above
  `models/*`. **If these three files aren't in the repo Render builds
  from, the service will crash on startup** (`ColorVocabulary.load()`
  etc. will raise `FileNotFoundError`) rather than serve a broken API.
- `render.yaml` (or the equivalent dashboard configuration).
- `requirements.txt` (already tracked).

Health check path is `/health` — Render polls this to decide whether a
deploy is live; it returns `model_loaded: true` plus the loaded
vocabulary size and SVD dimension, so a passing health check is also
confirmation the real trained model came up correctly, not just that
the process is alive.

## Roadmap (one week)

1. ~~Dataset ingest~~ Done — see "Dataset" above.
2. ~~Co-occurrence (PPMI) recommender + SVD embedding recommender.~~ Done —
   `modeling/recommend.py`, `modeling/svd_recommend.py`.
3. ~~Held-out evaluation vs. baselines, with significance testing.~~ Done —
   `reports/evaluation_report.md`.
4. ~~FastAPI endpoints wired to the fitted model, deployable on Render.~~ Done —
   see "Deploying to Render" above.
5. Static frontend calling the API.
6. Dockerize; deploy to AWS (single small instance or App Runner /
   Elastic Beanstalk — kept simple given the timebox).

## Non-goals

- No OpenAI, Anthropic, or other LLM API calls anywhere in the
  pipeline. Recommendations come from a model fit on painting pixel
  data, not from a language model.
- No frontend framework/build step — the UI is a thin client over
  the API, kept deliberately small so the project stays focused on
  the ML/data-science work.
