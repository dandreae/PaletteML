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

## Repository layout

```
src/paletteml/
  config.py          Paths and constants
  data/               Dataset acquisition (ingest.py) and access (dataset.py)
  color/              Dominant-color extraction (extraction.py) and
                      perceptual color-space conversion (space.py)
  modeling/           Co-occurrence embedding (embedding.py) and
                      recommendation (recommend.py)
  evaluation/         Held-out color-prediction metrics + baselines
  api/                FastAPI app (main.py) and request/response schemas

scripts/              CLI entry points: build_dataset.py, train.py, evaluate.py
frontend/             Minimal static HTML/CSS/JS UI (no build step, no framework)
data/                 raw/ processed/ external/  (gitignored, regenerable)
models/               Trained model artifacts (gitignored)
reports/              Evaluation metrics tables and figures (gitignored)
notebooks/            Exploratory analysis
tests/                pytest suite
docker/               Dockerfile / docker-compose.yml (stubs until API works end-to-end)
```

All Python modules currently contain module docstrings and `TODO` /
`NotImplementedError` stubs — the shape of the pipeline exists, none
of the ML logic is implemented yet.

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

Run the API skeleton:

```bash
uvicorn paletteml.api.main:app --reload
# GET http://127.0.0.1:8000/health -> {"status": "ok"}
```

**Note on Python version:** this machine only has Python 3.14
installed. If any dependency (notably `scikit-image` /
`scikit-learn`) turns out not to have prebuilt wheels for 3.14 when
you run the install, the fastest fix is installing a 3.12 interpreter
alongside it (`py install 3.12` via the Python launcher) and creating
the venv with `py -3.12 -m venv .venv` instead. Not needed unless
`pip install` actually fails.

## Roadmap (one week)

1. Dataset ingest: pull a few thousand painting images + metadata
   (WikiArt subset), write manifest.
2. Dominant-color extraction pipeline over the full dataset ->
   `data/processed/palettes.parquet`.
3. Co-occurrence embedding + clustering, fit on a train split.
4. Recommendation function + held-out evaluation vs. baselines,
   written up in `reports/`.
5. FastAPI endpoints wired to the fitted model.
6. Static frontend calling the API.
7. Dockerize; deploy to AWS (single small instance or App Runner /
   Elastic Beanstalk — kept simple given the timebox).

## Non-goals

- No OpenAI, Anthropic, or other LLM API calls anywhere in the
  pipeline. Recommendations come from a model fit on painting pixel
  data, not from a language model.
- No frontend framework/build step — the UI is a thin client over
  the API, kept deliberately small so the project stays focused on
  the ML/data-science work.
