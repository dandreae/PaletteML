"""FastAPI application exposing the trained recommenders, plus the
static frontend (see the bottom of this file).

Model artifacts are loaded once at startup (see `lifespan` below) and
reused for every request — nothing is retrained or refit per-request.
If artifacts are missing or corrupt, startup fails loudly (the process
won't come up) rather than serving a broken API — the correct failure
mode for a deploy, surfaced immediately instead of as scattered 500s.

Run locally:  uvicorn paletteml.api.main:app --reload
Production:   see README's "Deploying to Render" section.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from paletteml.api.schemas import (
    CompareResponse,
    HealthResponse,
    RecommendationItem,
    RecommendRequest,
    RecommendResponse,
)
from paletteml.config import FRONTEND_DIR, MODELS_DIR
from paletteml.modeling.baseline import PopularityBaseline
from paletteml.modeling.co_occurrence import CoOccurrenceModel
from paletteml.modeling.embedding import ColorEmbedding
from paletteml.modeling.recommend import CoOccurrenceRecommender
from paletteml.modeling.svd_recommend import SvdEmbeddingRecommender
from paletteml.modeling.vocabulary import ColorVocabulary


@dataclass
class ModelState:
    """Every trained artifact + the recommenders built on top of them, held once."""

    vocabulary: ColorVocabulary
    co_occurrence: CoOccurrenceModel
    embedding: ColorEmbedding
    co_recommender: CoOccurrenceRecommender
    svd_recommender: SvdEmbeddingRecommender
    popularity: PopularityBaseline


def load_model_state(models_dir=MODELS_DIR) -> ModelState:
    vocabulary = ColorVocabulary.load(models_dir / "color_vocabulary.json")
    co_occurrence = CoOccurrenceModel.load(models_dir / "co_occurrence.json")
    embedding = ColorEmbedding.load(models_dir / "color_embedding.json")
    return ModelState(
        vocabulary=vocabulary,
        co_occurrence=co_occurrence,
        embedding=embedding,
        co_recommender=CoOccurrenceRecommender(vocabulary, co_occurrence),
        svd_recommender=SvdEmbeddingRecommender(vocabulary, embedding),
        popularity=PopularityBaseline(vocabulary, co_occurrence),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model_state()
    yield
    # nothing to release — model state is plain in-memory numpy/dataclasses


app = FastAPI(
    title="PaletteML API",
    description=(
        "Recommends colors that work well with a seed color, learned from "
        "real paintings — not an LLM wrapper. See /compare to see the "
        "learned SVD/co-occurrence recommenders against a popularity baseline."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# --- CORS ---
#
# The frontend is served by this same app (mounted at "/" below), so in
# production browser requests are same-origin and CORS is never
# exercised at all. This exists for two other real cases: (1) local
# development, running the frontend from a different port/tool while
# pointed at this API, and (2) the documented fallback of deploying the
# frontend separately (e.g. a Render Static Site) — see README's
# "Deploying to Render" section. No wildcard "*" origin is used, and no
# origin is trusted by default beyond localhost dev; a real deployed
# frontend origin must be added explicitly via ALLOWED_ORIGINS.
#
# ALLOWED_ORIGINS: comma-separated list of additional exact origins to
# allow, e.g. "https://paletteml.onrender.com,https://example.com".
# Unset/empty is fine — same-origin deployment needs nothing here.
LOCAL_DEV_ORIGIN_REGEX = r"http://(localhost|127\.0\.0\.1)(:\d+)?"  # any local dev port


def parse_allowed_origins(raw: str) -> list[str]:
    """Parse the ALLOWED_ORIGINS env var into a list of exact origins.

    Pulled out as its own function (rather than inlined where it's
    used) so it's directly unit-testable without needing to reimport
    this module with a different environment each time.
    """
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_allowed_origins(os.environ.get("ALLOWED_ORIGINS", "")),
    allow_origin_regex=LOCAL_DEV_ORIGIN_REGEX,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=False,  # no cookies/auth in use — keep this off
)


def get_model_state(request: Request) -> ModelState:
    return request.app.state.model


def _to_items(recs, top_n: int) -> list[RecommendationItem]:
    return [RecommendationItem(hex=r.hex, score=r.score) for r in recs[:top_n]]


@app.get("/health", response_model=HealthResponse)
def health(model: ModelState = Depends(get_model_state)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=True,
        vocab_size=model.vocabulary.size,
        n_training_artworks=model.co_occurrence.n_artworks,
        svd_dimension=model.embedding.n_components,
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest, model: ModelState = Depends(get_model_state)) -> RecommendResponse:
    """Top SVD-embedding-based recommendations for one or more seed colors."""
    recs = model.svd_recommender.recommend(payload.colors, top_n=payload.top_n)
    return RecommendResponse(seed_colors=payload.colors, recommendations=_to_items(recs, payload.top_n))


@app.post("/compare", response_model=CompareResponse)
def compare(payload: RecommendRequest, model: ModelState = Depends(get_model_state)) -> CompareResponse:
    """The same seed colors, ranked independently by all three recommenders.

    Popularity ignores seed colors by design (see modeling/baseline.py)
    — its ranking is identical regardless of `payload.colors`, included
    here purely as a comparison baseline, same as in the evaluation
    reports under reports/.
    """
    svd_recs = model.svd_recommender.recommend(payload.colors, top_n=payload.top_n)
    co_recs = model.co_recommender.recommend(payload.colors, top_n=payload.top_n)
    pop_recs = model.popularity.recommend(top_n=payload.top_n)
    return CompareResponse(
        seed_colors=payload.colors,
        results={
            "svd": _to_items(svd_recs, payload.top_n),
            "co_occurrence": _to_items(co_recs, payload.top_n),
            "popularity": _to_items(pop_recs, payload.top_n),
        },
    )


# --- static frontend ---
#
# Mounted LAST and at "/" deliberately: Starlette matches routes in
# registration order, so /health, /recommend, /compare, and FastAPI's
# own /docs, /redoc, /openapi.json (registered when FastAPI(...) was
# constructed above) are all matched first. This mount only ever
# handles requests that don't match one of those — it can't shadow them
# regardless of what paths exist under frontend/. html=True makes
# StaticFiles serve frontend/index.html for "/" and for any other
# unmatched path fall through to a normal 404, rather than needing a
# hand-written homepage route.
#
# Guarded by existence so a checkout without frontend/ (unlikely, since
# it's committed, but possible in a stripped-down environment) still
# serves the API correctly — just without a homepage.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
