"""FastAPI application exposing the trained recommenders.

Model artifacts are loaded once at startup (see `lifespan` below) and
reused for every request — nothing is retrained or refit per-request.
If artifacts are missing or corrupt, startup fails loudly (the process
won't come up) rather than serving a broken API — the correct failure
mode for a deploy, surfaced immediately instead of as scattered 500s.

Run locally:  uvicorn paletteml.api.main:app --reload
Production:   see README's "Deploying to Render" section.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import Depends, FastAPI, Request

from paletteml.api.schemas import (
    CompareResponse,
    HealthResponse,
    RecommendationItem,
    RecommendRequest,
    RecommendResponse,
)
from paletteml.config import MODELS_DIR
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
