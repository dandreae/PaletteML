"""Pydantic request/response models for the API.

Hex validation reuses color/space.py's own parser (hex_to_rgb) rather
than re-implementing hex-format rules — the same principle followed
throughout this project (one source of truth per conversion).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from paletteml.color.space import hex_to_rgb


def _validate_hex_color(value: str) -> str:
    try:
        hex_to_rgb(value)
    except ValueError as exc:
        raise ValueError(f"invalid hex color {value!r}: {exc}") from exc
    return value if value.startswith("#") else f"#{value}"


class RecommendRequest(BaseModel):
    """One or more seed colors to recommend companions for."""

    colors: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description='One or more seed hex colors, e.g. ["#b23a2f"]',
        examples=[["#b23a2f"]],
    )
    top_n: int = Field(5, ge=1, le=20, description="Number of recommendations to return")

    @field_validator("colors")
    @classmethod
    def validate_colors(cls, value: list[str]) -> list[str]:
        return [_validate_hex_color(c) for c in value]


class RecommendationItem(BaseModel):
    """One recommended color."""

    hex: str = Field(..., description='Recommended color, e.g. "#412215"')
    score: float = Field(..., description="Recommender's ranking score (method-specific — see /compare)")


class RecommendResponse(BaseModel):
    seed_colors: list[str]
    recommendations: list[RecommendationItem]


class CompareResponse(BaseModel):
    """Same seed colors, ranked independently by each of the three recommenders."""

    seed_colors: list[str]
    results: dict[str, list[RecommendationItem]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vocab_size: int | None = None
    n_training_artworks: int | None = None
    svd_dimension: int | None = None
