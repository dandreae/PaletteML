"""Pydantic request/response models for the API.

TODO: define once the model interface is settled, e.g.

    class RecommendRequest(BaseModel):
        colors: list[str]          # hex colors, 1-5 provided by the user
        n_suggestions: int = 4

    class RecommendResponse(BaseModel):
        suggestions: list[str]     # recommended hex colors
        source_paintings: list[str] | None = None  # for interpretability
"""
