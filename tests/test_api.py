"""Tests for the FastAPI app (api/main.py, api/schemas.py).

Two flavors, deliberately kept separate:
  - "real artifact" tests exercise the actual lifespan against the
    real, committed models/*.json — proving the whole deployed stack
    actually works end-to-end, which is exactly what needs proving
    before a Render deploy.
  - "fake state" tests use FastAPI's dependency_overrides to inject a
    small synthetic ModelState, for precise/fast assertions that don't
    depend on real trained values (and don't trigger the real
    lifespan/disk load at all — see _fake_client below).
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from paletteml.api.main import ModelState, app, get_model_state, parse_allowed_origins
from paletteml.modeling.baseline import PopularityBaseline
from paletteml.modeling.co_occurrence import CoOccurrenceModel
from paletteml.modeling.embedding import ColorEmbedding
from paletteml.modeling.recommend import CoOccurrenceRecommender
from paletteml.modeling.svd_recommend import SvdEmbeddingRecommender
from paletteml.modeling.vocabulary import ColorVocabulary

RANDOM_STATE = 42


def _fake_model_state() -> ModelState:
    """A tiny, fully synthetic ModelState — independent of real trained artifacts."""
    points = np.array(
        [[40.0, 55.0, 35.0], [55.0, -40.0, 30.0], [30.0, 10.0, -55.0], [60.0, 20.0, 45.0]] * 3,
        dtype=np.float64,
    )
    vocabulary = ColorVocabulary.fit(points, vocab_size=4, random_state=RANDOM_STATE)

    color_counts = np.array([10, 8, 6, 4], dtype=np.int64)
    pair_counts = np.zeros((4, 4), dtype=np.int64)
    pair_counts[0, 1] = pair_counts[1, 0] = 6
    pair_counts[0, 2] = pair_counts[2, 0] = 3
    co_occurrence = CoOccurrenceModel(
        vocab_size=4, n_artworks=20, color_counts=color_counts, pair_counts=pair_counts
    )
    embedding = ColorEmbedding.fit(co_occurrence, n_components=3)

    return ModelState(
        vocabulary=vocabulary,
        co_occurrence=co_occurrence,
        embedding=embedding,
        co_recommender=CoOccurrenceRecommender(vocabulary, co_occurrence),
        svd_recommender=SvdEmbeddingRecommender(vocabulary, embedding),
        popularity=PopularityBaseline(vocabulary, co_occurrence),
    )


@pytest.fixture
def fake_client():
    """Plain (non-context-manager) TestClient: lifespan never runs, so
    this never touches models/ on disk — fully isolated from real
    trained artifacts. Only the dependency-overridden fake state is used.
    """
    state = _fake_model_state()
    app.dependency_overrides[get_model_state] = lambda: state
    client = TestClient(app)
    try:
        yield client, state
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def real_client():
    """Context-manager TestClient: triggers the real lifespan, loading
    the real, committed models/*.json — an end-to-end smoke test of
    exactly what Render will run.
    """
    with TestClient(app) as client:
        yield client


class TestHealthRealArtifacts:
    def test_health_ok_with_real_trained_artifacts(self, real_client):
        response = real_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["vocab_size"] == 64
        assert body["svd_dimension"] == 16
        assert body["n_training_artworks"] > 0


class TestRecommendRealArtifacts:
    def test_recommend_returns_well_formed_response(self, real_client):
        response = real_client.post("/recommend", json={"colors": ["#b23a2f"]})
        assert response.status_code == 200
        body = response.json()
        assert body["seed_colors"] == ["#b23a2f"]
        recs = body["recommendations"]
        assert 1 <= len(recs) <= 5
        for rec in recs:
            assert rec["hex"].startswith("#")
            assert len(rec["hex"]) == 7
            assert isinstance(rec["score"], float)
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_respects_top_n(self, real_client):
        response = real_client.post("/recommend", json={"colors": ["#2f6b8e"], "top_n": 3})
        assert response.status_code == 200
        assert len(response.json()["recommendations"]) <= 3

    def test_recommend_accepts_multiple_seeds(self, real_client):
        response = real_client.post("/recommend", json={"colors": ["#b23a2f", "#2f6b8e"]})
        assert response.status_code == 200
        assert response.json()["seed_colors"] == ["#b23a2f", "#2f6b8e"]


class TestCompareRealArtifacts:
    def test_compare_returns_all_three_methods(self, real_client):
        response = real_client.post("/compare", json={"colors": ["#3a7d44"]})
        assert response.status_code == 200
        results = response.json()["results"]
        assert set(results.keys()) == {"svd", "co_occurrence", "popularity"}
        for method, recs in results.items():
            assert isinstance(recs, list)
            for rec in recs:
                assert "hex" in rec and "score" in rec


class TestValidation:
    def test_invalid_hex_returns_422(self, fake_client):
        client, _state = fake_client
        response = client.post("/recommend", json={"colors": ["not-a-color"]})
        assert response.status_code == 422
        assert "invalid hex" in response.text.lower()

    def test_empty_colors_list_returns_422(self, fake_client):
        client, _state = fake_client
        response = client.post("/recommend", json={"colors": []})
        assert response.status_code == 422

    def test_too_many_colors_returns_422(self, fake_client):
        client, _state = fake_client
        response = client.post("/recommend", json={"colors": ["#ffffff"] * 11})
        assert response.status_code == 422

    def test_top_n_zero_returns_422(self, fake_client):
        client, _state = fake_client
        response = client.post("/recommend", json={"colors": ["#ffffff"], "top_n": 0})
        assert response.status_code == 422

    def test_top_n_too_large_returns_422(self, fake_client):
        client, _state = fake_client
        response = client.post("/recommend", json={"colors": ["#ffffff"], "top_n": 100})
        assert response.status_code == 422

    def test_hex_without_hash_prefix_is_normalized(self, fake_client):
        client, _state = fake_client
        response = client.post("/recommend", json={"colors": ["b23a2f"]})
        assert response.status_code == 200
        assert response.json()["seed_colors"] == ["#b23a2f"]

    def test_missing_colors_field_returns_422(self, fake_client):
        client, _state = fake_client
        response = client.post("/recommend", json={})
        assert response.status_code == 422


class TestFakeStateBehavior:
    def test_health_reflects_injected_state(self, fake_client):
        client, state = fake_client
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["vocab_size"] == state.vocabulary.size == 4
        assert body["svd_dimension"] == state.embedding.n_components == 3

    def test_recommend_never_includes_the_seed_color_itself(self, fake_client):
        client, state = fake_client
        seed_hex = state.vocabulary.entries[0].hex
        response = client.post("/recommend", json={"colors": [seed_hex], "top_n": 3})
        assert response.status_code == 200
        result_hexes = {r["hex"] for r in response.json()["recommendations"]}
        assert seed_hex not in result_hexes

    def test_compare_popularity_ignores_seed_colors(self, fake_client):
        client, state = fake_client
        seed_a = state.vocabulary.entries[0].hex
        seed_b = state.vocabulary.entries[3].hex

        response_a = client.post("/compare", json={"colors": [seed_a]})
        response_b = client.post("/compare", json={"colors": [seed_b]})

        pop_a = response_a.json()["results"]["popularity"]
        pop_b = response_b.json()["results"]["popularity"]
        assert pop_a == pop_b  # identical regardless of seed, by design

    def test_recommend_caps_gracefully_when_top_n_exceeds_available_candidates(self, fake_client):
        client, state = fake_client
        seed_hex = state.vocabulary.entries[0].hex
        response = client.post("/recommend", json={"colors": [seed_hex], "top_n": 20})
        assert response.status_code == 200
        # only vocab_size - 1 candidates can ever exist for a single seed
        assert len(response.json()["recommendations"]) <= state.vocabulary.size - 1


class TestParseAllowedOrigins:
    """Unit tests for the ALLOWED_ORIGINS env var parser.

    Tested as a pure function rather than by reimporting api.main with
    different environments — see parse_allowed_origins()'s docstring.
    """

    def test_empty_string_gives_no_extra_origins(self):
        assert parse_allowed_origins("") == []

    def test_single_origin(self):
        assert parse_allowed_origins("https://example.com") == ["https://example.com"]

    def test_multiple_origins_with_whitespace(self):
        assert parse_allowed_origins("https://a.com, https://b.com") == [
            "https://a.com",
            "https://b.com",
        ]

    def test_skips_empty_entries_from_stray_commas(self):
        assert parse_allowed_origins("https://a.com,,https://b.com,") == [
            "https://a.com",
            "https://b.com",
        ]


class TestCorsBehavior:
    """Exercises the live CORSMiddleware on the real (env-default) app."""

    def test_allows_localhost_dev_origin(self, fake_client):
        client, _state = fake_client
        response = client.get("/health", headers={"Origin": "http://localhost:5500"})
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5500"

    def test_allows_127_0_0_1_dev_origin_with_port(self, fake_client):
        client, _state = fake_client
        response = client.get("/health", headers={"Origin": "http://127.0.0.1:3000"})
        assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:3000"

    def test_does_not_allow_arbitrary_origin_by_default(self, fake_client):
        client, _state = fake_client
        response = client.get("/health", headers={"Origin": "https://evil.example.com"})
        assert response.headers.get("access-control-allow-origin") != "https://evil.example.com"

    def test_preflight_for_recommend_succeeds_for_allowed_origin(self, fake_client):
        client, _state = fake_client
        response = client.options(
            "/recommend",
            headers={
                "Origin": "http://localhost:5500",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5500"

    def test_no_wildcard_origin_is_ever_returned(self, fake_client):
        client, _state = fake_client
        response = client.get("/health", headers={"Origin": "http://localhost:5500"})
        assert response.headers.get("access-control-allow-origin") != "*"


class TestStaticFrontend:
    """The homepage and static assets are served by the same app as the API."""

    def test_homepage_serves_index_html(self, real_client):
        response = real_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "PaletteML" in response.text

    def test_static_assets_are_served(self, real_client):
        for path in ("/style.css", "/app.js", "/config.js"):
            response = real_client.get(path)
            assert response.status_code == 200, path

    def test_static_mount_does_not_shadow_api_routes(self, real_client):
        # regression check: mounting StaticFiles at "/" must never
        # intercept these — see the ordering comment in api/main.py
        assert real_client.get("/health").status_code == 200
        assert real_client.post("/recommend", json={"colors": ["#b23a2f"]}).status_code == 200
        assert real_client.post("/compare", json={"colors": ["#b23a2f"]}).status_code == 200
        assert real_client.get("/docs").status_code == 200
        assert real_client.get("/openapi.json").status_code == 200

    def test_unknown_path_returns_404_not_the_homepage(self, real_client):
        response = real_client.get("/this-path-does-not-exist")
        assert response.status_code == 404
