"""Smoke test that the API app imports and boots.

Once real logic exists, add tests per module (color space round-trip,
palette extraction shape, embedding fit, evaluation metrics on a
tiny synthetic dataset).
"""

from fastapi.testclient import TestClient

from paletteml.api.main import app


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
