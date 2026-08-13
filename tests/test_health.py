"""Smoke test that the API app imports and boots.

See tests/test_api.py for the full API test suite. This file stays as
a minimal, self-contained "does the app even come up" check.

Uses `with TestClient(app) as client:` deliberately — without it,
FastAPI's lifespan (which loads model artifacts into app.state) never
runs, and the request would raise instead of returning a real response.
Since `app` is a shared module-level object, a plain TestClient(app)
here could appear to work by accident if another test's real_client
fixture happened to run first in the same session and left app.state
populated — order-dependent and fragile. Always use the context
manager form for any test that needs the real app.state.
"""

from fastapi.testclient import TestClient

from paletteml.api.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
