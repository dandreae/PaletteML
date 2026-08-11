"""FastAPI entry point.

TODO: implement once the model can be loaded from `models/`:
  - GET  /health
  - POST /recommend        (seed colors -> suggested companion colors)
  - GET  /palettes/sample  (a few example painting palettes, for the UI)
  - mount `frontend/` as static files (or serve it separately)

Run with: uvicorn paletteml.api.main:app --reload
"""

from fastapi import FastAPI

app = FastAPI(title="PaletteML API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}
