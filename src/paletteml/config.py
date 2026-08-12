"""Central project configuration (paths, constants).

Kept as plain module-level constants for now. If this grows, switch
to pydantic-settings so values can be overridden via environment
variables / .env for Docker and AWS deployment.
"""

from pathlib import Path

# --- filesystem layout ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# --- color extraction ---
# Number of dominant colors (K-Means clusters) to extract per painting.
N_DOMINANT_COLORS = 5

# Longest side (px) images are downscaled to before clustering pixels.
# Clustering doesn't need full resolution; this keeps extraction fast.
EXTRACTION_MAX_DIMENSION = 256

# --- reproducibility ---
RANDOM_SEED = 42
