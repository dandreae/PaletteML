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

# --- color vocabulary / co-occurrence modeling ---
# Number of shared vocabulary colors (K-Means clusters in Lab) that
# all extracted painting palettes get mapped onto. Sized for a
# ~300-artwork / ~1500-dominant-color training set: too few bins
# (e.g. 16) collapses perceptually distinct colors together; too many
# (e.g. 256) leaves most K-Means clusters with only a handful of
# points, making pairwise co-occurrence counts too sparse/noisy to
# trust. 64 gives ~20-25 samples/cluster at this dataset size — a
# starting point, not yet tuned against an evaluation metric.
DEFAULT_VOCAB_SIZE = 64

# --- reproducibility ---
RANDOM_SEED = 42
