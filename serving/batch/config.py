"""
Configuration for the batch pre-computation pipeline.

All settings are read from environment variables with sensible defaults.
"""

import os


# --- Model and data paths ---
MODEL_PATH: str = os.environ.get("MODEL_PATH", "/models/recipe_gnn_model.pt")
DATA_DIR: str = os.environ.get("DATA_DIR", "/data")

# --- Redis connection ---
REDIS_HOST: str = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT: int = int(os.environ.get("REDIS_PORT", "6379"))

# --- Cache backend: "redis" or "memory" ---
# When set to "memory", results are written to a local JSON file instead of Redis.
CACHE_BACKEND: str = os.environ.get("CACHE_BACKEND", "redis")

# --- Recommendation parameters ---
TOP_N: int = int(os.environ.get("TOP_N", "10"))
BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "128"))

# --- Output paths ---
RECIPE_METADATA_OUTPUT: str = os.environ.get(
    "RECIPE_METADATA_OUTPUT", "/data/recipe_metadata.json"
)
RECOMMENDATIONS_OUTPUT: str = os.environ.get(
    "RECOMMENDATIONS_OUTPUT", "/data/recommendations.json"
)

# --- Training hyperparameters (used only when model must be trained) ---
HIDDEN_CHANNELS: int = int(os.environ.get("HIDDEN_CHANNELS", "64"))
TRAIN_EPOCHS: int = int(os.environ.get("TRAIN_EPOCHS", "5"))
TRAIN_LR: float = float(os.environ.get("TRAIN_LR", "0.001"))
TRAIN_BATCH_SIZE: int = int(os.environ.get("TRAIN_BATCH_SIZE", "128"))
NUM_NEIGHBORS: list = [20, 10]
NEG_SAMPLING_RATIO: float = 2.0
TOP_K_INGREDIENTS: int = 500

# Append-only audit of every batch invocation. Monitoring and canary tooling
# tail this file to know which version is currently serving, so it must be
# configurable per deployment (docker volume, host path, etc.).
BATCH_RUN_LOG: str = os.environ.get("BATCH_RUN_LOG", "/logs/batch_run.jsonl")

# MODEL_DIR lives next to the checkpoint; current_version.txt is written here
# so the serving side can resolve which artifact set it is loading.
MODEL_DIR: str = os.path.dirname(MODEL_PATH) or "/models"

# Kaggle credentials for auto-downloading the Food.com dataset on first run.
# Empty defaults let the container start; the download step errors clearly
# if these are blank when a download is actually needed.
KAGGLE_USERNAME: str = os.environ.get("KAGGLE_USERNAME", "")
KAGGLE_KEY: str = os.environ.get("KAGGLE_KEY", "")

S3_ENDPOINT_URL: str = os.environ.get("S3_ENDPOINT_URL", "")
S3_ACCESS_KEY: str = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY: str = os.environ.get("S3_SECRET_KEY", "")
S3_BUCKET: str = os.environ.get("S3_BUCKET", "")
