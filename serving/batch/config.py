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
