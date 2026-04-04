"""Configuration for the recipe recommendation serving API.

All settings are loaded from environment variables with sensible defaults
so the service runs out-of-the-box locally and in Docker on Chameleon cloud.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings populated from environment variables."""

    # Cache backend: "redis" or "memory"
    CACHE_BACKEND: str = "memory"

    # Redis connection details (only used when CACHE_BACKEND == "redis")
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # Path to the JSON file produced by the batch job containing recipe metadata
    # Expected format: { "<recipe_id>": { "name": "...", "tags": [...] }, ... }
    RECIPE_METADATA_PATH: str = "/data/recipe_metadata.json"

    # Uvicorn listen port
    PORT: int = 8000

    # Path to pre-computed recommendations JSON (used when CACHE_BACKEND == "memory")
    RECOMMENDATIONS_PATH: str = "/data/recommendations.json"

    # Directory where per-request serving logs are written
    LOG_DIR: str = "/logs"

    model_config = {"env_prefix": "", "case_sensitive": True}


# Singleton used throughout the application
settings = Settings()
