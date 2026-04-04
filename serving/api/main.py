"""FastAPI serving layer for GNN-based recipe recommendations.

Pre-computed recommendations are looked up from a swappable cache (Redis or
in-memory dict).  Recipe metadata (name, tags) is resolved from a JSON file
produced by the batch pipeline.  No live model inference happens at request
time, keeping p99 latency well under 50 ms.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cache import CacheBackend, MemoryCache, RedisCache, create_cache
from config import settings

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_DIR = Path(settings.LOG_DIR)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Structured JSON logger for the serving audit trail
from pythonjsonlogger import json as jsonlog  # python-json-logger >= 3.x compat

_serving_handler = logging.FileHandler(LOG_DIR / "serving.log")
_serving_handler.setFormatter(
    jsonlog.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )
)

serving_logger = logging.getLogger("serving")
serving_logger.setLevel(logging.INFO)
serving_logger.addHandler(_serving_handler)

# General app logger (stdout)
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_console)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    user_id: str


class RecipeRecommendation(BaseModel):
    recipe_id: str
    predicted_score: float
    name: str
    tags: Optional[List[str]] = None


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[RecipeRecommendation]
    served_at: str


# ---------------------------------------------------------------------------
# Application state loaded at startup
# ---------------------------------------------------------------------------

# Populated during the lifespan event
cache: CacheBackend
recipe_metadata: Dict[str, Any] = {}


def _load_recipe_metadata() -> Dict[str, Any]:
    """Load recipe metadata from the JSON file specified in config.

    Expected format:
        { "<recipe_id>": { "name": "...", "tags": ["tag1", ...] }, ... }

    If the file does not exist yet (batch job hasn't run), return an empty
    dict and log a warning so the service can still start.
    """
    path = Path(settings.RECIPE_METADATA_PATH)
    if not path.exists():
        logger.warning(
            "Recipe metadata file not found at %s — "
            "recommendations will have 'Unknown' names until the batch job runs.",
            path,
        )
        return {}
    with open(path, "r") as f:
        data: Dict[str, Any] = json.load(f)
    logger.info("Loaded metadata for %d recipes from %s", len(data), path)
    return data


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cache, recipe_metadata

    # 1. Recipe metadata
    recipe_metadata = _load_recipe_metadata()

    # 2. Cache backend
    cache = create_cache(
        settings.CACHE_BACKEND,
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
    )
    logger.info("Cache backend: %s", settings.CACHE_BACKEND)

    # 3. If using in-memory cache, seed from the recommendations JSON produced
    #    by the batch pipeline (if the file exists).
    if isinstance(cache, MemoryCache):
        recs_path = Path(settings.RECOMMENDATIONS_PATH)
        if recs_path.exists():
            with open(recs_path) as f:
                recs_data = json.load(f)
            cache.load_bulk(recs_data)
            logger.info(
                "Loaded %d user recommendations from %s into memory cache",
                len(recs_data),
                recs_path,
            )
        else:
            logger.warning(
                "Recommendations file not found at %s — "
                "memory cache will be empty until batch job runs.",
                recs_path,
            )

    yield  # ---- application runs ----

    # Teardown
    if isinstance(cache, RedisCache):
        await cache.close()
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GNN Recipe Recommendation API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Liveness / readiness probe."""
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """Return pre-computed recipe recommendations for *user_id*.

    Looks up a JSON blob from the cache keyed by user_id, resolves recipe
    metadata, and returns the enriched list.
    """
    user_id = request.user_id

    raw: Optional[str] = await cache.get(user_id)
    if raw is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No pre-computed recommendations found for user '{user_id}'. "
                "The user may not exist or the batch pipeline has not processed "
                "this user yet."
            ),
        )

    # Expected cache value: list of {"recipe_id": "...", "predicted_score": ...}
    try:
        cached_recs: List[Dict[str, Any]] = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error("Corrupt cache entry for user %s: %s", user_id, exc)
        raise HTTPException(
            status_code=500,
            detail="Corrupted recommendation data. Please retry later.",
        )

    # Enrich each recommendation with metadata
    enriched: List[RecipeRecommendation] = []
    served_recipe_ids: List[str] = []
    for rec in cached_recs:
        rid = str(rec["recipe_id"])
        meta = recipe_metadata.get(rid, {})
        enriched.append(
            RecipeRecommendation(
                recipe_id=rid,
                predicted_score=round(float(rec["predicted_score"]), 4),
                name=meta.get("name", rec.get("name", "Unknown")),
                tags=meta.get("tags"),
            )
        )
        served_recipe_ids.append(rid)

    served_at = datetime.now(timezone.utc).isoformat()

    # Audit log (for the feedback loop)
    serving_logger.info(
        "recommendations_served",
        extra={
            "user_id": user_id,
            "recipe_ids": served_recipe_ids,
            "count": len(served_recipe_ids),
            "served_at": served_at,
        },
    )

    return RecommendResponse(
        user_id=user_id,
        recommendations=enriched,
        served_at=served_at,
    )


# ---------------------------------------------------------------------------
# Entrypoint (for running directly with `python main.py`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        log_level="info",
    )
