"""Monitoring service for the GNN recipe recommender.

Receives telemetry from the serving API (via /track) and user rating
events (via /feedback), exposes aggregate metrics and promotion/rollback
decisions.  Runs as a standalone FastAPI service on port 9090.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from decisions import evaluate
from feedback_store import FeedbackStore
from metrics import RollingMetrics


FEEDBACK_LOG_PATH = os.environ.get("FEEDBACK_LOG_PATH", "/logs/feedback.jsonl")
SERVING_LOG_PATH = os.environ.get("SERVING_LOG_PATH", "/logs/serving.log")
DECISIONS_LOG_PATH = os.environ.get("DECISIONS_LOG_PATH", "/logs/decisions.jsonl")
MODEL_VERSION_PATH = os.environ.get("MODEL_VERSION_PATH", "/models/current_version.txt")

REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))

# 0 means "we don't know the denominator", so coverage is reported as null
# rather than an inflated or deflated number.
EXPECTED_USER_COUNT = int(os.environ.get("EXPECTED_USER_COUNT", "0"))

FEEDBACK_WINDOW_DAYS = 7


app = FastAPI(title="GNN Recipe Recommender — Monitoring", version="1.0.0")

# Module-level singletons so the deques persist across requests; FastAPI
# instantiates the app once per worker.
rolling = RollingMetrics()
feedback_store = FeedbackStore(
    feedback_path=FEEDBACK_LOG_PATH,
    serving_path=SERVING_LOG_PATH,
)


class TrackEvent(BaseModel):
    user_id: str
    recipe_ids: list[str] = Field(default_factory=list)
    predicted_scores: list[float] = Field(default_factory=list)
    latency_ms: float
    status_code: int
    timestamp: str


class FeedbackEvent(BaseModel):
    user_id: str
    recipe_id: str
    rating: float
    timestamp: str | None = None


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _compose_metrics() -> dict[str, Any]:
    snap = rolling.snapshot()
    since = _now_utc() - timedelta(days=FEEDBACK_WINDOW_DAYS)
    snap["feedback_rate"] = feedback_store.feedback_rate(since)
    snap["avg_rating_on_recommended"] = feedback_store.avg_rating(since)
    return snap


def _read_model_version() -> str | None:
    # First-line-only read so an accidental multi-line file doesn't leak
    # unrelated content into the response.
    try:
        with open(MODEL_VERSION_PATH, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            return line or None
    except FileNotFoundError:
        return None
    except OSError:
        return None


def _recommendation_coverage() -> float | None:
    # EXPECTED_USER_COUNT=0 sentinel: operators haven't declared the
    # universe size yet, so coverage isn't meaningful.
    if EXPECTED_USER_COUNT <= 0:
        return None
    try:
        import redis  # Imported lazily so a Redis-less dev run still starts.
    except ImportError:
        return None
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
        # SCAN rather than KEYS — KEYS blocks the server on large keyspaces
        # and the monitoring endpoint should never be able to DOS Redis.
        count = 0
        for _ in client.scan_iter(match="user:*", count=500):
            count += 1
        return count / EXPECTED_USER_COUNT
    except Exception:
        # Redis unreachable is a soft failure for monitoring — the service
        # itself must stay green even if a dependency is flapping.
        return None


def _score_distribution() -> dict[str, float] | None:
    scores = list(rolling.predicted_scores)
    if not scores:
        return None
    return {
        "min": float(min(scores)),
        "max": float(max(scores)),
        "mean": float(sum(scores) / len(scores)),
    }


@app.post("/track")
def track(event: TrackEvent) -> dict[str, Any]:
    rolling.record_track(event.model_dump())
    return {"ok": True}


@app.post("/feedback")
def feedback(event: FeedbackEvent) -> dict[str, Any]:
    feedback_store.append(
        user_id=event.user_id,
        recipe_id=event.recipe_id,
        rating=event.rating,
        timestamp=event.timestamp,
    )
    return {"ok": True}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return _compose_metrics()


@app.get("/model-health")
def model_health() -> dict[str, Any]:
    return {
        "current_model_version": _read_model_version(),
        "recommendation_coverage": _recommendation_coverage(),
        "score_distribution": _score_distribution(),
    }


@app.get("/promote-decision")
def promote_decision() -> dict[str, Any]:
    snap = _compose_metrics()
    result = evaluate(snap, feedback_store, _now_utc())

    # Persist every decision for audit — even holds — so the SRE review
    # of "why didn't we promote" has a ground-truth trail.
    try:
        Path(DECISIONS_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(DECISIONS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, sort_keys=True) + "\n")
    except OSError:
        # Failing to log a decision must not fail the endpoint itself;
        # the decision is still returned to the caller.
        pass

    return result
