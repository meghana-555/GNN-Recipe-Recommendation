"""File-backed feedback and served-audit store.

JSONL is chosen over a database because the monitoring service must stay
operational even if the primary datastore is down, and the volumes are
small enough (feedback events + served events) that linear scans over a
7-30 day window are acceptable and trivially re-computable.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _parse_ts(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    if isinstance(raw, str):
        # Tolerate trailing "Z" — fromisoformat accepts +00:00 but not Z
        # until 3.11, so normalize defensively for portability.
        s = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return None


def _iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    # Missing files are a normal cold-start condition, not an error.
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip corrupt lines rather than blowing up the endpoint;
                # the decision log itself captures justifications.
                continue


class FeedbackStore:
    """Lightweight reader/writer over the feedback and serving JSONL files."""

    def __init__(
        self,
        feedback_path: str,
        serving_path: str,
    ) -> None:
        self.feedback_path = feedback_path
        self.serving_path = serving_path

    def append(
        self,
        user_id: str,
        recipe_id: str,
        rating: float,
        timestamp: str | None = None,
    ) -> None:
        ts = timestamp or datetime.now(tz=timezone.utc).isoformat()
        record = {
            "user_id": user_id,
            "recipe_id": recipe_id,
            "rating": float(rating),
            "timestamp": ts,
        }
        # Ensure the parent directory exists — /logs is usually mounted, but
        # guard the non-Docker dev path where it may not be pre-created.
        Path(self.feedback_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def load_feedback(self, since_ts: datetime) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rec in _iter_jsonl(self.feedback_path):
            ts = _parse_ts(rec.get("timestamp"))
            if ts is None or ts < since_ts:
                continue
            out.append(rec)
        return out

    def load_served(self, since_ts: datetime) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rec in _iter_jsonl(self.serving_path):
            # Serving audit uses "served_at"; fall back to "timestamp" for
            # logs written by older versions of the serving API.
            ts = _parse_ts(rec.get("served_at") or rec.get("timestamp"))
            if ts is None or ts < since_ts:
                continue
            out.append(rec)
        return out

    def feedback_rate(self, since_ts: datetime) -> float:
        served = self.load_served(since_ts)
        feedback = self.load_feedback(since_ts)
        if not served:
            return 0.0
        return len(feedback) / len(served)

    def avg_rating(self, since_ts: datetime) -> float | None:
        feedback = self.load_feedback(since_ts)
        if not feedback:
            return None
        ratings = [float(r["rating"]) for r in feedback if "rating" in r]
        if not ratings:
            return None
        return sum(ratings) / len(ratings)

    def feedback_count(self, since_ts: datetime) -> int:
        return len(self.load_feedback(since_ts))

    def precision_at_k(
        self,
        window_start: datetime,
        window_end: datetime,
        k: int = 10,
    ) -> float:
        """Precision@k joined over (user_id, recipe_id).

        A joined pair counts toward the denominator only if the feedback
        timestamp is strictly after the serve timestamp — earlier feedback
        is about a prior serve of the same recipe and would inflate the
        metric. Numerator is the subset with rating >= 4.
        """
        served = self.load_served(window_start)
        feedback = [
            f for f in self.load_feedback(window_start)
            if _parse_ts(f.get("timestamp")) is not None
            and _parse_ts(f.get("timestamp")) <= window_end
        ]

        # Build a (user_id, recipe_id) -> list[(served_at, ...)] index so
        # the join is O(F) instead of O(S*F).
        served_index: dict[tuple[str, str], list[datetime]] = {}
        for s in served:
            uid = s.get("user_id")
            if uid is None:
                continue
            served_at = _parse_ts(s.get("served_at") or s.get("timestamp"))
            if served_at is None or served_at > window_end:
                continue
            recipe_ids = s.get("recipe_ids") or []
            # Only the top-k positions count for Precision@k.
            for rid in recipe_ids[:k]:
                served_index.setdefault((uid, str(rid)), []).append(served_at)

        denom = 0
        numer = 0
        for fb in feedback:
            uid = fb.get("user_id")
            rid = fb.get("recipe_id")
            if uid is None or rid is None:
                continue
            fb_ts = _parse_ts(fb.get("timestamp"))
            if fb_ts is None:
                continue
            served_times = served_index.get((uid, str(rid)))
            if not served_times:
                continue
            # Require at least one serve strictly before the feedback event.
            if not any(st < fb_ts for st in served_times):
                continue
            denom += 1
            try:
                if float(fb.get("rating", 0)) >= 4:
                    numer += 1
            except (TypeError, ValueError):
                continue

        if denom == 0:
            return 0.0
        return numer / denom
