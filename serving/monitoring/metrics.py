"""In-memory rolling aggregators for serving telemetry.

Keeps bounded deques so the monitoring service stays flat-memory under
sustained traffic; percentiles and means are computed on snapshot read.
"""
# Assisted by Claude

from __future__ import annotations

import statistics
import time
from collections import deque
from typing import Any, Deque


RPM_WINDOW_SEC = 300
SNAPSHOT_SIZE = 1000


class RollingMetrics:
    """Bounded rolling window over the most recent /track events.

    The deques are capped at SNAPSHOT_SIZE because the /metrics contract
    defines "last 1000 requests" as the aggregation window; request_timestamps
    is unbounded-write but pruned on each read to keep rpm accurate without
    requiring a background sweeper.
    """

    def __init__(self, maxlen: int = SNAPSHOT_SIZE) -> None:
        self.latency_ms: Deque[float] = deque(maxlen=maxlen)
        self.predicted_scores: Deque[float] = deque(maxlen=maxlen)
        self.status_codes: Deque[int] = deque(maxlen=maxlen)
        # Unbounded here; pruned at read time against RPM_WINDOW_SEC.
        self.request_timestamps: Deque[float] = deque()

    def record_track(self, payload: dict[str, Any]) -> None:
        # Record wall-clock receipt time, not payload.timestamp, so rpm
        # reflects what the monitoring service actually observes.
        self.request_timestamps.append(time.time())

        latency = payload.get("latency_ms")
        if latency is not None:
            self.latency_ms.append(float(latency))

        status = payload.get("status_code")
        if status is not None:
            self.status_codes.append(int(status))

        scores = payload.get("predicted_scores") or []
        # Flatten per-recommendation so avg_predicted_score reflects the
        # per-item score distribution rather than per-request averages.
        for s in scores:
            try:
                self.predicted_scores.append(float(s))
            except (TypeError, ValueError):
                continue

    def _prune_timestamps(self, now: float) -> None:
        cutoff = now - RPM_WINDOW_SEC
        ts = self.request_timestamps
        while ts and ts[0] < cutoff:
            ts.popleft()

    def _percentile(self, data: list[float], pct: float) -> float | None:
        if not data:
            return None
        if len(data) == 1:
            return float(data[0])
        # statistics.quantiles with n=100 returns the 99 cut points between
        # percentiles 1..99; index pct-1 gives the requested percentile.
        qs = statistics.quantiles(data, n=100, method="inclusive")
        idx = min(max(int(pct) - 1, 0), len(qs) - 1)
        return float(qs[idx])

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        self._prune_timestamps(now)

        requests_in_window = len(self.request_timestamps)
        # Normalize to per-minute so the number stays stable as the window
        # length is tuned; 300s window / 60 = divide by 5.
        rpm = requests_in_window / (RPM_WINDOW_SEC / 60.0)

        latencies = list(self.latency_ms)
        scores = list(self.predicted_scores)
        statuses = list(self.status_codes)

        avg_score = float(sum(scores) / len(scores)) if scores else None
        p50 = self._percentile(sorted(latencies), 50) if latencies else None
        p95 = self._percentile(sorted(latencies), 95) if latencies else None

        if statuses:
            errors = sum(1 for s in statuses if s >= 400)
            error_rate = errors / len(statuses)
        else:
            error_rate = 0.0

        return {
            "requests_per_minute": rpm,
            "avg_predicted_score": avg_score,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "error_rate": error_rate,
            "window": {
                "requests": SNAPSHOT_SIZE,
                "rpm_window_sec": RPM_WINDOW_SEC,
                "feedback_days": 7,
            },
        }
