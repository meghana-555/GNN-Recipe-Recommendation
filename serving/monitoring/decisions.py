"""Promotion/rollback decision logic for canary model versions.

Pure function over a metrics snapshot and a FeedbackStore; the caller
owns persistence so this module can be unit-tested without filesystem
side effects.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from feedback_store import FeedbackStore


# >5% errors indicates operational failure — well above normal tail of
# 4xx noise, and low enough to catch real regressions fast.
ERROR_RATE_ROLLBACK = 0.05

# 0.2 stars is ~10% of 5-star scale — user-visible
RATING_DROP_ROLLBACK = 0.2

# 20% engagement drop = users disengaging
FEEDBACK_RATE_DROP = 0.20

# conservative threshold to avoid over-promotion
RATING_IMPROVE_PROMOTE = 0.1

# below this, statistical signal too weak
MIN_FEEDBACK_COUNT = 50

# longer window for baseline stability
BASELINE_DAYS = 30

# canary comparison window
RECENT_DAYS = 7


def _rel_drop(recent: float, prior: float) -> float:
    """Relative drop from prior to recent, clamped at 0 when prior is 0.

    A zero prior means we have no baseline to compare against; returning 0
    prevents a spurious "100% drop" rollback during cold start.
    """
    if prior <= 0:
        return 0.0
    return max(0.0, (prior - recent) / prior)


def evaluate(
    rolling_snapshot: dict[str, Any],
    feedback: FeedbackStore,
    now: datetime,
) -> dict[str, Any]:
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    recent_start = now - timedelta(days=RECENT_DAYS)
    prior_start = now - timedelta(days=2 * RECENT_DAYS)
    baseline_start = now - timedelta(days=BASELINE_DAYS)

    recent_feedback_count = feedback.feedback_count(recent_start)
    recent_avg_rating = feedback.avg_rating(recent_start)
    recent_fb_rate = feedback.feedback_rate(recent_start)

    # Prior 7d window = [now-14d, now-7d); compute as "since prior_start"
    # minus "since recent_start" to avoid a second file scan over a raw
    # date range.
    prior_feedback_all = feedback.load_feedback(prior_start)
    prior_served_all = feedback.load_served(prior_start)
    prior_feedback = [
        f for f in prior_feedback_all
        if _ts(f.get("timestamp")) is not None and _ts(f.get("timestamp")) < recent_start
    ]
    prior_served = [
        s for s in prior_served_all
        if _ts(s.get("served_at") or s.get("timestamp")) is not None
        and _ts(s.get("served_at") or s.get("timestamp")) < recent_start
    ]
    prior_fb_rate = (len(prior_feedback) / len(prior_served)) if prior_served else 0.0
    prior_ratings = [float(f["rating"]) for f in prior_feedback if "rating" in f]
    prior_avg_rating = (sum(prior_ratings) / len(prior_ratings)) if prior_ratings else None

    baseline_avg_rating = feedback.avg_rating(baseline_start)

    error_rate = float(rolling_snapshot.get("error_rate") or 0.0)

    recent_p_at_10 = feedback.precision_at_k(recent_start, now, k=10)
    prior_p_at_10 = feedback.precision_at_k(prior_start, recent_start, k=10)

    metrics_for_return: dict[str, Any] = dict(rolling_snapshot)
    metrics_for_return["feedback_rate"] = recent_fb_rate
    metrics_for_return["avg_rating_on_recommended"] = recent_avg_rating

    ts_iso = now.astimezone(timezone.utc).isoformat()

    # Operational failures win over everything else — bad latency/errors
    # trump rating-based signals because users can't rate what they can't see.
    if error_rate > ERROR_RATE_ROLLBACK:
        return {
            "decision": "rollback",
            "justification": (
                f"error_rate {error_rate:.3f} exceeds threshold {ERROR_RATE_ROLLBACK}"
            ),
            "metrics": metrics_for_return,
            "timestamp": ts_iso,
        }

    # Below MIN_FEEDBACK_COUNT the rating- and engagement-based rules are
    # statistically unreliable, so hold rather than fire either direction.
    if recent_feedback_count < MIN_FEEDBACK_COUNT:
        return {
            "decision": "hold",
            "justification": (
                f"insufficient feedback (count={recent_feedback_count} "
                f"< MIN_FEEDBACK_COUNT={MIN_FEEDBACK_COUNT})"
            ),
            "metrics": metrics_for_return,
            "timestamp": ts_iso,
        }

    if (
        recent_avg_rating is not None
        and baseline_avg_rating is not None
        and (baseline_avg_rating - recent_avg_rating) > RATING_DROP_ROLLBACK
    ):
        return {
            "decision": "rollback",
            "justification": (
                f"avg_rating dropped {baseline_avg_rating - recent_avg_rating:.3f} "
                f"vs {BASELINE_DAYS}d baseline (> {RATING_DROP_ROLLBACK})"
            ),
            "metrics": metrics_for_return,
            "timestamp": ts_iso,
        }

    if _rel_drop(recent_fb_rate, prior_fb_rate) > FEEDBACK_RATE_DROP:
        return {
            "decision": "rollback",
            "justification": (
                f"feedback_rate dropped "
                f"{_rel_drop(recent_fb_rate, prior_fb_rate):.1%} vs prior "
                f"{RECENT_DAYS}d (> {FEEDBACK_RATE_DROP:.0%})"
            ),
            "metrics": metrics_for_return,
            "timestamp": ts_iso,
        }

    rating_improved = (
        recent_avg_rating is not None
        and prior_avg_rating is not None
        and (recent_avg_rating - prior_avg_rating) > RATING_IMPROVE_PROMOTE
    )
    precision_improved = recent_p_at_10 > prior_p_at_10

    if rating_improved and precision_improved:
        return {
            "decision": "promote",
            "justification": (
                f"precision@10 {recent_p_at_10:.3f} > prior {prior_p_at_10:.3f} "
                f"and avg_rating improved "
                f"{(recent_avg_rating or 0) - (prior_avg_rating or 0):.3f} "
                f"(> {RATING_IMPROVE_PROMOTE})"
            ),
            "metrics": metrics_for_return,
            "timestamp": ts_iso,
        }

    return {
        "decision": "hold",
        "justification": (
            f"no threshold crossed: error_rate={error_rate:.3f}, "
            f"recent_rating={recent_avg_rating}, prior_rating={prior_avg_rating}, "
            f"p@10_recent={recent_p_at_10:.3f}, p@10_prior={prior_p_at_10:.3f}"
        ),
        "metrics": metrics_for_return,
        "timestamp": ts_iso,
    }


def _ts(raw: Any) -> datetime | None:
    # Local re-implementation to avoid a cross-module import cycle; kept
    # minimal since feedback_store's parser is the canonical one.
    if not isinstance(raw, str):
        return None
    s = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
