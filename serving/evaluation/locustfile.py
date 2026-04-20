"""Locust load test for the GNN-based recipe recommendation API.

Run with::

    locust -f locustfile.py --host http://localhost:8000

Or headless::

    locust -f locustfile.py --host http://localhost:8000 \
        --headless -u 100 -r 10 --run-time 60s

Configuration via environment variables
----------------------------------------
USER_POOL_SIZE : int
    Number of synthetic user ids (``user_0`` .. ``user_{N-1}``).  Default 1000.
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, tag, task


_USER_POOL_SIZE: int = int(os.environ.get("USER_POOL_SIZE", "1000"))


def _random_user_id() -> str:
    """Return a random user id from the configured pool."""
    return f"user:{random.randint(0, _USER_POOL_SIZE - 1)}"


class RecommendationUser(HttpUser):
    """Simulated user that requests recipe recommendations.

    Task weights
    -------------
    * ``POST /recommend`` — 95 %
    * ``GET  /health``    —  5 %

    Wait time between requests is 1-3 seconds, giving roughly
    2-5 requests/second per simulated user at peak.
    """

    wait_time = between(1, 3)

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    @tag("recommend")
    @task(95)
    def get_recommendation(self) -> None:
        """Request recommendations for a random user."""
        user_id = _random_user_id()
        with self.client.post(
            "/recommend",
            json={"user_id": user_id},
            name="/recommend",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                body = response.json()
                if "recommendations" not in body:
                    response.failure("Response missing 'recommendations' key")
                elif not isinstance(body["recommendations"], list):
                    response.failure("'recommendations' is not a list")
                else:
                    response.success()
            else:
                response.failure(
                    f"Unexpected status {response.status_code}: "
                    f"{response.text[:200]}"
                )

    @tag("health")
    @task(5)
    def health_check(self) -> None:
        """Hit the health endpoint."""
        with self.client.get(
            "/health",
            name="/health",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                body = response.json()
                if body.get("status") == "healthy":
                    response.success()
                else:
                    response.failure(f"Unhealthy status: {body}")
            else:
                response.failure(
                    f"Health check failed with status {response.status_code}"
                )
