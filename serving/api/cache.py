"""Swappable cache abstraction for pre-computed recommendations.

Supports two backends:
  - "redis"  : async Redis via the ``redis`` (redis-py >= 4.2) async client
  - "memory" : plain Python dict (useful for local dev / single-process deploys)

Values are stored and returned as raw JSON strings so the caller is
responsible for serialisation.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Optional

import redis.asyncio as aioredis


class CacheBackend(ABC):
    """Minimal async-compatible cache interface."""

    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        """Return the cached JSON string for *key*, or ``None``."""

    @abstractmethod
    async def set(self, key: str, value: str) -> None:  # noqa: A003
        """Store a JSON string under *key*."""


class RedisCache(CacheBackend):
    """Thin async wrapper around a Redis connection."""

    def __init__(self, host: str = "localhost", port: int = 6379) -> None:
        self._client: aioredis.Redis = aioredis.Redis(
            host=host,
            port=port,
            decode_responses=True,
        )

    async def get(self, key: str) -> Optional[str]:
        return await self._client.get(key)

    async def set(self, key: str, value: str) -> None:  # noqa: A003
        await self._client.set(key, value)

    async def close(self) -> None:
        await self._client.aclose()


class MemoryCache(CacheBackend):
    """Simple dict-backed cache — no TTL, no eviction."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    async def set(self, key: str, value: str) -> None:  # noqa: A003
        self._store[key] = value

    # Convenience: bulk-load pre-computed recommendations at startup
    def load_bulk(self, data: dict[str, Any]) -> None:
        """Load a mapping of ``{user_id: recommendations_payload}``."""
        for k, v in data.items():
            self._store[k] = v if isinstance(v, str) else json.dumps(v)


def create_cache(backend_type: str, **kwargs: Any) -> CacheBackend:
    """Instantiate the requested cache backend.

    Parameters
    ----------
    backend_type:
        ``"redis"`` or ``"memory"``.
    **kwargs:
        Forwarded to the backend constructor (e.g. ``host``, ``port``).
    """
    backend_type = backend_type.lower().strip()
    if backend_type == "redis":
        return RedisCache(
            host=kwargs.get("host", "localhost"),
            port=int(kwargs.get("port", 6379)),
        )
    if backend_type == "memory":
        return MemoryCache()
    raise ValueError(
        f"Unknown cache backend {backend_type!r}. Choose 'redis' or 'memory'."
    )
