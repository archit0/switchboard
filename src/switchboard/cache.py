"""Exact-match response cache.

The cheapest API call is the one you never make. Repeated/identical prompts
(very common in agent loops and eval harnesses) return instantly at zero cost.
A semantic cache (embed the prompt, nearest-neighbour over past prompts) is the
natural next step — the gateway exposes `gemini-embedding-*` and
`text-embedding-3-*` for exactly this — but exact-match already captures the
biggest, safest wins without false hits.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any


def _key(messages: list[dict], mode: str) -> str:
    blob = json.dumps({"m": messages, "mode": mode}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, max_items: int = 4096, ttl_seconds: float | None = None):
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self._max = max_items
        self._ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def get(self, messages: list[dict], mode: str) -> Any | None:
        k = _key(messages, mode)
        with self._lock:
            item = self._store.get(k)
            if item is None:
                self.misses += 1
                return None
            ts, val = item
            if self._ttl is not None and (time.time() - ts) > self._ttl:
                del self._store[k]
                self.misses += 1
                return None
            self.hits += 1
            return val

    def put(self, messages: list[dict], mode: str, value: Any) -> None:
        k = _key(messages, mode)
        with self._lock:
            if len(self._store) >= self._max and k not in self._store:
                # drop oldest
                oldest = min(self._store, key=lambda x: self._store[x][0])
                del self._store[oldest]
            self._store[k] = (time.time(), value)
