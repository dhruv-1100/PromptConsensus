"""
request_coordinator.py
Deduplicates identical pipeline requests so stream retries can safely rejoin
or reuse the same result instead of running the full pipeline twice.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Tuple


RequestSource = Literal["new", "joined", "cached"]


@dataclass
class InflightRequest:
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: str | None = None


_LOCK = threading.Lock()
_INFLIGHT: Dict[str, InflightRequest] = {}
_COMPLETED: Dict[str, Tuple[float, Any]] = {}
_TTL_SECONDS = 300


def build_request_key(raw_query: str, domain: str, demo_mode: bool) -> str:
    payload = json.dumps(
        {
            "raw_query": raw_query.strip(),
            "domain": (domain or "general").strip().lower(),
            "demo_mode": bool(demo_mode),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prune_completed(now: float) -> None:
    expired = [key for key, (timestamp, _) in _COMPLETED.items() if now - timestamp > _TTL_SECONDS]
    for key in expired:
        _COMPLETED.pop(key, None)


def run_deduplicated(
    request_key: str,
    worker: Callable[[], Any],
) -> tuple[Any, RequestSource]:
    now = time.time()
    with _LOCK:
        _prune_completed(now)

        completed = _COMPLETED.get(request_key)
        if completed:
            return completed[1], "cached"

        inflight = _INFLIGHT.get(request_key)
        if inflight:
            source: RequestSource = "joined"
        else:
            inflight = InflightRequest()
            _INFLIGHT[request_key] = inflight
            source = "new"

    if source == "joined":
        inflight.event.wait()
        if inflight.error:
            raise RuntimeError(inflight.error)
        return inflight.result, "joined"

    try:
        result = worker()
    except Exception as exc:
        with _LOCK:
            inflight.error = str(exc)
            inflight.event.set()
            _INFLIGHT.pop(request_key, None)
        raise

    with _LOCK:
        inflight.result = result
        _COMPLETED[request_key] = (time.time(), result)
        inflight.event.set()
        _INFLIGHT.pop(request_key, None)

    return result, "new"
