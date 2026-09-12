"""
request_coordinator.py
Deduplicates identical pipeline requests so stream retries can safely rejoin
or reuse the same result instead of running the full pipeline twice.
"""
from __future__ import annotations

import hashlib
import json
import os
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

# A full pipeline is four sequential LLM stages, each capped at 120s, so a joined
# caller has to wait generously. It must still be bounded: without a timeout a
# leader thread that dies without signalling would block every joiner forever.
_JOIN_TIMEOUT_SECONDS = float(os.getenv("PIPELINE_JOIN_TIMEOUT_SECONDS", "600"))


def build_request_key(
    raw_query: str,
    domain: str,
    demo_mode: bool,
    engine: str = "asyncio",
) -> str:
    """
    Identity of a pipeline request. The engine is part of it so a run on one
    orchestrator never serves a cached result to a request for the other.
    """
    payload = json.dumps(
        {
            "raw_query": raw_query.strip(),
            "domain": (domain or "general").strip().lower(),
            "demo_mode": bool(demo_mode),
            "engine": (engine or "asyncio").strip().lower(),
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
        if not inflight.event.wait(_JOIN_TIMEOUT_SECONDS):
            raise RuntimeError(
                "Timed out waiting for an identical in-flight optimisation request to finish "
                f"after {_JOIN_TIMEOUT_SECONDS:g}s. Retry the request."
            )
        if inflight.error:
            raise RuntimeError(inflight.error)
        return inflight.result, "joined"

    # BaseException, not Exception: a cancelled or killed worker must still release
    # everyone waiting on this key.
    try:
        result = worker()
    except BaseException as exc:
        with _LOCK:
            inflight.error = str(exc) or exc.__class__.__name__
            _INFLIGHT.pop(request_key, None)
            inflight.event.set()
        raise

    with _LOCK:
        inflight.result = result
        _COMPLETED[request_key] = (time.time(), result)
        _INFLIGHT.pop(request_key, None)
        inflight.event.set()

    return result, "new"
