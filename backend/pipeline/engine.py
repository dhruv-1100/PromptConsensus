"""
pipeline/engine.py
Chooses which orchestrator runs the pipeline.

Two engines implement the same five stages over the same agent functions:

  asyncio   (default) — pipeline/graph.py, hand-rolled asyncio.gather fan-out
  langgraph            — pipeline/langgraph_graph.py, a LangGraph StateGraph

The default comes from PIPELINE_ENGINE and a request may override it per call,
which makes the two directly comparable on identical inputs.
"""
from __future__ import annotations

import os
from typing import Callable

DEFAULT_ENGINE = "asyncio"
AVAILABLE_ENGINES = ("asyncio", "langgraph")


def resolve_engine_name(requested: str | None = None) -> str:
    """Normalise an engine name from the request, else the environment, else the default."""
    name = (requested or os.getenv("PIPELINE_ENGINE") or DEFAULT_ENGINE).strip().lower()
    if name not in AVAILABLE_ENGINES:
        raise ValueError(
            f"Unknown pipeline engine '{name}'. Choose one of: {', '.join(AVAILABLE_ENGINES)}."
        )
    return name


def get_pipeline_runner(requested: str | None = None) -> tuple[Callable[..., dict], str]:
    """Return (run_pipeline, engine_name) for the selected engine."""
    name = resolve_engine_name(requested)

    if name == "langgraph":
        from pipeline.langgraph_graph import run_pipeline
    else:
        from pipeline.graph import run_pipeline

    return run_pipeline, name
