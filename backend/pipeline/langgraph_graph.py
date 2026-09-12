"""
pipeline/langgraph_graph.py
The same five-stage ConsensusPrompt pipeline expressed as a LangGraph StateGraph.

This is an alternative orchestrator, not a replacement: pipeline/graph.py remains
the default engine, and both call the identical agent functions, so a run through
either produces the same ConsensusState. Select this one with
PIPELINE_ENGINE=langgraph or by passing engine="langgraph" to /api/optimize.

Graph shape:

    START → intent ─┬→ rewrite (one parallel task per rewriter spec) ─┐
                    ├→ rewrite                                        ├→ collect
                    └→ rewrite ───────────────────────────────────────┘
                                                                          ↓
                                              END ← chairman ← council ←──┘

The rewriter fan-out uses Send, so each configured rewriter is a real parallel
task in one superstep rather than a hand-rolled gather.
"""
from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Any, Callable, Dict, List

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from agents.council import chairman_synthesise_candidates, peer_review_candidates
from agents.intent_extractor import extract_intent
from live_mode_utils import extract_prompt_and_perspective
from pipeline.graph import (
    DEFAULT_REWRITER_SPECS,
    _validate_candidate_outputs,
    _validate_review_outputs,
    record_optimisation_insight,
)
from pipeline.state import ConsensusState

try:  # langgraph >= 0.2
    from langgraph.types import Send
except ImportError:  # pragma: no cover - older langgraph layout
    from langgraph.constants import Send

load_dotenv()


class LangGraphState(ConsensusState, total=False):
    """ConsensusState plus the fan-in channel the parallel rewriters write to."""

    # Each parallel rewrite task appends one {"index", "text"} record; operator.add
    # merges the branches back together in arbitrary completion order.
    rewriter_outputs: Annotated[List[Dict[str, Any]], operator.add]


def build_pipeline_graph(
    rewriter_specs: List[Dict[str, Any]],
    *,
    demo_mode: bool = False,
    intent_model: str | None = None,
    reviewer_models: List[str] | None = None,
    chairman_model: str | None = None,
    notify: Callable[[str, str, int], None] | None = None,
):
    """
    Compile the pipeline as a StateGraph.

    Per-run configuration (models, demo mode, the progress callback) is captured in
    closures rather than stored in state: the specs carry Python callables, which do
    not belong in a serialisable channel.
    """
    def announce(stage_key: str, message: str, pct: int) -> None:
        if notify:
            notify(stage_key, message, pct)

    async def intent_node(state: LangGraphState) -> Dict[str, Any]:
        announce("intent", "Extracting intent from query", 10)
        intent = await asyncio.to_thread(
            extract_intent,
            state["raw_query"],
            demo_mode,
            intent_model,
        )
        announce("intent_complete", "Intent extraction complete", 25)
        return {"intent": intent}

    def fan_out_rewriters(state: LangGraphState) -> List[Send]:
        announce(
            "rewriters",
            f"Rewriting with {len(rewriter_specs)} parallel strategies",
            30,
        )
        return [
            Send(
                "rewrite",
                {
                    "spec_index": index,
                    "raw_query": state["raw_query"],
                    "intent": state.get("intent", {}),
                },
            )
            for index in range(len(rewriter_specs))
        ]

    async def rewrite_node(payload: Dict[str, Any]) -> Dict[str, Any]:
        index = payload["spec_index"]
        spec = rewriter_specs[index]
        text = await asyncio.to_thread(
            spec["runner"],
            payload["raw_query"],
            payload["intent"],
            demo_mode,
            spec.get("model_name"),
        )
        return {"rewriter_outputs": [{"index": index, "text": text}]}

    def collect_node(state: LangGraphState) -> Dict[str, Any]:
        # Branches finish out of order, so restore the configured spec order before
        # anything downstream sees the candidates.
        outputs = sorted(state.get("rewriter_outputs", []), key=lambda item: item["index"])
        texts = [item["text"] for item in outputs]

        raw_candidates = {
            rewriter_specs[item["index"]]["candidate_name"]: item["text"]
            for item in outputs
        }
        _validate_candidate_outputs(raw_candidates)

        update: Dict[str, Any] = {
            "candidate_a": "",
            "candidate_b": "",
            "candidate_c": "",
            "all_candidates": {},
            "candidate_order": [],
            "perspectives": {},
        }
        for spec, candidate_text in zip(rewriter_specs, texts):
            prompt, perspective = extract_prompt_and_perspective(
                candidate_text,
                spec["perspective"],
                step=f"rewriter_{str(spec['agent_key']).lower()}",
            )
            update[f"candidate_{str(spec['agent_key']).lower()}"] = prompt
            update["all_candidates"][spec["candidate_name"]] = prompt
            update["candidate_order"].append(spec["candidate_name"])
            update["perspectives"][spec["candidate_name"]] = perspective

        announce("rewriters_complete", "All rewriters finished", 60)
        return update

    async def council_node(state: LangGraphState) -> Dict[str, Any]:
        announce("review", "Council: anonymised peer review in progress", 65)
        candidates = _candidates_for_review(state)

        reviews, aggregate, label_map, diagnostics = await asyncio.to_thread(
            peer_review_candidates,
            state["raw_query"],
            candidates,
            reviewer_models,
            demo_mode,
        )
        _validate_review_outputs(reviews, len(candidates))
        announce("review_complete", "Council review complete — aggregating rankings", 85)
        return {
            "peer_reviews": reviews,
            "aggregate_rankings": aggregate,
            "label_map": label_map,
            "consensus_diagnostics": diagnostics,
        }

    async def chairman_node(state: LangGraphState) -> Dict[str, Any]:
        announce("chairman", "Chairman synthesising final prompt", 88)
        intent = state.get("intent", {})
        optimised_prompt, chairman_info = await asyncio.to_thread(
            chairman_synthesise_candidates,
            state["raw_query"],
            _candidates_for_review(state),
            state.get("peer_reviews", []),
            state.get("aggregate_rankings", []),
            state.get("label_map", {}),
            intent.get("topic_domain", state.get("domain", "general")),
            chairman_model,
            demo_mode,
        )
        announce("complete", "Consensus reached", 100)
        return {"chairman": chairman_info, "optimised_prompt": optimised_prompt}

    builder = StateGraph(LangGraphState)
    builder.add_node("intent", intent_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("collect", collect_node)
    builder.add_node("council", council_node)
    builder.add_node("chairman", chairman_node)

    builder.add_edge(START, "intent")
    builder.add_conditional_edges("intent", fan_out_rewriters, ["rewrite"])
    builder.add_edge("rewrite", "collect")
    builder.add_edge("collect", "council")
    builder.add_edge("council", "chairman")
    builder.add_edge("chairman", END)

    return builder.compile()


def _hydrate_candidate_aliases(state: Dict[str, Any]) -> None:
    """
    pipeline.graph writes one candidate_<letter> key per rewriter, but the typed
    state channels only carry a/b/c. Restore the rest (runs with more than three
    rewriters) so both engines return exactly the same keys.
    """
    all_candidates = state.get("all_candidates", {})
    for name in state.get("candidate_order", []):
        letter = name.replace("Candidate ", "").strip().lower()
        if letter:
            state[f"candidate_{letter}"] = all_candidates.get(name, "")


def _candidates_for_review(state: Dict[str, Any]) -> List[tuple[str, str]]:
    """Rebuild the (candidate_name, prompt) pairs the council agents expect."""
    all_candidates = state.get("all_candidates", {})
    order = state.get("candidate_order", []) or list(all_candidates.keys())
    return [(name, all_candidates[name]) for name in order if name in all_candidates]


def run_pipeline(
    raw_query: str,
    domain: str = "general",
    demo_mode: bool = False,
    progress_callback=None,
    *,
    intent_model: str | None = None,
    rewriter_specs: List[Dict[str, Any]] | None = None,
    reviewer_models: List[str] | None = None,
    chairman_model: str | None = None,
    record_analytics: bool = True,
) -> ConsensusState:
    """
    Execute the full ConsensusPrompt pipeline (S1 → S3a/b/c) through LangGraph.

    Signature and return value match pipeline.graph.run_pipeline so the two engines
    are interchangeable behind the API.
    """
    # None means "use the defaults"; an explicitly empty list is a caller error,
    # which the `or` idiom used to swallow by silently running the default three.
    active_rewriter_specs = DEFAULT_REWRITER_SPECS if rewriter_specs is None else rewriter_specs
    if not active_rewriter_specs:
        raise RuntimeError("At least one rewriter must be configured.")

    def notify(stage_key: str, message: str, pct: int) -> None:
        if progress_callback:
            progress_callback({"stage": stage_key, "message": message, "progress": pct})

    graph = build_pipeline_graph(
        list(active_rewriter_specs),
        demo_mode=demo_mode,
        intent_model=intent_model,
        reviewer_models=reviewer_models,
        chairman_model=chairman_model,
        notify=notify,
    )

    # A fresh loop keeps this callable from a plain thread (FastAPI's sync route
    # pool, asyncio.to_thread, the ablation runner) exactly like the default engine.
    loop = asyncio.new_event_loop()
    try:
        final_state = loop.run_until_complete(
            graph.ainvoke({"raw_query": raw_query, "domain": domain})
        )
    finally:
        loop.close()

    state: ConsensusState = {
        key: value for key, value in final_state.items() if key != "rewriter_outputs"
    }
    _hydrate_candidate_aliases(state)

    # Matches pipeline.graph: synthetic demo runs stay out of the insights log.
    if record_analytics and not demo_mode:
        record_optimisation_insight(
            state,
            state.get("intent", {}),
            state.get("aggregate_rankings", []),
        )

    return state
