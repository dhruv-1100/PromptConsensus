"""
Tests that the asyncio and LangGraph orchestrators are interchangeable.

Both engines are driven with stub agents (no network, no API key) and must
produce the same ConsensusState, the same progress events, and the same errors.
"""
import json
import time

import pytest

from agents.council import _consensus_diagnostics
from pipeline import graph as asyncio_engine
from pipeline import langgraph_graph as langgraph_engine
from pipeline.engine import AVAILABLE_ENGINES, get_pipeline_runner, resolve_engine_name

ENGINES = [asyncio_engine, langgraph_engine]
ENGINE_IDS = ["asyncio", "langgraph"]


# ─── Stub agents ─────────────────────────────────────────────────────────────

def stub_intent(raw_query, demo_mode=False, model_name=None):
    return {"topic_domain": "healthcare", "format_domain": "clinical document"}


def stub_rewriter(raw_query, intent, demo_mode=False, model_name=None):
    return json.dumps(
        {"optimised_prompt": f"prompt from {model_name}", "perspective_used": f"P-{model_name}"}
    )


def stub_review(raw_query, candidates, reviewer_models=None, demo_mode=False):
    labels = [f"Response {chr(65 + i)}" for i in range(len(candidates))]
    label_map = {label: name for label, (name, _) in zip(labels, candidates)}
    reviews = [
        {"reviewer": f"R{i}", "model": "stub", "evaluation": "fine", "parsed_ranking": labels}
        for i in range(3)
    ]
    aggregate = [
        {
            "label": label,
            "candidate": label_map[label].replace("Candidate ", ""),
            "average_rank": float(index + 1),
            "votes": 3,
            "ranks": [index + 1] * 3,
        }
        for index, label in enumerate(labels)
    ]
    return reviews, aggregate, label_map, _consensus_diagnostics(reviews, aggregate)


def stub_chairman(
    raw_query, candidates, peer_reviews, aggregate, label_map,
    topic_domain="general", chairman_model=None, demo_mode=False,
):
    return "synthesised prompt " * 10, {"model": "stub-chair", "rationale": "consensus"}


@pytest.fixture
def stubbed(monkeypatch):
    """Point both engines at the stub agents."""
    for module in ENGINES:
        monkeypatch.setattr(module, "extract_intent", stub_intent)
        monkeypatch.setattr(module, "peer_review_candidates", stub_review)
        monkeypatch.setattr(module, "chairman_synthesise_candidates", stub_chairman)


def specs_for(models, runner=stub_rewriter):
    specs = asyncio_engine.build_rewriter_specs_for_models(list(models))
    for spec in specs:
        spec["runner"] = runner
    return specs


def run(module, models=("m0", "m1", "m2"), **kwargs):
    return module.run_pipeline(
        "Write a discharge summary",
        "healthcare",
        demo_mode=False,
        rewriter_specs=specs_for(models),
        record_analytics=False,
        **kwargs,
    )


# ─── Engine selection ────────────────────────────────────────────────────────

class TestEngineSelection:
    def test_defaults_to_asyncio(self, monkeypatch):
        monkeypatch.delenv("PIPELINE_ENGINE", raising=False)
        assert resolve_engine_name() == "asyncio"

    def test_environment_sets_the_default(self, monkeypatch):
        monkeypatch.setenv("PIPELINE_ENGINE", "langgraph")
        assert resolve_engine_name() == "langgraph"

    def test_request_overrides_the_environment(self, monkeypatch):
        monkeypatch.setenv("PIPELINE_ENGINE", "langgraph")
        assert resolve_engine_name("asyncio") == "asyncio"

    @pytest.mark.parametrize("name", ["LangGraph", " langgraph ", "ASYNCIO"])
    def test_names_are_normalised(self, name):
        assert resolve_engine_name(name) in AVAILABLE_ENGINES

    def test_unknown_engine_raises(self):
        with pytest.raises(ValueError, match="Unknown pipeline engine"):
            resolve_engine_name("celery")

    @pytest.mark.parametrize("name,expected_module", [("asyncio", "graph"), ("langgraph", "langgraph_graph")])
    def test_returns_the_matching_runner(self, name, expected_module):
        runner, resolved = get_pipeline_runner(name)
        assert resolved == name
        assert runner.__module__.endswith(expected_module)


# ─── Behaviour of each engine ────────────────────────────────────────────────

@pytest.mark.parametrize("module", ENGINES, ids=ENGINE_IDS)
class TestEachEngine:
    def test_produces_a_complete_state(self, module, stubbed):
        state = run(module)
        assert state["raw_query"] == "Write a discharge summary"
        assert state["domain"] == "healthcare"
        assert state["intent"]["topic_domain"] == "healthcare"
        assert state["optimised_prompt"].startswith("synthesised prompt")
        assert state["chairman"]["model"] == "stub-chair"
        assert state["consensus_diagnostics"]["consensus_label"] == "high"

    def test_candidates_keep_the_configured_order(self, module, stubbed):
        state = run(module)
        assert state["candidate_order"] == ["Candidate A", "Candidate B", "Candidate C"]
        assert state["candidate_a"] == "prompt from m0"
        assert state["candidate_b"] == "prompt from m1"
        assert state["candidate_c"] == "prompt from m2"
        assert state["perspectives"]["Candidate A"] == "P-m0"

    def test_emits_the_progress_stages_the_ui_expects(self, module, stubbed):
        events = []
        run(module, progress_callback=events.append)
        assert [e["stage"] for e in events] == [
            "intent",
            "intent_complete",
            "rewriters",
            "rewriters_complete",
            "review",
            "review_complete",
            "chairman",
            "complete",
        ]
        assert [e["progress"] for e in events] == [10, 25, 30, 60, 65, 85, 88, 100]

    def test_supports_more_than_three_rewriters(self, module, stubbed):
        state = run(module, models=[f"m{i}" for i in range(5)])
        assert state["candidate_order"][-1] == "Candidate E"
        assert state["candidate_e"] == "prompt from m4"
        assert len(state["all_candidates"]) == 5

    def test_supports_a_single_rewriter(self, module, stubbed):
        state = run(module, models=["solo"])
        assert state["candidate_a"] == "prompt from solo"
        assert state["candidate_b"] == ""
        assert state["candidate_c"] == ""

    def test_rejects_an_empty_rewriter_list(self, module, stubbed):
        with pytest.raises((RuntimeError, ValueError)):
            module.run_pipeline("q", "general", rewriter_specs=[], record_analytics=False)

    def test_rejects_a_blank_candidate(self, module, stubbed):
        specs = specs_for(["m0", "m1"], runner=lambda *a, **k: "   ")
        with pytest.raises(RuntimeError, match="not all configured rewriter roles"):
            module.run_pipeline("q", "general", rewriter_specs=specs, record_analytics=False)

    def test_rejects_an_incomplete_review(self, module, stubbed, monkeypatch):
        def partial_review(raw_query, candidates, reviewer_models=None, demo_mode=False):
            reviews, aggregate, label_map, diagnostics = stub_review(raw_query, candidates)
            reviews[0]["parsed_ranking"] = reviews[0]["parsed_ranking"][:1]
            return reviews, aggregate, label_map, diagnostics

        monkeypatch.setattr(module, "peer_review_candidates", partial_review)
        with pytest.raises(RuntimeError, match="distinct ranked candidates"):
            run(module)

    def test_rewriter_failure_propagates(self, module, stubbed):
        def boom(*args, **kwargs):
            raise RuntimeError("OpenRouter call failed")

        specs = specs_for(["m0", "m1"], runner=boom)
        with pytest.raises(RuntimeError, match="OpenRouter call failed"):
            module.run_pipeline("q", "general", rewriter_specs=specs, record_analytics=False)

    def test_rewriters_run_in_parallel(self, module, stubbed):
        def slow(raw_query, intent, demo_mode=False, model_name=None):
            time.sleep(0.3)
            return stub_rewriter(raw_query, intent, demo_mode, model_name)

        specs = specs_for([f"m{i}" for i in range(4)], runner=slow)
        started = time.time()
        module.run_pipeline("q", "general", rewriter_specs=specs, record_analytics=False)
        # Serial execution would take ~1.2s.
        assert time.time() - started < 0.9

    def test_demo_mode_is_not_written_to_the_insights_log(self, module, tmp_path, monkeypatch):
        log = tmp_path / "insights.json"
        monkeypatch.setattr(asyncio_engine, "INSIGHTS_LOG_PATH", str(log))
        module.run_pipeline("Write a discharge summary", "healthcare", demo_mode=True)
        assert not log.exists()

    def test_live_runs_are_written_to_the_insights_log(self, module, stubbed, tmp_path, monkeypatch):
        log = tmp_path / "insights.json"
        monkeypatch.setattr(asyncio_engine, "INSIGHTS_LOG_PATH", str(log))
        module.run_pipeline(
            "q", "healthcare", demo_mode=False, rewriter_specs=specs_for(["m0", "m1", "m2"])
        )
        entries = json.loads(log.read_text())
        assert len(entries) == 1
        assert entries[0]["topic_domain"] == "healthcare"
        assert entries[0]["perspective_used"] == "P-m0"  # winner is Candidate A


# ─── Parity between the engines ──────────────────────────────────────────────

class TestEngineParity:
    @pytest.mark.parametrize("model_count", [1, 3, 5])
    def test_identical_state_for_identical_input(self, stubbed, model_count):
        models = [f"m{i}" for i in range(model_count)]
        states = {module.__name__: run(module, models=models) for module in ENGINES}
        asyncio_state, langgraph_state = states.values()

        assert sorted(asyncio_state) == sorted(langgraph_state)
        differing = [
            key for key in set(asyncio_state) | set(langgraph_state)
            if asyncio_state.get(key) != langgraph_state.get(key)
        ]
        assert differing == []

    def test_identical_progress_events(self, stubbed):
        collected = []
        for module in ENGINES:
            events = []
            run(module, progress_callback=events.append)
            collected.append(events)
        assert collected[0] == collected[1]

    def test_identical_demo_mode_output(self):
        # Demo fixtures exercise the real agent modules, unstubbed.
        states = [
            module.run_pipeline("Write a discharge summary", "healthcare", demo_mode=True)
            for module in ENGINES
        ]
        assert states[0] == states[1]
