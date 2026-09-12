"""
Endpoint-level tests. Every pipeline call runs in demo mode, so no model is
contacted and nothing is persisted.
"""
import json

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client():
    return TestClient(main.app)


class TestHealthAndConfig:
    def test_health(self, client):
        assert client.get("/api/health").json() == {"status": "ok", "service": "ConsensusPrompt"}

    def test_config_lists_models_and_engines(self, client):
        body = client.get("/api/config").json()
        assert set(body["models"]) >= {"intent_extractor", "rewriter_a", "reviewer_a", "chairman"}
        assert len(body["target_models"]) == 4
        assert body["available_engines"] == ["asyncio", "langgraph"]
        assert body["pipeline_engine"] in body["available_engines"]

    def test_config_survives_an_invalid_engine_setting(self, client, monkeypatch):
        monkeypatch.setenv("PIPELINE_ENGINE", "nonsense")
        response = client.get("/api/config")
        assert response.status_code == 200  # the UI still loads
        assert response.json()["pipeline_engine"] == "nonsense"


class TestCors:
    def test_wildcard_is_not_combined_with_credentials(self):
        # Browsers reject Access-Control-Allow-Origin: * alongside credentials.
        assert not ("*" in main.ALLOWED_ORIGINS and main.ALLOW_CREDENTIALS)

    def test_default_origins_are_the_dev_server(self):
        assert "http://localhost:3000" in main.ALLOWED_ORIGINS


class TestOptimize:
    @pytest.mark.parametrize("engine", [None, "asyncio", "langgraph"])
    def test_returns_the_full_pipeline_state(self, client, engine):
        body = {"raw_query": f"Write a discharge summary ({engine})", "domain": "healthcare", "demo_mode": True}
        if engine:
            body["engine"] = engine

        data = client.post("/api/optimize", json=body).json()
        assert set(data) == {
            "raw_query", "intent", "candidate_a", "candidate_b", "candidate_c",
            "peer_reviews", "aggregate_rankings", "label_map", "consensus_diagnostics",
            "chairman", "perspectives", "optimised_prompt",
        }
        assert len(data["peer_reviews"]) == 3
        assert data["optimised_prompt"]
        assert data["perspectives"]

    def test_unknown_engine_is_a_client_error(self, client):
        response = client.post("/api/optimize", json={"raw_query": "q", "engine": "celery"})
        assert response.status_code == 400
        assert "Unknown pipeline engine" in response.json()["detail"]

    def test_both_engines_agree(self, client):
        query = "Write a discharge summary for engine comparison"
        results = {
            engine: client.post(
                "/api/optimize",
                json={"raw_query": query, "domain": "healthcare", "demo_mode": True, "engine": engine},
            ).json()
            for engine in ("asyncio", "langgraph")
        }
        assert results["asyncio"] == results["langgraph"]

    def test_missing_raw_query_is_rejected(self, client):
        assert client.post("/api/optimize", json={"domain": "healthcare"}).status_code == 422


class TestOptimizeStream:
    @pytest.mark.parametrize("engine", ["asyncio", "langgraph"])
    def test_streams_every_stage_then_the_result(self, client, engine):
        events = []
        with client.stream(
            "POST",
            "/api/optimize/stream",
            json={
                "raw_query": f"Stream a discharge summary via {engine}",
                "domain": "healthcare",
                "demo_mode": True,
                "engine": engine,
            },
        ) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

        types = [event["type"] for event in events]
        assert types[0] == "start"
        assert events[0]["engine"] == engine
        assert types[-1] == "result"
        assert [e["stage"] for e in events if e["type"] == "progress"] == [
            "intent", "intent_complete", "rewriters", "rewriters_complete",
            "review", "review_complete", "chairman", "complete",
        ]
        assert events[-1]["data"]["optimised_prompt"]

    def test_reports_a_bad_engine_before_streaming(self, client):
        response = client.post("/api/optimize/stream", json={"raw_query": "q", "engine": "celery"})
        assert response.status_code == 400


class TestSafetyCheck:
    def test_flags_a_risky_healthcare_prompt(self, client):
        response = client.post(
            "/api/safety-check",
            json={
                "final_prompt": "List every medication and dose for this patient in a table.",
                "domain": "healthcare",
            },
        )
        body = response.json()
        assert body["risk_level"] == "high"
        assert body["requires_acknowledgement"] is True
        assert body["sensitive_domain"] is True

    def test_a_clean_general_prompt_raises_nothing(self, client):
        body = client.post(
            "/api/safety-check",
            json={"final_prompt": " ".join(["word"] * 80) + " with sections", "domain": "general"},
        ).json()
        assert body["risk_level"] == "none"
        assert body["checks"] == []
        assert body["requires_acknowledgement"] is False
        assert body["sensitive_domain"] is False


class TestFeedback:
    def test_demo_feedback_is_not_persisted(self, client, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("demo runs must not be written to disk")

        monkeypatch.setattr(main, "append_feedback_entry", fail)
        monkeypatch.setattr(main, "append_session_entry", fail)

        body = client.post(
            "/api/feedback",
            json={"quality": 5, "improvement": 5, "trust": 5, "control": 5, "demo_mode": True},
        ).json()
        assert body["status"] == "demo_mode_skipped"
        assert body["session_id"] == ""

    def test_live_feedback_is_recorded(self, client, monkeypatch):
        captured = {}

        def fake_feedback(entry):
            captured["feedback"] = entry
            return entry

        def fake_session(entry):
            captured["session"] = entry
            return {**entry, "session_id": "sess-1"}

        monkeypatch.setattr(main, "append_feedback_entry", fake_feedback)
        monkeypatch.setattr(main, "append_session_entry", fake_session)

        body = client.post(
            "/api/feedback",
            json={
                "quality": 4, "improvement": 5, "trust": 5, "control": 5,
                "raw_query": "q", "domain": "healthcare", "demo_mode": False,
                "peer_reviews": [{"reviewer": "R1"}],
            },
        ).json()

        assert body["status"] == "recorded_to_json"
        assert body["session_id"] == "sess-1"
        assert captured["feedback"]["trust"] == 5
        assert captured["session"]["peer_reviews"] == [{"reviewer": "R1"}]

    @pytest.mark.parametrize("missing", ["quality", "trust"])
    def test_ratings_are_required(self, client, missing):
        payload = {"quality": 5, "improvement": 5, "trust": 5, "control": 5}
        payload.pop(missing)
        assert client.post("/api/feedback", json=payload).status_code == 422


class TestSessionRoutes:
    def test_sessions_list_is_read_only(self, client, monkeypatch):
        monkeypatch.setattr(main, "list_sessions", lambda: [{"session_id": "a"}])
        assert client.get("/api/sessions").json() == {"sessions": [{"session_id": "a"}]}

    def test_json_export_is_an_attachment(self, client, monkeypatch):
        monkeypatch.setattr(main, "list_sessions", lambda: [{"session_id": "a"}])
        response = client.get("/api/sessions/export/json")
        assert response.headers["content-type"].startswith("application/json")
        assert "attachment" in response.headers["content-disposition"]
        assert json.loads(response.text) == [{"session_id": "a"}]

    def test_csv_export_is_an_attachment(self, client, monkeypatch):
        monkeypatch.setattr(main, "export_sessions_csv", lambda: "col\nvalue\n")
        response = client.get("/api/sessions/export/csv")
        assert response.headers["content-type"].startswith("text/csv")
        assert response.text == "col\nvalue\n"
