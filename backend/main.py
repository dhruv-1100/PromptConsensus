"""
main.py — ConsensusPrompt FastAPI Backend
Exposes the multi-agent prompt optimisation pipeline as REST endpoints.
"""
import os
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from feedback_memory import append_feedback_entry
from session_store import append_session_entry, list_sessions, export_sessions_csv, get_session_analytics
from safety_checks import run_safety_checks
from request_coordinator import build_request_key, run_deduplicated
from idiosyncrasy_detector import candidate_diversity_report
from preference_pairs import export_preferences_jsonl, get_preference_stats

load_dotenv()

app = FastAPI(
    title="ConsensusPrompt API",
    description="Multi-agent prompt optimisation middleware",
    version="1.0.0",
)

# CORS — allow Next.js dev + production origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ───────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    raw_query: str
    domain: str = "general"
    demo_mode: bool = True

class ExecuteRequest(BaseModel):
    final_prompt: str
    target_model: str = "tencent/hy3-preview:free"
    demo_mode: bool = True

class SafetyCheckRequest(BaseModel):
    final_prompt: str
    domain: str = "general"
    raw_query: str = ""
    intent: dict = {}

class FeedbackRequest(BaseModel):
    quality: int
    improvement: int
    trust: int
    control: int
    text: str = ""
    raw_query: str = ""
    domain: str = "general"
    optimised_prompt: str = ""
    final_prompt: str = ""
    llm_response: str = ""
    baseline_response: str = ""
    target_model: str = ""
    chairman_model: str = ""
    compare_mode: bool = False
    safety_report: dict = {}
    safety_acknowledged: bool = False
    intervention_labels: list[str] = []
    intent: dict = {}
    candidate_a: str = ""
    candidate_b: str = ""
    candidate_c: str = ""
    peer_reviews: list = []
    aggregate_rankings: list = []
    label_map: dict = {}
    consensus_diagnostics: dict = {}
    perspectives: dict = {}
    chairman: dict = {}
    demo_mode: bool = True


def format_pipeline_response(state: dict) -> dict:
    """Normalize pipeline state for frontend consumption."""
    return {
        "raw_query": state.get("raw_query", ""),
        "intent": state.get("intent", {}),
        "candidate_a": state.get("candidate_a", ""),
        "candidate_b": state.get("candidate_b", ""),
        "candidate_c": state.get("candidate_c", ""),
        "peer_reviews": state.get("peer_reviews", []),
        "aggregate_rankings": state.get("aggregate_rankings", []),
        "label_map": state.get("label_map", {}),
        "consensus_diagnostics": state.get("consensus_diagnostics", {}),
        "chairman": state.get("chairman", {}),
        "perspectives": state.get("perspectives", {}),
        "optimised_prompt": state.get("optimised_prompt", ""),
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "ConsensusPrompt"}


@app.get("/api/config")
def get_config():
    from config import MODELS, TARGET_MODELS
    return {"models": MODELS, "target_models": TARGET_MODELS}

@app.post("/api/optimize")
def optimize(req: OptimizeRequest):
    """
    Run the full S1→S3 pipeline:
    Intent extraction → 3 parallel rewrites → arbitration.
    Returns full pipeline state for the frontend to render.
    """
    from pipeline.graph import run_pipeline

    try:
        request_key = build_request_key(req.raw_query, req.domain, req.demo_mode)
        state, _ = run_deduplicated(
            request_key,
            lambda: run_pipeline(
                raw_query=req.raw_query,
                domain=req.domain,
                demo_mode=req.demo_mode,
            ),
        )
        return format_pipeline_response(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/optimize/stream")
async def optimize_stream(req: OptimizeRequest):
    """
    Run the full pipeline and stream progress events to the frontend.
    """
    from pipeline.graph import run_pipeline

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def progress_callback(event: dict):
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", **event})

        async def run_and_publish():
            try:
                request_key = build_request_key(req.raw_query, req.domain, req.demo_mode)
                state, source = await asyncio.to_thread(
                    run_deduplicated,
                    request_key,
                    lambda: run_pipeline(
                        req.raw_query,
                        req.domain,
                        req.demo_mode,
                        progress_callback,
                    ),
                )
                if source == "joined":
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {
                            "type": "progress",
                            "stage": "review_complete",
                            "message": "Rejoined an identical in-flight run",
                            "progress": 92,
                        },
                    )
                elif source == "cached":
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {
                            "type": "progress",
                            "stage": "complete",
                            "message": "Reused the most recent identical result",
                            "progress": 98,
                        },
                    )
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "result", "data": format_pipeline_response(state)},
                )
            except Exception as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "error", "message": str(exc)},
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "done"})

        task = asyncio.create_task(run_and_publish())
        
        # Send 2KB of padding to bypass Nginx/Vercel/browser initial buffering
        yield f": {' ' * 2048}\n\n"
        yield f"data: {json.dumps({'type': 'start', 'message': 'Connecting to backend', 'progress': 2})}\n\n"

        while True:
            event = await queue.get()
            if event.get("type") == "done":
                break
            yield f"data: {json.dumps(event)}\n\n"

        await task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/execute")
def execute(req: ExecuteRequest):
    """
    S5: Execute the user-approved prompt against the chosen LLM.
    Returns the LLM response.
    """
    from pipeline.graph import execute_prompt

    try:
        response = execute_prompt(
            final_prompt=req.final_prompt,
            target_model=req.target_model,
            demo_mode=req.demo_mode,
        )
        return {"response": response}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/safety-check")
def safety_check(req: SafetyCheckRequest):
    """
    Run lightweight pre-execution safety checks for sensitive domains.
    """
    return run_safety_checks(
        final_prompt=req.final_prompt,
        domain=req.domain,
        raw_query=req.raw_query,
        intent=req.intent,
    )


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    """
    Record user feedback and session context to a local JSON file.
    Demo-mode runs are NOT persisted — only live runs are stored.
    """
    # Skip persistence for demo runs
    if req.demo_mode:
        return {
            "status": "demo_mode_skipped",
            "ratings": {"quality": req.quality, "improvement": req.improvement, "trust": req.trust, "control": req.control},
            "session_id": "",
        }

    feedback_entry = {
        "quality": req.quality,
        "improvement": req.improvement,
        "trust": req.trust,
        "control": req.control,
        "text": req.text,
        "raw_query": req.raw_query,
        "domain": req.domain,
        "optimised_prompt": req.optimised_prompt,
        "final_prompt": req.final_prompt,
        "llm_response": req.llm_response,
        "baseline_response": req.baseline_response,
        "target_model": req.target_model,
        "chairman_model": req.chairman_model,
        "compare_mode": req.compare_mode,
        "safety_report": req.safety_report,
        "safety_acknowledged": req.safety_acknowledged,
        "intervention_labels": req.intervention_labels,
    }
    entry = append_feedback_entry(feedback_entry)
    session_entry = append_session_entry({
        **feedback_entry,
        "intent": req.intent,
        "candidate_a": req.candidate_a,
        "candidate_b": req.candidate_b,
        "candidate_c": req.candidate_c,
        "peer_reviews": req.peer_reviews,
        "aggregate_rankings": req.aggregate_rankings,
        "label_map": req.label_map,
        "consensus_diagnostics": req.consensus_diagnostics,
        "perspectives": req.perspectives,
        "chairman": req.chairman,
    })

    return {
        "status": "recorded_to_json",
        "ratings": entry,
        "session_id": session_entry["session_id"],
    }


@app.get("/api/sessions")
def sessions():
    """Return recent stored sessions for inspection."""
    return {"sessions": list_sessions()}


@app.get("/api/sessions/analytics")
def session_analytics():
    """Return summary metrics for the homepage analytics panel."""
    return get_session_analytics()


@app.get("/api/sessions/export/json")
def export_sessions_json():
    """Download all stored sessions as JSON."""
    payload = json.dumps(list_sessions(), indent=2, ensure_ascii=True)
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="consensusprompt-sessions.json"'},
    )


@app.get("/api/sessions/export/csv")
def export_sessions_csv_route():
    """Download flattened study sessions as CSV."""
    return Response(
        content=export_sessions_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="consensusprompt-sessions.csv"'},
    )



@app.get("/api/sessions/export/preferences")
def export_preferences():
    """Download DPO-format preference pairs as JSONL."""
    return Response(
        content=export_preferences_jsonl(),
        media_type="application/jsonl",
        headers={"Content-Disposition": 'attachment; filename="consensusprompt-preferences.jsonl"'},
    )


@app.get("/api/sessions/preference-stats")
def preference_stats():
    """Return aggregate statistics about the preference dataset."""
    return get_preference_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
