"""
main.py — ConsensusPrompt FastAPI Backend
Exposes the multi-agent prompt optimisation pipeline as REST endpoints.
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

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
    target_model: str = "gpt-4o"
    demo_mode: bool = True

class FeedbackRequest(BaseModel):
    quality: int
    improvement: int
    trust: int
    control: int
    text: str = ""


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "ConsensusPrompt"}


@app.post("/api/optimize")
def optimize(req: OptimizeRequest):
    """
    Run the full S1→S3 pipeline:
    Intent extraction → 3 parallel rewrites → arbitration.
    Returns full pipeline state for the frontend to render.
    """
    from pipeline.graph import run_pipeline

    state = run_pipeline(
        raw_query=req.raw_query,
        domain=req.domain,
        demo_mode=req.demo_mode,
    )

    return {
        "raw_query": state.get("raw_query", ""),
        "intent": state.get("intent", {}),
        "candidate_a": state.get("candidate_a", ""),
        "candidate_b": state.get("candidate_b", ""),
        "candidate_c": state.get("candidate_c", ""),
        "peer_reviews": state.get("peer_reviews", []),
        "aggregate_rankings": state.get("aggregate_rankings", []),
        "label_map": state.get("label_map", {}),
        "chairman": state.get("chairman", {}),
        "perspectives": state.get("perspectives", {}),
        "optimised_prompt": state.get("optimised_prompt", ""),
    }


@app.post("/api/execute")
def execute(req: ExecuteRequest):
    """
    S5: Execute the user-approved prompt against the chosen LLM.
    Returns the LLM response.
    """
    from pipeline.graph import execute_prompt

    response = execute_prompt(
        final_prompt=req.final_prompt,
        target_model=req.target_model,
        demo_mode=req.demo_mode,
    )

    return {"response": response}


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    """
    Record user feedback to a local JSON file.
    """
    import json
    import os
    from datetime import datetime

    feedback_file = "feedback.json"
    
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "quality": req.quality,
        "improvement": req.improvement,
        "trust": req.trust,
        "control": req.control,
        "text": req.text,
    }

    data = []
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, "r") as f:
                data = json.load(f)
        except Exception:
            pass
            
    data.append(entry)
    with open(feedback_file, "w") as f:
        json.dump(data, f, indent=4)

    return {
        "status": "recorded_to_json",
        "ratings": entry,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
