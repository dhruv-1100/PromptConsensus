"""
agents/intent_extractor.py
S1: Analyzes the user's raw query to extract intent, domain, and missing constraints.
Uses Gemini as the backbone LLM for structured intent analysis.
"""
import json
import os
import os
from langchain_core.messages import HumanMessage, SystemMessage
from live_mode_utils import invoke_openrouter_with_fallback, parse_json_response

# ---------------------------------------------------------------------------
# Demo-mode fixtures (used when DEMO_MODE=True in Streamlit session state)
# ---------------------------------------------------------------------------
DEMO_INTENT = {
    "intent": "Generate a professional clinical discharge summary for a diabetic patient",
    "topic_domain": "healthcare",
    "format_domain": "clinical document",
    "missing_info": [
        "Patient's age and comorbidities",
        "Medications at time of discharge",
        "Follow-up instructions",
    ],
    "constraints": [
        "Must comply with HIPAA language guidelines",
        "Should be concise yet comprehensive",
        "Avoid technical jargon where possible",
    ],
    "query_quality": "moderate",
}

SYSTEM_PROMPT = """You are an expert prompt analyst. Your job is to deeply analyse a user's raw query and extract structured intent information.

Return ONLY valid JSON with this exact schema:
{
  "intent": "<one sentence describing the core goal>",
  "topic_domain": "<domain: healthcare | education | research | legal | business | general>",
  "format_domain": "<format: JSON | blog post | email | code snippet | essay | clinical document | general>",
  "missing_info": ["<piece of info missing from query>", ...],
  "constraints": ["<implicit or explicit constraint>", ...],
  "query_quality": "<poor | moderate | good>"
}

Be specific, precise, and helpful. Identify all implicit assumptions and unstated requirements."""


def extract_intent(raw_query: str, demo_mode: bool = False) -> dict:
    """
    Extract structured intent from the user's raw query.
    Returns a dict with keys: intent, domain, missing_info, constraints, query_quality.
    """
    if demo_mode:
        return DEMO_INTENT

    from config import MODELS

    messages = [
        HumanMessage(content=f"{SYSTEM_PROMPT}\n\nAnalyse this query:\n\n{raw_query}"),
    ]

    content, _ = invoke_openrouter_with_fallback(
        messages,
        MODELS["intent_extractor"],
        allow_router=False,
        temperature=0.0,
        max_tokens=500,
    )
    return parse_json_response(content)
