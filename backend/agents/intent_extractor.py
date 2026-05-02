"""
agents/intent_extractor.py
S1: Analyzes the user's raw query to extract intent, domain, and missing constraints.
Uses Gemini as the backbone LLM for structured intent analysis.
"""
import json
import os
import os
from langchain_core.messages import HumanMessage
from live_mode_utils import invoke_openrouter_model, parse_json_response, log_structured_parse_failure

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

SYSTEM_PROMPT = """You are an intent extraction agent for a prompt-optimization pipeline.

Your task is to analyse the user's request and extract the underlying objective, domain, likely output shape, missing information, and constraints.

Rules:
- Infer the user's likely goal and preferred output format from the request itself.
- Capture missing information as analysis only. Do not turn missing information into instructions to ask the user for PDFs, URLs, uploads, DOIs, or other additional materials.
- Preserve user intent without adding unnecessary brevity requirements such as single-paragraph output, strict word limits, or compressed-summary constraints unless the user explicitly asked for them.
- If the user asks to summarize, assume a structured summary may be appropriate unless the user explicitly requested a single paragraph.
- Be concrete and concise.

Return ONLY valid JSON with this exact schema:
{
  "intent": "<one sentence describing the core goal>",
  "topic_domain": "<domain: healthcare | education | research | legal | business | general>",
  "format_domain": "<format: JSON | blog post | email | code snippet | essay | clinical document | report | summary | general>",
  "missing_info": ["<piece of info missing from query>", ...],
  "constraints": ["<implicit or explicit constraint>", ...],
  "query_quality": "<poor | moderate | good>"
}"""


def extract_intent(raw_query: str, demo_mode: bool = False) -> dict:
    """
    Extract structured intent from the user's raw query.
    Returns a dict with keys: intent, domain, missing_info, constraints, query_quality.
    """
    if demo_mode:
        return DEMO_INTENT

    from config import MODELS

    messages = [
        HumanMessage(
            content=(
                f"{SYSTEM_PROMPT}\n\n"
                f"Analyse this query and return only the JSON object:\n\n{raw_query}"
            )
        ),
    ]

    content, model_name = invoke_openrouter_model(
        messages,
        MODELS["intent_extractor"],
        temperature=0.0,
        max_tokens=500,
    )
    try:
        return parse_json_response(content)
    except Exception as exc:
        log_structured_parse_failure("intent_extractor", model_name, content, str(exc))
        raise RuntimeError(
            f"Intent extraction returned invalid structured output from {model_name}. "
            "Check backend/structured_parse_failures.json for the raw response."
        )
