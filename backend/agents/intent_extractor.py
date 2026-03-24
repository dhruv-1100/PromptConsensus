"""
agents/intent_extractor.py
S1: Analyzes the user's raw query to extract intent, domain, and missing constraints.
Uses Gemini as the backbone LLM for structured intent analysis.
"""
import json
import os
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

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
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=MODELS["intent_extractor"],
        temperature=0.0,
        max_tokens=500,
    )

    messages = [
        HumanMessage(content=f"{SYSTEM_PROMPT}\n\nAnalyse this query:\n\n{raw_query}"),
    ]

    import time as _time
    import random as _random
    for _attempt in range(5):
        try:
            response = llm.invoke(messages)
            break
        except Exception as _e:
            if '429' in str(_e) and _attempt < 4:
                _wait = 15 + _random.uniform(1, 5) * _attempt
                print(f"[Intent] 429 limit, waiting {round(_wait, 1)}s...")
                _time.sleep(_wait)
            else:
                raise
    content = response.content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    return json.loads(content)
