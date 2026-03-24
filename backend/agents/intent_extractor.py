"""
agents/intent_extractor.py
S1: Analyzes the user's raw query to extract intent, domain, and missing constraints.
Uses Gemini as the backbone LLM for structured intent analysis.
"""
import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Demo-mode fixtures (used when DEMO_MODE=True in Streamlit session state)
# ---------------------------------------------------------------------------
DEMO_INTENT = {
    "intent": "Generate a professional clinical discharge summary for a diabetic patient",
    "domain": "healthcare",
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
  "domain": "<domain: healthcare | education | research | legal | business | general>",
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

    llm = ChatGoogleGenerativeAI(
        model="gemma-3-1b-it",
        temperature=0.3,
        max_tokens=1024,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )

    messages = [
        HumanMessage(content=f"{SYSTEM_PROMPT}\n\nAnalyse this query:\n\n{raw_query}"),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    return json.loads(content)
