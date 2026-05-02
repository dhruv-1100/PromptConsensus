"""
agents/rewriter_c.py
S2 Agent C: Structured Templates + Domain Constraints Strategy
Rewrites the user's query using domain-specific output templates and explicit constraints.
Uses Gemini.
"""
import os
import os
from langchain_core.messages import HumanMessage
from live_mode_utils import invoke_openrouter_model

DEMO_CANDIDATE_C = """{
  "optimised_prompt": "Generate a clinical discharge summary. Use this template:\\n## DISCHARGE SUMMARY\\n**Patient:** [Age/Sex]\\n**Reason for Admission:** [Narrative]\\n**Discharge Medications:** [Table with columns: Medication | Dose | Route]\\n**Constraints:** HIPAA compliant, 8th-grade reading level.",
  "perspective_used": "Rigid Formatting Constraints"
}"""

SYSTEM_PROMPT = """You are Rewriter C in a prompt council.

Your job is to rewrite the user's request into a stronger prompt using explicit structure and constraints.

Priorities:
- Define a clear output shape.
- Add constraints that improve reliability, completeness, and verifiability.
- Use domain-appropriate sections, checklists, schemas, or templates when helpful.
- Keep the prompt strong without making it brittle or artificially compressed.

Rules:
- Do not optimize primarily through persona-play or step-by-step reasoning.
- Do not turn the prompt into a request for more user-supplied materials such as PDFs, URLs, uploads, DOIs, or external documents.
- Do not impose strict word limits, sentence caps, paragraph caps, or bullet-count caps unless the user explicitly asked for them.
- If the task is summarization, prefer a structured output with headings, sections, or bullets instead of a single paragraph.
- Constraints should improve quality and traceability, not strip out nuance.

Return ONLY valid JSON matching this schema:
{
  "optimised_prompt": "<final ready-to-use prompt>",
  "perspective_used": "<short name and explanation of the structural strategy>"
}"""


def _domain_specific_guidance(intent: dict) -> str:
    topic_domain = str(intent.get("topic_domain", "general")).strip().lower()
    if topic_domain == "research":
        return (
            "Research-domain guardrails:\n"
            "- Use section-based analytical structure rather than a single paragraph.\n"
            "- Prefer sections such as summary, key findings, methodology, limitations, evidence, and uncertainty when relevant.\n"
            "- Do not impose strict word limits, sentence caps, or bullet-count caps unless the user explicitly requested them.\n"
            "- Do not ask the user for a PDF, URL, DOI, upload, or other extra source material.\n"
            "- Constraints should improve evidentiary quality and traceability, not reduce depth or nuance."
        )
    return ""


def rewrite_structured_template(raw_query: str, intent: dict, demo_mode: bool = False) -> str:
    """
    Agent C: Rewrite using structured templates + domain constraints strategy.
    Returns the rewritten prompt string.
    """
    if demo_mode:
        return DEMO_CANDIDATE_C

    from config import MODELS

    context = f"""Topic Domain: {intent.get('topic_domain', 'general')}
Format Domain: {intent.get('format_domain', 'general')}
Core intent: {intent.get('intent', '')}
Missing information to address: {', '.join(intent.get('missing_info', []))}
Constraints to satisfy: {', '.join(intent.get('constraints', []))}"""
    domain_guidance = _domain_specific_guidance(intent)

    messages = [
        HumanMessage(
            content=(
                f"{SYSTEM_PROMPT}\n\n"
                f"{domain_guidance}\n\n" if domain_guidance else f"{SYSTEM_PROMPT}\n\n"
            ) + (
                f"Rewrite this query dynamically using the best possible perspective:\n\n"
                f"Original query: {raw_query}\n\n"
                f"Context:\n{context}"
            )
        ),
    ]

    content, _ = invoke_openrouter_model(
        messages,
        MODELS["rewriter_c"],
        temperature=0.7,
        max_tokens=1000,
    )
    return content
