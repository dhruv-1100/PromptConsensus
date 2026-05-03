"""
agents/rewriter_b.py
S2 Agent B: Role-Assignment + Few-Shot Formatting Strategy
Rewrites the user's query by assigning an expert persona and providing few-shot examples.
Uses Gemini.
"""
import os
import os
from langchain_core.messages import HumanMessage
from live_mode_utils import invoke_openrouter_model

DEMO_CANDIDATE_B = """{
  "optimised_prompt": "You are Dr. Sarah Chen, a board-certified endocrinologist. Here is an example discharge summary:\\n\\nPatient: [Name], 58 y/o male\\nDiagnosis: Type 2 Diabetes Mellitus, uncontrolled\\nHospital Course: Insulin drip initiated, transitioned to basal-bolus.\\nDischarge Medications: Glargine 20 units\\nFollow-up: Endocrinology in 10-14 days\\n\\nNow, using this expert voice, please write a comprehensive clinical discharge summary for a diabetic patient.",
  "perspective_used": "Few-Shot Persona Contextualization"
}"""

SYSTEM_PROMPT = """You are Rewriter B in a prompt council.

Your job is to rewrite the user's request into a stronger prompt using contextual framing.

Priorities:
- Choose an appropriate expert stance, audience framing, or operating perspective.
- Improve domain grounding and tone.
- Use examples only when they materially improve quality and do not overconstrain the task.
- Keep the prompt practical and immediately usable.

Rules:
- Do not optimize primarily through step-by-step reasoning scaffolds or rigid schemas.
- Do not turn the prompt into a request for more user-supplied materials such as PDFs, URLs, uploads, DOIs, or external documents.
- Do not introduce strict word limits, paragraph caps, or compressed-summary instructions unless the user explicitly asked for them.
- If the task is summarization, prefer a structured answer format with sections, bullets, or headings over a single paragraph.
- Use few-shot framing carefully; do not use examples that make the output artificially brief or shallow by default.

Return ONLY valid JSON matching this schema:
{
  "optimised_prompt": "<final ready-to-use prompt>",
  "perspective_used": "<short name and explanation of the contextual strategy>"
}"""


def _domain_specific_guidance(intent: dict) -> str:
    topic_domain = str(intent.get("topic_domain", "general")).strip().lower()
    if topic_domain == "research":
        return (
            "Research-domain guardrails:\n"
            "- Do not add one-paragraph requirements or tight word-count targets unless the user explicitly asked for them.\n"
            "- Do not ask the user to provide a paper PDF, URL, DOI, upload, or other additional source material.\n"
            "- Prefer an analytical research-review framing over a compressed abstract-style summary.\n"
            "- Prefer clearly labeled sections for findings, methods, limitations, evidence, and uncertainty when useful.\n"
            "- Avoid few-shot examples that push the answer toward a brief overview by default."
        )
    return ""


def rewrite_role_assignment(raw_query: str, intent: dict, demo_mode: bool = False) -> str:
    """
    Agent B: Rewrite using role-assignment + few-shot formatting strategy.
    Returns the rewritten prompt string.
    """
    if demo_mode:
        return DEMO_CANDIDATE_B

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

    content, actual_model = invoke_openrouter_model(
        messages,
        MODELS["rewriter_b"],
        temperature=0.7,
        max_tokens=4096,
    )
    return content
