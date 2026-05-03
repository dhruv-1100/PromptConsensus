"""
agents/rewriter_a.py
S2 Agent A: Chain-of-Thought Reasoning Strategy
Rewrites the user's query by inserting explicit step-by-step reasoning scaffolds.
Uses Gemini.
"""
import os
import os
from langchain_core.messages import HumanMessage
from live_mode_utils import invoke_openrouter_model

DEMO_CANDIDATE_A = """{
  "optimised_prompt": "Let's approach this systematically. First, establish the patient context by gathering all relevant medical history and current admission details. Then, step-by-step:\\n\\n1. Summarise the reason for admission and primary diagnosis (Type 2 Diabetes Mellitus with HbA1c of [X]%)\\n2. Document all interventions performed during the stay\\n3. Record the patient's response to treatment\\n4. List all discharge medications with dosage\\n5. Provide follow-up instructions\\n\\nNow generate a professional clinical discharge summary following these steps for a diabetic patient, ensuring each section is complete before proceeding.",
  "perspective_used": "Chain-of-Thought Heuristics"
}"""

SYSTEM_PROMPT = """You are Rewriter A in a prompt council.

Your job is to rewrite the user's request into a stronger prompt using reasoning scaffolds.

Priorities:
- Decompose the task into clear stages or steps.
- Add verification checkpoints where they improve quality.
- Clarify how to handle ambiguity or incomplete context without derailing the task.
- Keep the prompt directly usable by a downstream model.

Rules:
- Do not optimize primarily through persona-play or rigid templates.
- Do not turn the prompt into a request for more user-supplied materials such as PDFs, URLs, uploads, DOIs, or external documents.
- Missing information may be acknowledged inside the prompt as assumptions, placeholders, uncertainty labels, or instructions to proceed carefully.
- Do not introduce strict word limits, bullet-count caps, or single-paragraph output unless the user explicitly asked for them.
- If the task is summarization, prefer structured output with labeled sections, bullets, or headings over a single paragraph.

Return ONLY valid JSON matching this schema:
{
  "optimised_prompt": "<final ready-to-use prompt>",
  "perspective_used": "<short name and explanation of the reasoning strategy>"
}"""


def _domain_specific_guidance(intent: dict) -> str:
    topic_domain = str(intent.get("topic_domain", "general")).strip().lower()
    if topic_domain == "research":
        return (
            "Research-domain guardrails:\n"
            "- Preserve analytical decomposition, but keep the final answer structured and substantive.\n"
            "- Prefer sections such as key findings, methodology, limitations, evidence, and uncertainty when relevant.\n"
            "- Do not force a brief overview, a single paragraph, or a tight word budget unless the user explicitly requested that.\n"
            "- Do not ask the user for a PDF, URL, DOI, upload, or other extra source material.\n"
            "- If evidence is incomplete, instruct the model to qualify claims rather than ask the user for more materials."
        )
    return ""


def rewrite_chain_of_thought(raw_query: str, intent: dict, demo_mode: bool = False) -> str:
    """
    Agent A: Rewrite using chain-of-thought reasoning strategy.
    Returns the rewritten prompt string.
    """
    if demo_mode:
        return DEMO_CANDIDATE_A

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

    content, model_name = invoke_openrouter_model(
        messages,
        MODELS["rewriter_a"],
        temperature=0.7,
        max_tokens=4096,
    )
    return content
