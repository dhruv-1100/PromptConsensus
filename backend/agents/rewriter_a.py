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

SYSTEM_PROMPT = """You are Rewriter A in a prompt council. Your role is to produce the best possible prompt using a reasoning-scaffold strategy.

You MUST optimize primarily through:
- explicit decomposition
- ordered steps
- reasoning checkpoints
- handling ambiguity or missing information

You MUST NOT optimize primarily through persona assignment or rigid templates unless they are required as supporting details.

You MUST NOT turn the prompt into a request for more user-supplied materials such as PDFs, URLs, DOIs, uploads, or external documents.

When the user asks to summarize, prefer a structured output with clearly labeled sections, bullets, or headings rather than collapsing the answer into a single paragraph.

You MUST return your output ONLY as valid JSON matching this exact schema:
{
  "optimised_prompt": "<your final, ready-to-use prompt>",
  "perspective_used": "<the name and brief explanation of the optimization technique you chose>"
}
Do not include any other text, markdown fences, or preamble."""


def _domain_specific_guidance(intent: dict) -> str:
    topic_domain = str(intent.get("topic_domain", "general")).strip().lower()
    if topic_domain == "research":
        return (
            "Research-domain guardrails:\n"
            "- Do not force the final output into a single paragraph.\n"
            "- Do not impose strict word limits, bullet counts, or brevity targets unless the user explicitly requested them.\n"
            "- Do not instruct the assistant to ask the user for a PDF, URL, DOI, upload, or other additional source material.\n"
            "- Preserve decomposition for analysis, but allow the final answer to use multiple sections, bullets, and sufficient detail.\n"
            "- Prefer evidence-grounded structure over compressed summary phrasing."
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
        max_tokens=1000,
    )
    return content
