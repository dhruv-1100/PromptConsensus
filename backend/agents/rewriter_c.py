"""
agents/rewriter_c.py
S2 Agent C: Structured Templates + Domain Constraints Strategy
Rewrites the user's query using domain-specific output templates and explicit constraints.
Uses Gemini.
"""
import os
import os
from langchain_core.messages import HumanMessage, SystemMessage
from live_mode_utils import invoke_openrouter_with_fallback

DEMO_CANDIDATE_C = """{
  "optimised_prompt": "Generate a clinical discharge summary. Use this template:\\n## DISCHARGE SUMMARY\\n**Patient:** [Age/Sex]\\n**Reason for Admission:** [Narrative]\\n**Discharge Medications:** [Table with columns: Medication | Dose | Route]\\n**Constraints:** HIPAA compliant, 8th-grade reading level.",
  "perspective_used": "Rigid Formatting Constraints"
}"""

SYSTEM_PROMPT = """You are Rewriter C in a prompt council. Your role is to produce the best possible prompt using explicit structure and constraints.

You MUST optimize primarily through:
- output schemas or templates
- concrete constraints and quality checks
- domain-specific formatting requirements
- completion criteria the target model can verify while writing

You MUST NOT optimize primarily through persona assignment or step-by-step reasoning unless they are required as supporting details.

You MUST return your output ONLY as valid JSON matching this exact schema:
{
  "optimised_prompt": "<your final, ready-to-use prompt>",
  "perspective_used": "<the name and brief explanation of the optimization technique you chose>"
}
Do not include any other text, markdown fences, or preamble."""


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

    messages = [
        HumanMessage(
            content=f"{SYSTEM_PROMPT}\n\nRewrite this query dynamically using the best possible perspective:\n\nOriginal query: {raw_query}\n\nContext:\n{context}"
        ),
    ]

    content, _ = invoke_openrouter_with_fallback(
        messages,
        MODELS["rewriter_c"],
        allow_router=False,
        temperature=0.7,
        max_tokens=1000,
    )
    return content
