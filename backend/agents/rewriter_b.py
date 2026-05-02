"""
agents/rewriter_b.py
S2 Agent B: Role-Assignment + Few-Shot Formatting Strategy
Rewrites the user's query by assigning an expert persona and providing few-shot examples.
Uses Gemini.
"""
import os
import os
from langchain_core.messages import HumanMessage, SystemMessage
from live_mode_utils import invoke_openrouter_with_fallback

DEMO_CANDIDATE_B = """{
  "optimised_prompt": "You are Dr. Sarah Chen, a board-certified endocrinologist. Here is an example discharge summary:\\n\\nPatient: [Name], 58 y/o male\\nDiagnosis: Type 2 Diabetes Mellitus, uncontrolled\\nHospital Course: Insulin drip initiated, transitioned to basal-bolus.\\nDischarge Medications: Glargine 20 units\\nFollow-up: Endocrinology in 10-14 days\\n\\nNow, using this expert voice, please write a comprehensive clinical discharge summary for a diabetic patient.",
  "perspective_used": "Few-Shot Persona Contextualization"
}"""

SYSTEM_PROMPT = """You are Rewriter B in a prompt council. Your role is to produce the best possible prompt using contextual framing.

You MUST optimize primarily through:
- expert or audience role framing
- few-shot or contrastive examples when useful
- tone and domain grounding
- clarifying the expected perspective of the model

You MUST NOT optimize primarily through numbered reasoning scaffolds or rigid templates unless they are required as supporting details.

You MUST return your output ONLY as valid JSON matching this exact schema:
{
  "optimised_prompt": "<your final, ready-to-use prompt>",
  "perspective_used": "<the name and brief explanation of the optimization technique you chose>"
}
Do not include any other text, markdown fences, or preamble."""


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

    messages = [
        HumanMessage(
            content=f"{SYSTEM_PROMPT}\n\nRewrite this query dynamically using the best possible perspective:\n\nOriginal query: {raw_query}\n\nContext:\n{context}"
        ),
    ]

    content, _ = invoke_openrouter_with_fallback(
        messages,
        MODELS["rewriter_b"],
        allow_router=False,
        temperature=0.7,
        max_tokens=1000,
    )
    return content
