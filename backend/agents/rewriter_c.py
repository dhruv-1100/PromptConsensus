"""
agents/rewriter_c.py
S2 Agent C: Structured Templates + Domain Constraints Strategy
Rewrites the user's query using domain-specific output templates and explicit constraints.
Uses Gemini.
"""
import os
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

DEMO_CANDIDATE_C = """{
  "optimised_prompt": "Generate a clinical discharge summary. Use this template:\\n## DISCHARGE SUMMARY\\n**Patient:** [Age/Sex]\\n**Reason for Admission:** [Narrative]\\n**Discharge Medications:** [Table with columns: Medication | Dose | Route]\\n**Constraints:** HIPAA compliant, 8th-grade reading level.",
  "perspective_used": "Rigid Formatting Constraints"
}"""

SYSTEM_PROMPT = """You are an expert Prompt Engineer. Your job is to improve the user's raw query into a highly effective, robust, and detailed prompt.
Analyze the user's core intent, topic domain, and format domain, and dynamically choose the ABSOLUTE BEST prompt optimization technique (e.g., Chain-of-Thought, Few-Shot, Role-Assignment, Structured Templates, Meta-Prompting). 

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
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=MODELS["rewriter_c"],
        temperature=0.7,
        max_tokens=1000,
    )

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

    response = llm.invoke(messages)
    return response.content.strip()
