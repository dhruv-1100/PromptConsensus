"""
agents/rewriter_a.py
S2 Agent A: Chain-of-Thought Reasoning Strategy
Rewrites the user's query by inserting explicit step-by-step reasoning scaffolds.
Uses Gemini.
"""
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

DEMO_CANDIDATE_A = """{
  "optimised_prompt": "Let's approach this systematically. First, establish the patient context by gathering all relevant medical history and current admission details. Then, step-by-step:\\n\\n1. Summarise the reason for admission and primary diagnosis (Type 2 Diabetes Mellitus with HbA1c of [X]%)\\n2. Document all interventions performed during the stay\\n3. Record the patient's response to treatment\\n4. List all discharge medications with dosage\\n5. Provide follow-up instructions\\n\\nNow generate a professional clinical discharge summary following these steps for a diabetic patient, ensuring each section is complete before proceeding.",
  "perspective_used": "Chain-of-Thought Heuristics"
}"""

SYSTEM_PROMPT = """You are an expert Prompt Engineer. Your job is to improve the user's raw query into a highly effective, robust, and detailed prompt.
Analyze the user's core intent, topic domain, and format domain, and dynamically choose the ABSOLUTE BEST prompt optimization technique (e.g., Chain-of-Thought, Few-Shot, Role-Assignment, Structured Templates, Meta-Prompting). 

You MUST return your output ONLY as valid JSON matching this exact schema:
{
  "optimised_prompt": "<your final, ready-to-use prompt>",
  "perspective_used": "<the name and brief explanation of the optimization technique you chose>"
}
Do not include any other text, markdown fences, or preamble."""


def rewrite_chain_of_thought(raw_query: str, intent: dict, demo_mode: bool = False) -> str:
    """
    Agent A: Rewrite using chain-of-thought reasoning strategy.
    Returns the rewritten prompt string.
    """
    if demo_mode:
        return DEMO_CANDIDATE_A

    llm = ChatGoogleGenerativeAI(
        model="gemma-3-1b-it",
        temperature=0.7,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
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
