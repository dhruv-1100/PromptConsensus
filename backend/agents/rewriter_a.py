"""
agents/rewriter_a.py
S2 Agent A: Chain-of-Thought Reasoning Strategy
Rewrites the user's query by inserting explicit step-by-step reasoning scaffolds.
Uses Gemini.
"""
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

DEMO_CANDIDATE_A = """Let's approach this systematically. First, establish the patient context by gathering all relevant medical history and current admission details. Then, step-by-step:

1. Summarise the reason for admission and primary diagnosis (Type 2 Diabetes Mellitus with HbA1c of [X]%)
2. Document all interventions performed during the stay (insulin titration, dietary counselling, foot exam)
3. Record the patient's response to treatment and any complications encountered
4. List all discharge medications with dosage, frequency, and administration instructions
5. Provide specific follow-up instructions: endocrinology appointment within 2 weeks, daily glucose monitoring targets (80–130 mg/dL fasting), and emergency return criteria

Now generate a professional clinical discharge summary following these steps for a diabetic patient, ensuring each section is complete before proceeding to the next."""

SYSTEM_PROMPT = """You are an expert prompt engineer specialising in chain-of-thought reasoning strategies. 
Your task is to rewrite user queries to elicit better, more structured LLM responses by:
- Adding explicit step-by-step reasoning scaffolds
- Breaking complex tasks into numbered sequential steps
- Instructing the LLM to "think through" each component before generating output
- Making implicit reasoning explicit

The rewritten prompt should make the LLM's reasoning process transparent and structured.
Return ONLY the rewritten prompt, with no preamble or explanation."""


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

    context = f"""Domain: {intent.get('domain', 'general')}
Core intent: {intent.get('intent', '')}
Missing information to address: {', '.join(intent.get('missing_info', []))}
Constraints to satisfy: {', '.join(intent.get('constraints', []))}"""

    messages = [
        HumanMessage(
            content=f"{SYSTEM_PROMPT}\n\nRewrite this query using chain-of-thought reasoning:\n\nOriginal query: {raw_query}\n\nContext:\n{context}"
        ),
    ]

    response = llm.invoke(messages)
    return response.content.strip()
