"""
agents/rewriter_b.py
S2 Agent B: Role-Assignment + Few-Shot Formatting Strategy
Rewrites the user's query by assigning an expert persona and providing few-shot examples.
Uses Gemini.
"""
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

DEMO_CANDIDATE_B = """You are Dr. Sarah Chen, a board-certified endocrinologist with 15 years of experience at a major academic medical centre. You specialise in diabetes management and have written hundreds of discharge summaries for diabetic patients that are recognised for their clarity and completeness.

Here is an example of an excellent discharge summary section you might write:

---
EXAMPLE:
Patient: [Name], 58 y/o male
Diagnosis: Type 2 Diabetes Mellitus, uncontrolled (HbA1c 9.4%)
Hospital Course: Patient admitted for hyperglycaemic crisis. Insulin drip initiated, transitioned to basal-bolus regimen. Diabetes education provided by certified diabetes educator.
Discharge Medications: Glargine 20 units at bedtime; Lispro 6 units with meals; Metformin 1000mg BID (held during admission, resumed at discharge)
Follow-up: Endocrinology in 10–14 days; Primary care in 4 weeks; Daily glucose log to bring to all appointments
---

Now, using this same expert voice, level of detail, and structure, please write a comprehensive clinical discharge summary for a diabetic patient. Include all standard sections: reason for admission, hospital course, discharge condition, medications, follow-up plan, and patient education provided."""

SYSTEM_PROMPT = """You are an expert prompt engineer specialising in role-assignment and few-shot formatting strategies.
Your task is to rewrite user queries to elicit better LLM responses by:
- Assigning the LLM a specific, credentialed expert persona relevant to the domain
- Including 1-2 high-quality few-shot examples that demonstrate the desired output format
- Making the expected output structure crystal clear through example
- Calibrating the LLM's "voice" and expertise level for the domain

Return ONLY the rewritten prompt with persona assignment and examples included, no preamble."""


def rewrite_role_assignment(raw_query: str, intent: dict, demo_mode: bool = False) -> str:
    """
    Agent B: Rewrite using role-assignment + few-shot formatting strategy.
    Returns the rewritten prompt string.
    """
    if demo_mode:
        return DEMO_CANDIDATE_B

    llm = ChatGoogleGenerativeAI(
        model="gemma-3-1b-it",
        temperature=0.7,
        max_tokens=2048,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )

    context = f"""Domain: {intent.get('domain', 'general')}
Core intent: {intent.get('intent', '')}
Missing information to address: {', '.join(intent.get('missing_info', []))}
Constraints to satisfy: {', '.join(intent.get('constraints', []))}"""

    messages = [
        HumanMessage(
            content=f"{SYSTEM_PROMPT}\n\nRewrite this query using role-assignment and few-shot formatting:\n\nOriginal query: {raw_query}\n\nContext:\n{context}"
        ),
    ]

    response = llm.invoke(messages)
    return response.content.strip()
