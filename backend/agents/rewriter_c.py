"""
agents/rewriter_c.py
S2 Agent C: Structured Templates + Domain Constraints Strategy
Rewrites the user's query using domain-specific output templates and explicit constraints.
Uses Gemini.
"""
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

DEMO_CANDIDATE_C = """Generate a clinical discharge summary for a diabetic patient using the following structured format. Adhere strictly to all constraints listed below.

**OUTPUT FORMAT (use these exact section headers):**

## DISCHARGE SUMMARY

**Patient Information:** [Age, Sex — no identifying information]
**Admission Date:** [Date] | **Discharge Date:** [Date]
**Primary Diagnosis:** [ICD-10 code + plain language description]
**Attending Physician:** [Name, Credentials]

**Reason for Admission:**
[2–3 sentences describing presenting symptoms and circumstances]

**Hospital Course:**
[Chronological narrative of treatment, interventions, and patient response. Include: blood glucose trends, medication adjustments, consultations ordered, patient education sessions conducted]

**Discharge Condition:** ☐ Stable ☐ Improved ☐ Guarded

**Discharge Medications:**
| Medication | Dose | Route | Frequency | Duration |
|------------|------|-------|-----------|----------|
| [Name] | [Dose] | [PO/SC/IV] | [Freq] | [Until follow-up] |

**Follow-Up Instructions:**
- Appointment 1: Endocrinology — within 2 weeks
- Appointment 2: Primary Care — within 4 weeks
- Daily monitoring: Blood glucose targets [80–130 mg/dL fasting; <180 mg/dL postprandial]
- Return to ED if: blood glucose >300 mg/dL, signs of DKA, or chest pain

**Patient Education Provided:** ☐ Insulin self-administration ☐ Glucose monitoring ☐ Dietary guidance ☐ Sick-day rules

**CONSTRAINTS:**
- Use clinical but accessible language (Grade 8 reading level for patient instructions)
- All medications must include generic name, brand name in parentheses
- Must flag any high-risk medications with ⚠️
- Do not include any patient identifying information (HIPAA compliance)
- Discharge instructions section must be comprehensible to a non-medical patient"""

SYSTEM_PROMPT = """You are an expert prompt engineer specialising in structured template and domain constraint strategies.
Your task is to rewrite user queries by:
- Providing a detailed output template with exact section headers and formatting specifications
- Adding explicit domain-specific constraints the LLM must adhere to
- Specifying output format (tables, bullet points, specific notation)
- Including quality requirements and compliance requirements relevant to the domain
- Making every requirement explicit so the LLM has zero ambiguity

Return ONLY the rewritten prompt including the template and constraints, no preamble."""


def rewrite_structured_template(raw_query: str, intent: dict, demo_mode: bool = False) -> str:
    """
    Agent C: Rewrite using structured templates + domain constraints strategy.
    Returns the rewritten prompt string.
    """
    if demo_mode:
        return DEMO_CANDIDATE_C

    llm = ChatGoogleGenerativeAI(
        model="gemma-3-1b-it",
        temperature=0.5,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )

    context = f"""Domain: {intent.get('domain', 'general')}
Core intent: {intent.get('intent', '')}
Missing information to address: {', '.join(intent.get('missing_info', []))}
Constraints to satisfy: {', '.join(intent.get('constraints', []))}"""

    messages = [
        HumanMessage(
            content=f"{SYSTEM_PROMPT}\n\nRewrite this query using structured templates and domain constraints:\n\nOriginal query: {raw_query}\n\nContext:\n{context}"
        ),
    ]

    response = llm.invoke(messages)
    return response.content.strip()
