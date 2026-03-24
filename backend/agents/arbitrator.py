"""
agents/arbitrator.py
S3: Consensus Arbitration Agent (Judge)
Evaluates the three candidate prompts and recommends the best one,
or synthesises a hybrid incorporating the strongest elements of each.
Uses Gemini as the judge LLM.
"""
import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

DEMO_ARBITRATION = {
    "recommended": "hybrid",
    "winner": None,
    "rationale": "Agent C's structured template provides the clearest output format and domain compliance requirements (HIPAA, reading level), while Agent A's step-by-step reasoning scaffolds ensure the LLM works systematically through the complex clinical content. The hybrid merges C's structural rigor with A's reasoning guidance, giving the user the best of both worlds.",
    "scores": {
        "Agent A (Chain-of-Thought)": {
            "clarity": 8,
            "completeness": 7,
            "faithfulness": 9,
            "domain_fit": 8,
            "total": 32,
        },
        "Agent B (Role-Assignment)": {
            "clarity": 7,
            "completeness": 8,
            "faithfulness": 7,
            "domain_fit": 9,
            "total": 31,
        },
        "Agent C (Structured Template)": {
            "clarity": 9,
            "completeness": 9,
            "faithfulness": 8,
            "domain_fit": 9,
            "total": 35,
        },
    },
    "hybrid_used": True,
}

DEMO_OPTIMISED = """You are a board-certified internist completing a clinical discharge summary. Work through the following steps systematically, then populate the required template.

**STEP 1:** Identify the primary diagnosis, admission trigger, and key comorbidities.
**STEP 2:** Chronologically trace the hospital course — interventions, medication changes, glucose trends.
**STEP 3:** Confirm all discharge medications and reconcile with admission medications.
**STEP 4:** Formulate specific, measurable follow-up instructions.

**Then generate the discharge summary using this exact format:**

## DISCHARGE SUMMARY

**Primary Diagnosis:** [ICD-10 + description]
**Reason for Admission:** [2–3 sentences]
**Hospital Course:** [Narrative with glucose trends, medication adjustments, patient education]
**Discharge Condition:** Stable / Improved / Guarded

**Medications at Discharge:**
| Medication (Generic/Brand) | Dose | Frequency | Notes |
|----------------------------|------|-----------|-------|
| [Entry] | | | ⚠️ if high-risk |

**Follow-Up:**
- Endocrinology: within 2 weeks
- Primary Care: within 4 weeks
- Daily glucose targets: 80–130 mg/dL fasting

**Constraints:** HIPAA compliant language; patient instructions at Grade 8 reading level; flag high-risk medications."""

SYSTEM_PROMPT = """You are an expert judge evaluating three AI-generated prompt rewrites. Your goal is to select the best prompt or synthesise a superior hybrid.

Evaluate each candidate on four dimensions (score 1–10 each):
1. **Clarity**: Is the prompt clear and unambiguous to an LLM?
2. **Completeness**: Does it capture all the user's intent and constraints?
3. **Faithfulness**: Does it preserve the original query's core goal?
4. **Domain fit**: Is it appropriately calibrated for the stated domain?

Return ONLY valid JSON with this schema:
{
  "recommended": "hybrid" | "A" | "B" | "C",
  "winner": null | "A" | "B" | "C",
  "rationale": "<plain-language explanation of your choice, written for a non-technical user>",
  "scores": {
    "Agent A (Chain-of-Thought)": {"clarity": N, "completeness": N, "faithfulness": N, "domain_fit": N, "total": N},
    "Agent B (Role-Assignment)": {"clarity": N, "completeness": N, "faithfulness": N, "domain_fit": N, "total": N},
    "Agent C (Structured Template)": {"clarity": N, "completeness": N, "faithfulness": N, "domain_fit": N, "total": N}
  },
  "hybrid_used": true | false
}"""

HYBRID_PROMPT = """You are a world-class prompt engineer. Based on the evaluation below, synthesise the single best prompt by combining the strongest elements from all three candidates.

Your synthesised prompt should be immediately usable as a final prompt for an LLM.
Return ONLY the synthesised prompt, with no preamble or explanation."""


def arbitrate(
    raw_query: str,
    candidate_a: str,
    candidate_b: str,
    candidate_c: str,
    intent: dict,
    demo_mode: bool = False,
) -> tuple[dict, str]:
    """
    Judge agent that evaluates three candidates and returns:
    - arbitration: dict with evaluation scores and rationale
    - optimised_prompt: the recommended or synthesised prompt string
    """
    if demo_mode:
        return DEMO_ARBITRATION, DEMO_OPTIMISED

    llm = ChatGoogleGenerativeAI(
        model="gemma-3-1b-it",
        temperature=0.2,
        max_tokens=2048,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )

    # Step 1: Evaluate all three candidates
    eval_messages = [
        HumanMessage(
            content=f"""{SYSTEM_PROMPT}\n\nOriginal query: {raw_query}
Domain: {intent.get('domain', 'general')}

CANDIDATE A (Chain-of-Thought):
{candidate_a}

CANDIDATE B (Role-Assignment):
{candidate_b}

CANDIDATE C (Structured Template):
{candidate_c}

Please evaluate all three candidates and provide your verdict."""
        ),
    ]

    eval_response = llm.invoke(eval_messages)
    content = eval_response.content.strip()

    # Strip markdown code fences
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    arbitration = json.loads(content)

    # Step 2: If hybrid recommended, synthesise the best prompt
    if arbitration.get("recommended") == "hybrid":
        synth_messages = [
            HumanMessage(
                content=f"""{HYBRID_PROMPT}\n\nEvaluation result: {json.dumps(arbitration, indent=2)}

Original query: {raw_query}

CANDIDATE A: {candidate_a}

CANDIDATE B: {candidate_b}

CANDIDATE C: {candidate_c}

Synthesise the optimal prompt now."""
            ),
        ]
        synth_response = llm.invoke(synth_messages)
        optimised_prompt = synth_response.content.strip()
    else:
        winner = arbitration.get("recommended", "C")
        mapping = {"A": candidate_a, "B": candidate_b, "C": candidate_c}
        optimised_prompt = mapping.get(winner, candidate_c)

    return arbitration, optimised_prompt
