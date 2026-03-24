"""
pipeline/graph.py
Pipeline orchestrating all ConsensusPrompt agents.
Runs intent extraction → three parallel rewrites → council peer review → chairman synthesis.
"""
import json
import os
import datetime
from typing import Any
from dotenv import load_dotenv

from pipeline.state import ConsensusState
from agents.intent_extractor import extract_intent
from agents.rewriter_a import rewrite_chain_of_thought
from agents.rewriter_b import rewrite_role_assignment
from agents.rewriter_c import rewrite_structured_template
from agents.council import peer_review, chairman_synthesise

load_dotenv()


async def run_rewriters_async(
    raw_query: str, intent: dict, demo_mode: bool
) -> tuple[str, str, str]:
    """Run all three rewriting agents concurrently."""
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, rewrite_chain_of_thought, raw_query, intent, demo_mode),
        loop.run_in_executor(None, rewrite_role_assignment, raw_query, intent, demo_mode),
        loop.run_in_executor(None, rewrite_structured_template, raw_query, intent, demo_mode),
    ]
    results = await asyncio.gather(*tasks)
    return results[0], results[1], results[2]


def run_pipeline(
    raw_query: str,
    domain: str = "general",
    demo_mode: bool = False,
    progress_callback=None,
) -> ConsensusState:
    """
    Execute the full ConsensusPrompt pipeline (S1 → S2 → S3a/b/c).
    Returns a populated ConsensusState dict.
    """
    state: ConsensusState = {
        "raw_query": raw_query,
        "domain": domain,
    }

    def notify(stage: str, pct: int):
        if progress_callback:
            progress_callback(stage, pct)

    # S1: Intent Extraction
    notify("Extracting intent", 10)
    intent = extract_intent(raw_query, demo_mode=demo_mode)
    state["intent"] = intent
    notify("Intent extracted", 25)

    # S2: Parallel Agent Rewriting
    notify("Rewriting with three strategies", 30)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    candidate_a, candidate_b, candidate_c = loop.run_until_complete(
        run_rewriters_async(raw_query, intent, demo_mode)
    )

    # Parse JSON structured candidate outputs to isolate the prompt and perspective
    import json
    def parse_cand(cand_text: str):
        try:
            # Strip markdown fences if present
            clean = cand_text.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
                clean = clean.strip()
            data = json.loads(clean)
            return data.get("optimised_prompt", cand_text), data.get("perspective_used", "Unknown")
        except Exception:
            return cand_text, "Unknown"

    prompt_a, persp_a = parse_cand(candidate_a)
    prompt_b, persp_b = parse_cand(candidate_b)
    prompt_c, persp_c = parse_cand(candidate_c)

    state["candidate_a"] = prompt_a
    state["candidate_b"] = prompt_b
    state["candidate_c"] = prompt_c
    state["perspectives"] = {"Candidate A": persp_a, "Candidate B": persp_b, "Candidate C": persp_c}
    notify("Rewriting complete", 60)

    # S3a + S3b: Council peer review + aggregate ranking
    notify("Council peer review in progress", 65)
    reviews, aggregate, label_map = peer_review(
        raw_query=raw_query,
        candidate_a=prompt_a,
        candidate_b=prompt_b,
        candidate_c=prompt_c,
        demo_mode=demo_mode,
    )
    state["peer_reviews"] = reviews
    state["aggregate_rankings"] = aggregate
    state["label_map"] = label_map
    notify("Council ranking complete", 85)

    # S3c: Chairman synthesis
    notify("Chairman synthesising final prompt", 88)
    optimised_prompt, chairman_info = chairman_synthesise(
        raw_query=raw_query,
        candidate_a=prompt_a,
        candidate_b=prompt_b,
        candidate_c=prompt_c,
        peer_reviews=reviews,
        aggregate=aggregate,
        label_map=label_map,
        demo_mode=demo_mode,
    )
    state["chairman"] = chairman_info
    state["optimised_prompt"] = optimised_prompt
    notify("Consensus reached", 100)

    # Analytics Logging
    try:
        if aggregate and len(aggregate) > 0:
            winning_label = aggregate[0].get("label", "Unknown") # 'Candidate A'
            winning_perspective = state["perspectives"].get(winning_label, "Unknown")
            
            entry = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "topic_domain": intent.get("topic_domain", "general"),
                "format_domain": intent.get("format_domain", "general"),
                "winning_model": winning_label,
                "perspective_used": winning_perspective
            }
            log_path = os.path.join(os.path.dirname(__file__), "..", "optimisation_insights.json")
            
            logs = []
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    logs = json.load(f)
            logs.append(entry)
            with open(log_path, "w") as f:
                json.dump(logs, f, indent=4)
    except Exception as e:
        print("Failed to write optimisation insights:", e)

    return state


def execute_prompt(final_prompt: str, target_model: str = "gpt-4o", demo_mode: bool = False) -> str:
    """
    S5: Execute the approved final prompt against the chosen target LLM.
    """
    if demo_mode:
        return """## DISCHARGE SUMMARY

**Primary Diagnosis:** Type 2 Diabetes Mellitus, Uncontrolled (E11.65)
**Reason for Admission:** 58-year-old male presented with hyperglycaemia (blood glucose 387 mg/dL), polyuria, and polydipsia x 3 days. HbA1c found to be 9.4% on admission labs.

**Hospital Course:** Patient was admitted and started on an insulin drip with hourly glucose monitoring. Blood glucose normalised to 140-180 mg/dL range within 14 hours. Transitioned to basal-bolus insulin regimen (Glargine 20 units at bedtime; Lispro 6 units with meals). Certified Diabetes Educator conducted two sessions covering self-monitoring of blood glucose, insulin administration technique, carbohydrate counting, and sick-day management. Renal function monitored throughout; Metformin held during admission.

**Discharge Condition:** Stable, Improved

**Medications at Discharge:**
| Medication (Generic/Brand) | Dose | Route | Frequency | Notes |
|---|---|---|---|---|
| Insulin glargine (Lantus) | 20 units | Subcutaneous | At bedtime | High-risk |
| Insulin lispro (Humalog) | 6 units | Subcutaneous | With each meal | High-risk |
| Metformin (Glucophage) | 1000 mg | Oral | Twice daily | Resume at discharge |

**Follow-Up Instructions:**
- Endocrinology: within 10-14 days -- bring glucose log
- Primary Care: within 4 weeks
- Daily glucose targets: 80-130 mg/dL before meals; below 180 mg/dL 2 hours after meals
- **Return to Emergency Department if:** blood glucose exceeds 300 mg/dL, or patient experiences chest pain, severe abdominal pain, or fruity breath

**Patient Education Provided:** Insulin self-administration, Glucose monitoring technique, Carbohydrate counting, Sick-day management rules, Signs of hypoglycaemia/hyperglycaemia

---
*Discharge summary generated with ConsensusPrompt*"""

    load_dotenv()

    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model="gemma-3-1b-it",
        temperature=0.7,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )

    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=final_prompt)])
    return response.content
