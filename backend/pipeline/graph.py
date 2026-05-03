"""
pipeline/graph.py
Pipeline orchestrating all ConsensusPrompt agents.
Runs intent extraction → three parallel rewrites → council peer review → chairman synthesis.
"""
import json
import os
import datetime
import asyncio
from typing import Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from pipeline.state import ConsensusState
from agents.intent_extractor import extract_intent
from agents.rewriter_a import rewrite_chain_of_thought
from agents.rewriter_b import rewrite_role_assignment
from agents.rewriter_c import rewrite_structured_template
from agents.council import peer_review, chairman_synthesise
from live_mode_utils import invoke_openrouter_model, extract_prompt_and_perspective

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


def _validate_candidate_outputs(candidate_a: str, candidate_b: str, candidate_c: str) -> None:
    candidates = {
        "Candidate A": candidate_a,
        "Candidate B": candidate_b,
        "Candidate C": candidate_c,
    }
    missing = [name for name, text in candidates.items() if not str(text or "").strip()]
    if missing:
        raise RuntimeError(
            "Prompt generation failed because not all three rewriter roles produced usable output: "
            + ", ".join(missing)
        )


def _validate_review_outputs(peer_reviews: list[dict]) -> None:
    if len(peer_reviews) != 3:
        raise RuntimeError(
            f"Review process failed because exactly 3 reviewer outputs were expected, but received {len(peer_reviews)}."
        )

    incomplete = []
    for review in peer_reviews:
        reviewer = review.get("reviewer", "Unknown reviewer")
        evaluation = str(review.get("evaluation") or "").strip()
        ranking = review.get("parsed_ranking") or []
        unique_ranking = list(dict.fromkeys(ranking))
        if not evaluation or len(unique_ranking) != 3:
            incomplete.append(reviewer)

    if incomplete:
        raise RuntimeError(
            "Review process failed because these reviewer roles did not return exactly three distinct ranked candidates: "
            + ", ".join(incomplete)
        )


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

    def notify(stage_key: str, message: str, pct: int):
        if progress_callback:
            progress_callback({
                "stage": stage_key,
                "message": message,
                "progress": pct,
            })

    # S1: Intent Extraction
    notify("intent", "Extracting intent from query", 10)
    intent = extract_intent(raw_query, demo_mode=demo_mode)
    state["intent"] = intent
    notify("intent_complete", "Intent extraction complete", 25)

    # S2: Parallel Agent Rewriting
    notify("rewriters", "Rewriting with three parallel strategies", 30)
    # Use a fresh event loop to avoid conflicts when called from asyncio.to_thread()
    loop = asyncio.new_event_loop()
    try:
        candidate_a, candidate_b, candidate_c = loop.run_until_complete(
            run_rewriters_async(raw_query, intent, demo_mode)
        )
    finally:
        loop.close()
    _validate_candidate_outputs(candidate_a, candidate_b, candidate_c)

    DEFAULT_PERSPECTIVES = {
        "A": "Chain-of-Thought Reasoning",
        "B": "Role-Assignment & Few-Shot",
        "C": "Structured Domain Templates",
    }

    def parse_cand(cand_text: str, agent_key: str):
        return extract_prompt_and_perspective(
            cand_text,
            DEFAULT_PERSPECTIVES[agent_key],
            step=f"rewriter_{agent_key.lower()}",
        )

    prompt_a, persp_a = parse_cand(candidate_a, "A")
    prompt_b, persp_b = parse_cand(candidate_b, "B")
    prompt_c, persp_c = parse_cand(candidate_c, "C")

    state["candidate_a"] = prompt_a
    state["candidate_b"] = prompt_b
    state["candidate_c"] = prompt_c
    state["perspectives"] = {"Candidate A": persp_a, "Candidate B": persp_b, "Candidate C": persp_c}
    notify("rewriters_complete", "All rewriters finished", 60)

    # S3a + S3b: Council peer review + aggregate ranking
    notify("review", "Council: anonymised peer review in progress", 65)
    reviews, aggregate, label_map, diagnostics = peer_review(
        raw_query=raw_query,
        candidate_a=prompt_a,
        candidate_b=prompt_b,
        candidate_c=prompt_c,
        demo_mode=demo_mode,
    )
    state["peer_reviews"] = reviews
    state["aggregate_rankings"] = aggregate
    state["label_map"] = label_map
    state["consensus_diagnostics"] = diagnostics
    _validate_review_outputs(reviews)
    notify("review_complete", "Council review complete — aggregating rankings", 85)

    # S3c: Chairman synthesis
    notify("chairman", "Chairman synthesising final prompt", 88)
    optimised_prompt, chairman_info = chairman_synthesise(
        raw_query=raw_query,
        candidate_a=prompt_a,
        candidate_b=prompt_b,
        candidate_c=prompt_c,
        peer_reviews=reviews,
        aggregate=aggregate,
        label_map=label_map,
        topic_domain=intent.get("topic_domain", domain),
        demo_mode=demo_mode,
    )
    state["chairman"] = chairman_info
    state["optimised_prompt"] = optimised_prompt
    notify("complete", "Consensus reached", 100)

    # Analytics Logging
    try:
        if aggregate and len(aggregate) > 0:
            winning_label = aggregate[0].get("label", "Unknown") # 'Candidate A'
            winning_candidate = state["label_map"].get(winning_label, "Unknown")
            winning_perspective = state["perspectives"].get(winning_candidate, "Unknown")
            
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


def execute_prompt(
    final_prompt: str,
    target_model: str = "tencent/hy3-preview:free",
    demo_mode: bool = False,
) -> str:
    """
    S5: Execute the approved final prompt against the chosen target LLM.
    """
    if demo_mode:
        prompt_lower = final_prompt.lower()
        optimisation_signals = [
            "step 1",
            "template",
            "constraints",
            "follow-up",
            "grade 8",
            "hipaa",
            "you are",
            "##",
            "| medication",
        ]
        is_optimised = sum(signal in prompt_lower for signal in optimisation_signals) >= 2 or len(final_prompt) > 220

        if not is_optimised:
            return """The patient was admitted for uncontrolled diabetes with high blood glucose and symptoms of polyuria and polydipsia. He improved after treatment with insulin and monitoring in the hospital.

He is being discharged in stable condition on insulin therapy and metformin. He should follow up with endocrinology within two weeks and with primary care within one month. He was educated about checking glucose, taking insulin correctly, and returning for care if symptoms worsen."""

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

    content, _ = invoke_openrouter_model(
        [HumanMessage(content=final_prompt)],
        target_model,
        temperature=0.7,
        max_tokens=4096,
    )
    return content
