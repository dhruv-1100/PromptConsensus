"""
pipeline/graph.py
Pipeline orchestrating all ConsensusPrompt agents.
Runs intent extraction → parallel rewrites → council peer review → chairman synthesis.
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
from agents.council import peer_review_candidates, chairman_synthesise_candidates
from live_mode_utils import invoke_openrouter_model, extract_prompt_and_perspective

load_dotenv()


DEFAULT_REWRITER_SPECS = [
    {
        "candidate_name": "Candidate A",
        "agent_key": "A",
        "perspective": "Chain-of-Thought Reasoning",
        "runner": rewrite_chain_of_thought,
    },
    {
        "candidate_name": "Candidate B",
        "agent_key": "B",
        "perspective": "Role-Assignment & Few-Shot",
        "runner": rewrite_role_assignment,
    },
    {
        "candidate_name": "Candidate C",
        "agent_key": "C",
        "perspective": "Structured Domain Templates",
        "runner": rewrite_structured_template,
    },
]


def _index_to_label(index: int) -> str:
    """Convert a zero-based index to spreadsheet-style labels: A, B, ..., Z, AA, AB, ..."""
    if index < 0:
        raise ValueError("Index must be non-negative.")

    label = ""
    value = index
    while True:
        value, remainder = divmod(value, 26)
        label = chr(ord("A") + remainder) + label
        if value == 0:
            break
        value -= 1
    return label


def build_rewriter_specs_for_models(model_names: list[str]) -> list[dict[str, Any]]:
    """Duplicate the A/B/C prompt families as needed for longer model lists."""
    if not model_names:
        raise ValueError("At least one rewriter model must be provided.")

    specs: list[dict[str, Any]] = []
    base_count = len(DEFAULT_REWRITER_SPECS)
    for idx, model_name in enumerate(model_names):
        base_spec = dict(DEFAULT_REWRITER_SPECS[idx % base_count])
        label = _index_to_label(idx)
        base_spec["candidate_name"] = f"Candidate {label}"
        base_spec["agent_key"] = label
        base_spec["model_name"] = model_name
        base_spec["prompt_family"] = DEFAULT_REWRITER_SPECS[idx % base_count]["candidate_name"]
        specs.append(base_spec)
    return specs


async def run_rewriters_async(
    raw_query: str,
    intent: dict,
    demo_mode: bool,
    rewriter_specs: list[dict[str, Any]],
) -> list[str]:
    """Run configured rewriting agents concurrently."""
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(
            None,
            spec["runner"],
            raw_query,
            intent,
            demo_mode,
            spec.get("model_name"),
        )
        for spec in rewriter_specs
    ]
    return list(await asyncio.gather(*tasks))


def _validate_candidate_outputs(candidates: dict[str, str]) -> None:
    missing = [name for name, text in candidates.items() if not str(text or "").strip()]
    if missing:
        raise RuntimeError(
            "Prompt generation failed because not all configured rewriter roles produced usable output: "
            + ", ".join(missing)
        )


def _validate_review_outputs(peer_reviews: list[dict], expected_candidate_count: int) -> None:
    if len(peer_reviews) == 0:
        raise RuntimeError(
            "Review process failed because no reviewer outputs were returned."
        )

    incomplete = []
    for review in peer_reviews:
        reviewer = review.get("reviewer", "Unknown reviewer")
        evaluation = str(review.get("evaluation") or "").strip()
        ranking = review.get("parsed_ranking") or []
        unique_ranking = list(dict.fromkeys(ranking))
        if not evaluation or len(unique_ranking) != expected_candidate_count:
            incomplete.append(reviewer)

    if incomplete:
        raise RuntimeError(
            f"Review process failed because these reviewer roles did not return exactly {expected_candidate_count} distinct ranked candidates: "
            + ", ".join(incomplete)
        )


def run_pipeline(
    raw_query: str,
    domain: str = "general",
    demo_mode: bool = False,
    progress_callback=None,
    *,
    intent_model: str | None = None,
    rewriter_specs: list[dict[str, Any]] | None = None,
    reviewer_models: list[str] | None = None,
    chairman_model: str | None = None,
    record_analytics: bool = True,
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

    active_rewriter_specs = rewriter_specs or DEFAULT_REWRITER_SPECS
    if not active_rewriter_specs:
        raise RuntimeError("At least one rewriter must be configured.")

    # S1: Intent Extraction
    notify("intent", "Extracting intent from query", 10)
    intent = extract_intent(raw_query, demo_mode=demo_mode, model_name=intent_model)
    state["intent"] = intent
    notify("intent_complete", "Intent extraction complete", 25)

    # S2: Parallel Agent Rewriting
    notify("rewriters", f"Rewriting with {len(active_rewriter_specs)} parallel strategies", 30)
    # Use a fresh event loop to avoid conflicts when called from asyncio.to_thread()
    loop = asyncio.new_event_loop()
    try:
        candidate_outputs = loop.run_until_complete(
            run_rewriters_async(raw_query, intent, demo_mode, active_rewriter_specs)
        )
    finally:
        loop.close()
    raw_candidates = {
        spec["candidate_name"]: text
        for spec, text in zip(active_rewriter_specs, candidate_outputs)
    }
    _validate_candidate_outputs(raw_candidates)

    state["candidate_a"] = ""
    state["candidate_b"] = ""
    state["candidate_c"] = ""
    state["all_candidates"] = {}
    state["candidate_order"] = []
    state["perspectives"] = {}

    candidates_for_review: list[tuple[str, str]] = []
    for spec, candidate_text in zip(active_rewriter_specs, candidate_outputs):
        prompt, perspective = extract_prompt_and_perspective(
            candidate_text,
            spec["perspective"],
            step=f"rewriter_{str(spec['agent_key']).lower()}",
        )
        state[f"candidate_{str(spec['agent_key']).lower()}"] = prompt
        state["all_candidates"][spec["candidate_name"]] = prompt
        state["candidate_order"].append(spec["candidate_name"])
        state["perspectives"][spec["candidate_name"]] = perspective
        candidates_for_review.append((spec["candidate_name"], prompt))
    notify("rewriters_complete", "All rewriters finished", 60)

    # S3a + S3b: Council peer review + aggregate ranking
    notify("review", "Council: anonymised peer review in progress", 65)
    reviews, aggregate, label_map, diagnostics = peer_review_candidates(
        raw_query=raw_query,
        candidates=candidates_for_review,
        reviewer_models=reviewer_models,
        demo_mode=demo_mode,
    )
    state["peer_reviews"] = reviews
    state["aggregate_rankings"] = aggregate
    state["label_map"] = label_map
    state["consensus_diagnostics"] = diagnostics
    _validate_review_outputs(reviews, len(candidates_for_review))
    notify("review_complete", "Council review complete — aggregating rankings", 85)

    # S3c: Chairman synthesis
    notify("chairman", "Chairman synthesising final prompt", 88)
    optimised_prompt, chairman_info = chairman_synthesise_candidates(
        raw_query=raw_query,
        candidates=candidates_for_review,
        peer_reviews=reviews,
        aggregate=aggregate,
        label_map=label_map,
        topic_domain=intent.get("topic_domain", domain),
        chairman_model=chairman_model,
        demo_mode=demo_mode,
    )
    state["chairman"] = chairman_info
    state["optimised_prompt"] = optimised_prompt
    notify("complete", "Consensus reached", 100)

    # Analytics Logging
    try:
        if record_analytics and aggregate and len(aggregate) > 0:
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
    model_name: str | None = None,
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
        model_name or target_model,
        temperature=0.7,
        max_tokens=None,
    )
    return content
