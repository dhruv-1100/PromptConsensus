"""
agents/council.py
LLM Council peer review — replaces the single arbitrator.

After the three rewriting agents produce candidates, this module:
  S3a  Anonymised peer review   (3 parallel LLM calls)
  S3b  Aggregate ranking        (pure computation)
  S3c  Chairman synthesis       (1 LLM call)
"""
import json
import os
import re
import asyncio
from typing import List, Dict, Any, Tuple
from langchain_core.messages import HumanMessage
from feedback_memory import build_chairman_feedback_context
from adaptation_memory import build_adaptation_context
from live_mode_utils import invoke_openrouter_model

# ── Demo fixtures ──────────────────────────────────────────────────────────────

DEMO_PEER_REVIEWS = [
    {
        "reviewer": "Autonomous Engineer A",
        "model": "gpt-4o",
        "evaluation": (
            "Response X provides strong step-by-step reasoning scaffolds that guide the model through "
            "each section of the discharge summary. However, it lacks explicit output format constraints "
            "and does not address HIPAA compliance.\n\n"
            "Response Y assigns a relevant expert persona (board-certified internist) and offers "
            "few-shot context, but the reasoning is less structured and could lead to a more free-form "
            "response that misses sections.\n\n"
            "Response Z combines structured output requirements with domain constraints (HIPAA, reading "
            "level) and step-by-step guidance. It is the most complete, though slightly verbose.\n\n"
            "FINAL RANKING:\n1. Response Z\n2. Response X\n3. Response Y"
        ),
        "parsed_ranking": ["Response Z", "Response X", "Response Y"],
    },
    {
        "reviewer": "Autonomous Engineer B",
        "model": "claude-3-5-sonnet",
        "evaluation": (
            "Response X excels at breaking down the task into numbered steps, ensuring no section is "
            "missed. The instruction to 'complete each section before proceeding' is effective for "
            "structured output. Weakness: no output format template.\n\n"
            "Response Y effectively uses role-assignment (board-certified internist) which grounds the "
            "model in clinical expertise. However, the prompt is more open-ended and may produce "
            "inconsistent formatting.\n\n"
            "Response Z provides a complete template with table format for medications and explicit "
            "constraints (HIPAA, Grade 8 reading level). Combined with step-by-step reasoning, this "
            "is the most production-ready prompt.\n\n"
            "FINAL RANKING:\n1. Response Z\n2. Response X\n3. Response Y"
        ),
        "parsed_ranking": ["Response Z", "Response X", "Response Y"],
    },
    {
        "reviewer": "Autonomous Engineer C",
        "model": "gpt-4o",
        "evaluation": (
            "Response X: Strong reasoning scaffolds with clear step numbers. Good for ensuring "
            "completeness. Lacks format specification — the model might choose its own layout.\n\n"
            "Response Y: The persona assignment adds domain authority, and the few-shot framing is "
            "useful. But without explicit output structure, the result could vary across runs.\n\n"
            "Response Z: Best balance of structure, domain constraints, and reasoning guidance. The "
            "medication table template and compliance requirements make it immediately usable in a "
            "clinical context. Step-by-step reasoning ensures thoroughness.\n\n"
            "FINAL RANKING:\n1. Response Z\n2. Response Y\n3. Response X"
        ),
        "parsed_ranking": ["Response Z", "Response Y", "Response X"],
    },
]

DEMO_AGGREGATE = [
    {"label": "Response Z", "candidate": "C", "average_rank": 1.0, "votes": 3, "ranks": [1, 1, 1]},
    {"label": "Response X", "candidate": "A", "average_rank": 2.0, "votes": 3, "ranks": [2, 2, 3]},
    {"label": "Response Y", "candidate": "B", "average_rank": 2.67, "votes": 3, "ranks": [3, 3, 2]},
]

DEMO_LABEL_MAP = {
    "Response X": "Candidate A",
    "Response Y": "Candidate B",
    "Response Z": "Candidate C",
}

DEMO_CHAIRMAN = {
    "model": "claude-3-5-sonnet",
    "rationale": (
        "All three reviewers ranked the structured-template candidate first, citing its complete "
        "output format, domain compliance constraints, and combined step-by-step guidance. "
        "The synthesis below merges the structural rigor of that candidate with the explicit "
        "reasoning scaffolds from the chain-of-thought candidate."
    ),
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
| [Entry] | | | Flag if high-risk |

**Follow-Up:**
- Endocrinology: within 2 weeks
- Primary Care: within 4 weeks
- Daily glucose targets: 80–130 mg/dL fasting

**Constraints:** HIPAA compliant language; patient instructions at Grade 8 reading level; flag high-risk medications."""

# ── Prompt templates ───────────────────────────────────────────────────────────

REVIEW_SYSTEM = """You are a senior prompt reviewer in a three-candidate council.

Evaluate all three candidate prompts strictly on prompt quality for the user's task.

Focus on:
- faithfulness to the user's intent
- structural clarity and usability
- completeness and constraint quality
- domain fit
- whether the prompt avoids harmful overcompression such as unnecessary single-paragraph or strict brevity requirements
- whether the prompt avoids redirecting the task into requests for more user-supplied materials such as PDFs, URLs, uploads, DOIs, or external documents

Give a concise strengths-and-weaknesses critique for each candidate.

IMPORTANT: End with a strict final ranking in exactly this format:
FINAL RANKING:
1. Response [letter]
2. Response [letter]
3. Response [letter]

Do not add any text after the ranking."""

# Constitutional principles for each reviewer (grounded in Constitutional AI, Lecture 8)
CONSTITUTIONAL_PRINCIPLES = {
    "Autonomous Engineer A": (
        "Constitutional principle — FAITHFULNESS TO USER INTENT: "
        "Pay special attention to whether each candidate preserves and serves the user's "
        "original goal without drifting, expanding scope unnecessarily, or reinterpreting "
        "the task. Penalise prompts that subtly shift what the user asked for."
    ),
    "Autonomous Engineer B": (
        "Constitutional principle — OUTPUT SAFETY AND BIAS: "
        "Pay special attention to whether each candidate could produce harmful, biased, "
        "misleading, or ethically questionable output. Penalise prompts that lack "
        "appropriate guardrails for sensitive content, omit disclaimers where needed, "
        "or could amplify stereotypes or misinformation."
    ),
    "Autonomous Engineer C": (
        "Constitutional principle — DOMAIN APPROPRIATENESS: "
        "Pay special attention to whether each candidate respects domain-specific norms, "
        "conventions, and constraints. Penalise prompts that apply generic templates to "
        "specialised fields without adapting terminology, structure, or compliance requirements."
    ),
}

CHAIRMAN_SYSTEM = """You are the chairman of a prompt-evaluation council.

Your job is to synthesize the strongest final prompt from the three candidate prompts, the peer reviews, and the aggregate ranking.

Synthesis priorities:
- preserve the user's actual goal
- keep the prompt directly usable
- prefer clarity, structure, and domain fit
- combine the strongest compatible elements across candidates
- remove unnecessary restrictions that degrade output quality

Use memory carefully:
- When human preference memory is provided, treat it as evidence about traits users trusted, approved, or edited toward in similar prior sessions.
- When adaptation memory is provided, treat it as evidence about which consensus patterns humans usually kept versus overrode.
- Use memory only when it fits the current task and domain.

Hard rules:
- Do not turn the synthesized prompt into a request for more user-supplied materials such as PDFs, URLs, uploads, DOIs, or external documents.
- Do not introduce strict brevity constraints, single-paragraph requirements, word limits, paragraph caps, or bullet-count caps unless the user explicitly requested them.
- If the task is summarization, prefer structured output with sections, headings, or bullets rather than a single paragraph.
- For research tasks, prefer enough space for nuanced findings, methodology, limitations, evidence, and uncertainty.

Return ONLY the synthesized prompt, with no preamble or explanation."""


# ── Core functions ─────────────────────────────────────────────────────────────

def _parse_ranking(text: str) -> List[str]:
    """Extract FINAL RANKING section from review text."""
    def dedupe_preserve_order(labels: List[str]) -> List[str]:
        unique: List[str] = []
        seen: set[str] = set()
        for label in labels:
            if label not in seen:
                unique.append(label)
                seen.add(label)
        return unique

    numbered_pattern = re.compile(
        r"(?im)^\s*(?:[#*\-]\s*)?(\d+)\s*[\.\):\-]?\s*(Response [A-Z])\b"
    )
    final_ranking_match = re.search(r"(?is)\bfinal\s+ranking\b\s*:?\s*(.+)$", text)
    if final_ranking_match:
        section = final_ranking_match.group(1)
        ordered_labels = [label for _, label in numbered_pattern.findall(section)]
        unique_labels = dedupe_preserve_order(ordered_labels)
        if len(unique_labels) == 3:
            return unique_labels[:3]

    ordered_labels = [label for _, label in numbered_pattern.findall(text)]
    unique_labels = dedupe_preserve_order(ordered_labels)
    if len(unique_labels) >= 3:
        return unique_labels[-3:]

    mentioned_labels = dedupe_preserve_order(re.findall(r"Response [A-Z]", text))
    if len(mentioned_labels) == 3:
        return mentioned_labels

    return []


def _consensus_diagnostics(peer_reviews: List[Dict], aggregate: List[Dict]) -> Dict[str, Any]:
    """
    Measure how strong the council agreement actually was.
    """
    if not aggregate:
        return {
            "winner_label": None,
            "winner_candidate": None,
            "winner_average_rank": None,
            "winner_margin": 0.0,
            "first_place_support_pct": 0.0,
            "reviewer_agreement_pct": 0.0,
            "consensus_strength_pct": 0.0,
            "consensus_label": "unknown",
            "is_unanimous_winner": False,
            "needs_human_review": True,
        }

    winner = aggregate[0]
    reviewer_count = len(peer_reviews)
    first_place_votes = sum(
        1 for review in peer_reviews if (review.get("parsed_ranking") or [None])[0] == winner.get("label")
    )
    first_place_support = (first_place_votes / reviewer_count) if reviewer_count else 0.0

    def ranking_similarity(left: List[str], right: List[str]) -> float:
        items = [item for item in left if item in right]
        if len(items) < 2:
            return 1.0
        position_right = {item: idx for idx, item in enumerate(right)}
        concordant = 0
        total = 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                total += 1
                if position_right[items[i]] < position_right[items[j]]:
                    concordant += 1
        return concordant / total if total else 1.0

    similarities = []
    for i in range(len(peer_reviews)):
        for j in range(i + 1, len(peer_reviews)):
            similarities.append(
                ranking_similarity(
                    peer_reviews[i].get("parsed_ranking") or [],
                    peer_reviews[j].get("parsed_ranking") or [],
                )
            )
    reviewer_agreement = sum(similarities) / len(similarities) if similarities else (1.0 if reviewer_count else 0.0)
    consensus_strength = (first_place_support + reviewer_agreement) / 2 if reviewer_count else 0.0

    if consensus_strength >= 0.8:
        label = "high"
    elif consensus_strength >= 0.55:
        label = "moderate"
    else:
        label = "weak"

    runner_up = aggregate[1] if len(aggregate) > 1 else {}
    winner_margin = round(float(runner_up.get("average_rank", 0) or 0) - float(winner.get("average_rank", 0) or 0), 2)
    return {
        "winner_label": winner.get("label"),
        "winner_candidate": winner.get("candidate"),
        "winner_average_rank": winner.get("average_rank"),
        "winner_margin": winner_margin,
        "first_place_support_pct": round(first_place_support * 100, 1),
        "reviewer_agreement_pct": round(reviewer_agreement * 100, 1),
        "consensus_strength_pct": round(consensus_strength * 100, 1),
        "consensus_label": label,
        "is_unanimous_winner": first_place_votes == reviewer_count and reviewer_count > 0,
        "needs_human_review": label != "high",
    }


def _anonymise_candidates(
    candidate_a: str, candidate_b: str, candidate_c: str
) -> Tuple[str, Dict[str, str]]:
    """Shuffle candidates into anonymous labels and return (prompt_text, label_map)."""
    import random
    items = [
        ("Candidate A", candidate_a),
        ("Candidate B", candidate_b),
        ("Candidate C", candidate_c),
    ]
    random.shuffle(items)
    labels = ["X", "Y", "Z"]
    label_map = {}
    parts = []
    for label, (agent_name, text) in zip(labels, items):
        label_map[f"Response {label}"] = agent_name
        parts.append(f"Response {label}:\n{text}")
    return "\n\n---\n\n".join(parts), label_map


def _single_review(
    reviewer_name: str,
    reviewer_model: str,
    user_query: str,
    anonymised_text: str,
    constitutional_principle: str = "",
) -> Dict:
    """Run one cross-examination peer review."""
    system = f"You are '{reviewer_name}', a senior AI Prompt Engineer." + "\n" + REVIEW_SYSTEM
    if constitutional_principle:
        system += "\n\n" + constitutional_principle
    messages = [
        HumanMessage(content=f"{system}\n\nOriginal user query: {user_query}\n\n{anonymised_text}"),
    ]
    text, actual_model = invoke_openrouter_model(
        messages,
        reviewer_model,
        temperature=0.0,
        max_tokens=1000,
    )
    return {
        "reviewer": reviewer_name,
        "model": actual_model,
        "evaluation": text,
        "parsed_ranking": _parse_ranking(text),
    }


def peer_review(
    raw_query: str,
    candidate_a: str, candidate_b: str, candidate_c: str,
    demo_mode: bool = False,
) -> Tuple[List[Dict], List[Dict], Dict[str, str], Dict[str, Any]]:
    """
    S3a+S3b: Anonymised peer review + aggregate ranking.
    Returns (peer_reviews, aggregate_rankings, label_map).
    """
    if demo_mode:
        diagnostics = _consensus_diagnostics(DEMO_PEER_REVIEWS, DEMO_AGGREGATE)
        return DEMO_PEER_REVIEWS, DEMO_AGGREGATE, DEMO_LABEL_MAP, diagnostics

    anonymised_text, label_map = _anonymise_candidates(candidate_a, candidate_b, candidate_c)

    reviewer_names = ["Autonomous Engineer A", "Autonomous Engineer B", "Autonomous Engineer C"]

    # Run reviews in parallel
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    from config import MODELS
    reviewer_models = [MODELS["reviewer_a"], MODELS["reviewer_b"], MODELS["reviewer_c"]]
    async def _run():
        tasks = [
            loop.run_in_executor(
                None,
                _single_review,
                name,
                model,
                raw_query,
                anonymised_text,
                CONSTITUTIONAL_PRINCIPLES.get(name, ""),
            )
            for name, model in zip(reviewer_names, reviewer_models)
        ]
        return await asyncio.gather(*tasks)

    reviews_raw = loop.run_until_complete(_run())

    peer_reviews = []
    for i, r in enumerate(reviews_raw):
        r["reviewer"] = f"Reviewer {i + 1} ({r['model']})"
        peer_reviews.append(r)

    # Aggregate rankings
    from collections import defaultdict
    positions = defaultdict(list)
    for review in peer_reviews:
        for pos, label in enumerate(review["parsed_ranking"], 1):
            if label in label_map:
                positions[label].append(pos)

    aggregate = []
    for label, ranks in positions.items():
        agent_name = label_map.get(label, label)
        # Extract candidate letter from "Candidate A" → "A", etc.
        candidate_letter = agent_name.replace("Candidate ", "") if agent_name.startswith("Candidate ") else "?"
        aggregate.append({
            "label": label,
            "candidate": candidate_letter,
            "average_rank": round(sum(ranks) / len(ranks), 2),
            "votes": len(ranks),
            "ranks": ranks,
        })
    aggregate.sort(key=lambda x: x["average_rank"])

    if not aggregate:
        for label, agent_name in label_map.items():
            candidate_letter = agent_name.replace("Candidate ", "") if agent_name.startswith("Candidate ") else "?"
            aggregate.append({
                "label": label,
                "candidate": candidate_letter,
                "average_rank": 2.0,
                "votes": 0,
                "ranks": [],
            })

    diagnostics = _consensus_diagnostics(peer_reviews, aggregate)
    return peer_reviews, aggregate, label_map, diagnostics


def chairman_synthesise(
    raw_query: str,
    candidate_a: str, candidate_b: str, candidate_c: str,
    peer_reviews: List[Dict], aggregate: List[Dict], label_map: Dict[str, str],
    topic_domain: str = "general",
    demo_mode: bool = False,
) -> Tuple[str, Dict]:
    """
    S3c: Chairman synthesises the final prompt from council output.
    Returns (optimised_prompt, chairman_info).
    """
    if demo_mode:
        return DEMO_OPTIMISED, DEMO_CHAIRMAN

    from config import MODELS
    # Chairman uses a high-competency model like DeepSeek to synthesise
    # Removed specific handling for gemini/gemma to default to OpenRouter via get_llm
    feedback_memory = build_chairman_feedback_context(topic_domain=topic_domain)
    adaptation_memory = build_adaptation_context(topic_domain=topic_domain)

    reviews_text = "\n\n".join([
        f"{r['reviewer']}:\n{r['evaluation']}" for r in peer_reviews
    ])
    ranking_text = "\n".join([
        f"{i + 1}. {a['label']} ({label_map.get(a['label'], '?')}) — avg rank {a['average_rank']}"
        for i, a in enumerate(aggregate)
    ])

    messages = [
        HumanMessage(content=(
            f"{CHAIRMAN_SYSTEM}\n\n"
            f"{feedback_memory}\n\n" if feedback_memory else f"{CHAIRMAN_SYSTEM}\n\n"
        ) + (
            f"{adaptation_memory}\n\n" if adaptation_memory else ""
        ) + (
            f"Original query: {raw_query}\n\n"
            f"Topic domain: {topic_domain}\n\n"
            f"Candidate A (Chain-of-Thought):\n{candidate_a}\n\n"
            f"Candidate B (Role-Assignment):\n{candidate_b}\n\n"
            f"Candidate C (Structured Template):\n{candidate_c}\n\n"
            f"PEER REVIEWS:\n{reviews_text}\n\n"
            f"AGGREGATE RANKING:\n{ranking_text}\n\n"
            f"Synthesise the optimal prompt now."
        )),
    ]

    optimised, actual_model = invoke_openrouter_model(
        messages,
        MODELS["chairman"],
        temperature=0.7,
        max_tokens=1000,
    )

    # Fallback: if the model returned a label/reference instead of an actual prompt,
    # use the winning candidate's real prompt text
    candidates_by_agent = {"Candidate A": candidate_a, "Candidate B": candidate_b, "Candidate C": candidate_c}
    is_label_like = len(optimised.split()) < 15 or "Response " in optimised or "Candidate " in optimised
    if is_label_like and aggregate:
        winner_label = aggregate[0]["label"]
        winner_agent = label_map.get(winner_label, "")
        optimised = candidates_by_agent.get(winner_agent, candidate_a)

    if aggregate:
        rationale = f"Based on council consensus: {aggregate[0]['label']} ranked first with average rank {aggregate[0]['average_rank']}."
    else:
        rationale = "Reviewer outputs did not yield a stable ranking, so the synthesis relied on the candidate prompts and qualitative review text."

    chairman_info = {
        "model": actual_model,
        "rationale": rationale,
    }

    return optimised, chairman_info
