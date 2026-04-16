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
from langchain_core.messages import HumanMessage, SystemMessage
from feedback_memory import build_chairman_feedback_context

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

REVIEW_SYSTEM = """You are a senior AI Prompt Engineer. You recently generated one of the 3 candidate prompts below. 
Your task is to critically cross-examine ALL THREE candidate prompts (Response X, Response Y, and Response Z) to determine the absolute best prompt optimization.
You MUST provide a brief critique of each candidate's strengths and weaknesses regarding structural rigor, logic, and constraint adherence.

IMPORTANT: End your review with a strict final ranking in this exact format:
FINAL RANKING:
1. Response [letter]
2. Response [letter]
3. Response [letter]

Do not add text after the ranking."""

CHAIRMAN_SYSTEM = """You are the chairman of a prompt-evaluation council. Based on peer reviews and aggregate rankings, synthesise the single best prompt by combining the strongest elements from all candidates.

When human preference memory is provided, treat it as evidence about which prompt traits people trusted, approved, or edited toward in prior sessions. Prefer those traits when they fit the current task.

Return ONLY the synthesised prompt, with no preamble or explanation."""


# ── Core functions ─────────────────────────────────────────────────────────────

def get_llm(model_name: str, temperature: float = 0.0):
    """Return a LangChain chat model for the given hint."""
    import os
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=model_name,
        temperature=temperature,
        max_tokens=1000
    )


def _parse_ranking(text: str) -> List[str]:
    """Extract FINAL RANKING section from review text."""
    if "FINAL RANKING:" in text:
        section = text.split("FINAL RANKING:")[1]
        matches = re.findall(r'\d+\.\s*(Response [A-Z])', section)
        if matches:
            return matches
    return re.findall(r'Response [A-Z]', text)


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


def _single_review(reviewer_name: str, reviewer_model: str, user_query: str, anonymised_text: str) -> Dict:
    """Run one cross-examination peer review."""
    llm = get_llm(reviewer_model)
    system = f"You are '{reviewer_name}', a senior AI Prompt Engineer." + "\n" + REVIEW_SYSTEM
    messages = [
        HumanMessage(content=f"{system}\n\nOriginal user query: {user_query}\n\n{anonymised_text}"),
    ]
    response = llm.invoke(messages)
    text = response.content.strip()
    return {
        "reviewer": reviewer_name,
        "model": reviewer_model,
        "evaluation": text,
        "parsed_ranking": _parse_ranking(text),
    }


def peer_review(
    raw_query: str,
    candidate_a: str, candidate_b: str, candidate_c: str,
    demo_mode: bool = False,
) -> Tuple[List[Dict], List[Dict], Dict[str, str]]:
    """
    S3a+S3b: Anonymised peer review + aggregate ranking.
    Returns (peer_reviews, aggregate_rankings, label_map).
    """
    if demo_mode:
        return DEMO_PEER_REVIEWS, DEMO_AGGREGATE, DEMO_LABEL_MAP

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
            loop.run_in_executor(None, _single_review, name, model, raw_query, anonymised_text)
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

    return peer_reviews, aggregate, label_map


def chairman_synthesise(
    raw_query: str,
    candidate_a: str, candidate_b: str, candidate_c: str,
    peer_reviews: List[Dict], aggregate: List[Dict], label_map: Dict[str, str],
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
    llm = get_llm(MODELS["chairman"], temperature=0.7)
    feedback_memory = build_chairman_feedback_context()

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
            f"Original query: {raw_query}\n\n"
            f"Candidate A (Chain-of-Thought):\n{candidate_a}\n\n"
            f"Candidate B (Role-Assignment):\n{candidate_b}\n\n"
            f"Candidate C (Structured Template):\n{candidate_c}\n\n"
            f"PEER REVIEWS:\n{reviews_text}\n\n"
            f"AGGREGATE RANKING:\n{ranking_text}\n\n"
            f"Synthesise the optimal prompt now."
        )),
    ]

    response = llm.invoke(messages)
    optimised = response.content.strip()

    # Fallback: if the model returned a label/reference instead of an actual prompt,
    # use the winning candidate's real prompt text
    candidates_by_agent = {"Candidate A": candidate_a, "Candidate B": candidate_b, "Candidate C": candidate_c}
    is_label_like = len(optimised.split()) < 15 or "Response " in optimised or "Candidate " in optimised
    if is_label_like and aggregate:
        winner_label = aggregate[0]["label"]
        winner_agent = label_map.get(winner_label, "")
        optimised = candidates_by_agent.get(winner_agent, candidate_a)

    chairman_info = {
        "model": MODELS["chairman"],
        "rationale": f"Based on council consensus: {aggregate[0]['label']} ranked first with average rank {aggregate[0]['average_rank']}.",
    }

    return optimised, chairman_info
