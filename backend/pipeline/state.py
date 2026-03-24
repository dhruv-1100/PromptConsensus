"""
pipeline/state.py
Shared state schema for the ConsensusPrompt pipeline.
"""
from typing import TypedDict, Optional, List, Dict


class ConsensusState(TypedDict, total=False):
    # S1 — Input
    raw_query: str
    domain: str            # e.g. "healthcare", "education", "general"
    target_model: str      # e.g. "gpt-4o", "claude-3-5-sonnet"

    # S1 — Intent extraction
    intent: dict           # {intent, domain, missing_info, constraints}

    # S2 — Three candidate rewrites
    candidate_a: str       # Chain-of-thought strategy
    candidate_b: str       # Role-assignment + few-shot strategy
    candidate_c: str       # Structured template + domain constraints

    # S3a — Council peer reviews
    peer_reviews: list     # [{reviewer, model, evaluation, parsed_ranking}, ...]
    label_map: dict        # {"Response X": "Agent A — Chain-of-Thought", ...}

    # S3b — Aggregate rankings
    aggregate_rankings: list  # [{label, candidate, average_rank, votes, ranks}, ...]

    # S3c — Chairman synthesis
    chairman: dict           # {model, rationale}
    optimised_prompt: str    # The chairman's synthesised best prompt

    # S4 — Human review
    human_action: str      # "approve" | "edit" | "reject"
    final_prompt: str      # User-approved (possibly edited) prompt

    # S5 — Execution & feedback
    llm_response: str
    feedback_rating: int   # 1–5
    feedback_text: str
