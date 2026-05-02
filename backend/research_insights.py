"""
research_insights.py
Derived HCAI-oriented metrics for ConsensusPrompt study sessions.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List
import re

from idiosyncrasy_detector import (
    candidate_diversity_report as _candidate_diversity,
)


def _safe_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard_distance(a: str, b: str) -> float:
    a_tokens = _tokenize(a)
    b_tokens = _tokenize(b)
    if not a_tokens and not b_tokens:
        return 0.0
    union = a_tokens | b_tokens
    if not union:
        return 0.0
    return 1.0 - (len(a_tokens & b_tokens) / len(union))


def _line_change_ratio(before: str, after: str) -> float:
    before_lines = [line.strip() for line in _safe_text(before).splitlines() if line.strip()]
    after_lines = [line.strip() for line in _safe_text(after).splitlines() if line.strip()]
    base = max(len(before_lines), len(after_lines), 1)
    overlap = len(set(before_lines) & set(after_lines))
    changed = max(len(before_lines), len(after_lines)) - overlap
    return changed / base


def _kendall_similarity(order_a: List[str], order_b: List[str]) -> float:
    items = [item for item in order_a if item in order_b]
    if len(items) < 2:
        return 1.0

    position_b = {item: idx for idx, item in enumerate(order_b)}
    concordant = 0
    total = 0
    for left, right in combinations(items, 2):
        total += 1
        if position_b[left] < position_b[right]:
            concordant += 1
    return concordant / total if total else 1.0


def _bucket_level(value: float, high: float, medium: float) -> str:
    if value >= high:
        return "high"
    if value >= medium:
        return "moderate"
    return "mixed"


def _normalize_labels(labels: List[Any]) -> List[str]:
    return [str(label).strip() for label in labels if str(label).strip()]


def _classify_consensus_response(
    human_edit_shift: float,
    intervention_labels: List[str],
) -> str:
    normalized = [label.lower() for label in intervention_labels]
    override_markers = (
        "restore my original intent",
        "preferred the baseline/raw condition more",
    )
    refine_markers = (
        "clarity or structure",
        "domain specificity",
        "caution or safety",
    )

    if any(marker in label for label in normalized for marker in override_markers):
        return "overrode"
    if human_edit_shift >= 0.35:
        return "overrode"
    if human_edit_shift == 0 and (
        not normalized or any("accepted council output as-is" in label for label in normalized)
    ):
        return "accepted"
    if any(marker in label for label in normalized for marker in refine_markers):
        return "refined"
    if human_edit_shift >= 0.08:
        return "refined"
    return "accepted"


def build_research_insights(session: Dict[str, Any]) -> Dict[str, Any]:
    peer_reviews = session.get("peer_reviews") or []
    aggregate_rankings = session.get("aggregate_rankings") or []
    winner = aggregate_rankings[0] if aggregate_rankings else {}
    winner_label = winner.get("label", "")
    winner_candidate = winner.get("candidate", "")

    first_place_votes = sum(
        1
        for review in peer_reviews
        if (review.get("parsed_ranking") or [None])[0] == winner_label
    )
    reviewer_count = len(peer_reviews)
    first_place_support = (first_place_votes / reviewer_count) if reviewer_count else 0.0

    similarities: List[float] = []
    for left, right in combinations(peer_reviews, 2):
        ranking_left = left.get("parsed_ranking") or []
        ranking_right = right.get("parsed_ranking") or []
        similarities.append(_kendall_similarity(ranking_left, ranking_right))

    reviewer_agreement = sum(similarities) / len(similarities) if similarities else (1.0 if reviewer_count else 0.0)
    consensus_strength = (first_place_support + reviewer_agreement) / 2 if reviewer_count else 0.0

    second = aggregate_rankings[1] if len(aggregate_rankings) > 1 else {}
    winner_margin = float(second.get("average_rank", 0) or 0) - float(winner.get("average_rank", 0) or 0)

    candidate_texts = [
        _safe_text(session.get("candidate_a")),
        _safe_text(session.get("candidate_b")),
        _safe_text(session.get("candidate_c")),
    ]
    candidate_texts = [text for text in candidate_texts if text.strip()]
    diversity_scores = [
        _jaccard_distance(left, right)
        for left, right in combinations(candidate_texts, 2)
    ]
    rewrite_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0

    raw_query = _safe_text(session.get("raw_query"))
    optimised_prompt = _safe_text(session.get("optimised_prompt"))
    final_prompt = _safe_text(session.get("final_prompt"))
    baseline_response = _safe_text(session.get("baseline_response"))
    llm_response = _safe_text(session.get("llm_response"))

    optimization_shift = _line_change_ratio(raw_query, optimised_prompt)
    human_edit_shift = _line_change_ratio(optimised_prompt, final_prompt)

    if human_edit_shift == 0:
        human_edit_level = "none"
    elif human_edit_shift < 0.2:
        human_edit_level = "light"
    else:
        human_edit_level = "substantial"

    output_delta = 0.0
    if baseline_response.strip():
        baseline_words = len(baseline_response.split())
        optimised_words = len(llm_response.split())
        if max(baseline_words, 1):
            output_delta = (optimised_words - baseline_words) / max(baseline_words, 1)

    intervention_labels = _normalize_labels(session.get("intervention_labels") or [])
    consensus_response = _classify_consensus_response(human_edit_shift, intervention_labels)

    # ── Candidate diversity (pattern convergence) ──
    cand_a = _safe_text(session.get("candidate_a"))
    cand_b = _safe_text(session.get("candidate_b"))
    cand_c = _safe_text(session.get("candidate_c"))
    candidate_convergence = "unknown"
    candidate_token_diversity = 0.0
    candidate_structural_diversity = 0.0
    candidate_style_divergence = 0.0
    if cand_a.strip() and cand_b.strip() and cand_c.strip():
        diversity_info = _candidate_diversity(cand_a, cand_b, cand_c)
        candidate_convergence = diversity_info.get("convergence_label", "unknown")
        candidate_token_diversity = diversity_info.get("avg_token_diversity", 0.0)
        candidate_structural_diversity = diversity_info.get("avg_structural_diversity", 0.0)
        candidate_style_divergence = diversity_info.get("avg_style_divergence", 0.0)

    return {
        "winner_candidate": winner_candidate or winner_label or None,
        "reviewer_count": reviewer_count,
        "winner_first_place_support_pct": round(first_place_support * 100, 1),
        "reviewer_agreement_pct": round(reviewer_agreement * 100, 1),
        "consensus_strength_pct": round(consensus_strength * 100, 1),
        "consensus_label": _bucket_level(consensus_strength, 0.8, 0.55),
        "winner_margin": round(winner_margin, 2),
        "rewrite_diversity_pct": round(rewrite_diversity * 100, 1),
        "diversity_label": _bucket_level(rewrite_diversity, 0.6, 0.35),
        "optimization_shift_pct": round(optimization_shift * 100, 1),
        "human_edit_shift_pct": round(human_edit_shift * 100, 1),
        "human_edit_level": human_edit_level,
        "accepted_without_edit": consensus_response == "accepted" and human_edit_shift == 0,
        "consensus_response": consensus_response,
        "accepted_with_refinement": consensus_response == "refined",
        "overrode_consensus": consensus_response == "overrode",
        "compare_mode": bool(session.get("compare_mode")),
        "response_length_delta_pct": round(output_delta * 100, 1) if baseline_response.strip() else None,
        "safety_acknowledged": bool(session.get("safety_acknowledged")),
        "intervention_labels": intervention_labels,
        # Candidate diversity metrics
        "candidate_pattern_convergence": candidate_convergence,
        "candidate_token_diversity": candidate_token_diversity,
        "candidate_structural_diversity": candidate_structural_diversity,
        "candidate_style_divergence": candidate_style_divergence,
    }
