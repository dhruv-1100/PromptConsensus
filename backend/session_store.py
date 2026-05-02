"""
session_store.py
Persistence and export helpers for full ConsensusPrompt study sessions.
"""
from __future__ import annotations

import csv
import io
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
from research_insights import build_research_insights


SESSIONS_FILE = os.path.join(os.path.dirname(__file__), "sessions.json")


def _read_sessions() -> List[Dict[str, Any]]:
    """Load stored sessions, returning an empty list on failure."""
    if not os.path.exists(SESSIONS_FILE):
        return []

    try:
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def append_session_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a full session record and return the stored payload."""
    sessions = _read_sessions()
    stored = {
        "session_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **entry,
    }
    stored["research_insights"] = build_research_insights(stored)
    sessions.append(stored)

    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=True)

    return stored


def list_sessions(limit: int | None = None) -> List[Dict[str, Any]]:
    """Return recent sessions, newest first."""
    sessions = []
    for session in reversed(_read_sessions()):
        hydrated = dict(session)
        hydrated["research_insights"] = build_research_insights(hydrated)
        sessions.append(hydrated)
    return sessions[:limit] if limit else sessions


def get_session_analytics() -> Dict[str, Any]:
    """Compute homepage analytics from stored study sessions."""
    sessions = _read_sessions()
    total = len(sessions)

    if total == 0:
        return {
            "total_sessions": 0,
            "avg_quality": 0,
            "avg_trust": 0,
            "avg_improvement": 0,
            "avg_control": 0,
            "edit_rate": 0,
            "compare_mode_rate": 0,
            "top_domain": None,
            "top_winner": None,
            "avg_consensus_strength": 0,
            "avg_rewrite_diversity": 0,
            "avg_human_intervention": 0,
            "acceptance_rate": 0,
        }

    def avg(field: str) -> float:
        vals = [float(s.get(field, 0) or 0) for s in sessions]
        return round(sum(vals) / len(vals), 2)

    edited_count = 0
    compare_count = 0
    domain_counts: Dict[str, int] = {}
    winner_counts: Dict[str, int] = {}
    consensus_total = 0.0
    diversity_total = 0.0
    intervention_total = 0.0
    accepted_without_edit = 0

    for session in sessions:
        if (session.get("optimised_prompt") or "").strip() != (session.get("final_prompt") or "").strip():
            edited_count += 1
        if session.get("compare_mode"):
            compare_count += 1

        domain = (session.get("domain") or "general").strip() or "general"
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        aggregate = session.get("aggregate_rankings") or []
        winner = aggregate[0] if aggregate else {}
        winner_key = winner.get("candidate") or winner.get("label")
        if winner_key:
            winner_counts[winner_key] = winner_counts.get(winner_key, 0) + 1

        insights = session.get("research_insights") or build_research_insights(session)
        consensus_total += float(insights.get("consensus_strength_pct", 0) or 0)
        diversity_total += float(insights.get("rewrite_diversity_pct", 0) or 0)
        intervention_total += float(insights.get("human_edit_shift_pct", 0) or 0)
        if insights.get("accepted_without_edit"):
            accepted_without_edit += 1

    top_domain = max(domain_counts.items(), key=lambda item: item[1])[0] if domain_counts else None
    top_winner = max(winner_counts.items(), key=lambda item: item[1])[0] if winner_counts else None

    return {
        "total_sessions": total,
        "avg_quality": avg("quality"),
        "avg_trust": avg("trust"),
        "avg_improvement": avg("improvement"),
        "avg_control": avg("control"),
        "edit_rate": round((edited_count / total) * 100, 1),
        "compare_mode_rate": round((compare_count / total) * 100, 1),
        "top_domain": top_domain,
        "top_winner": top_winner,
        "avg_consensus_strength": round(consensus_total / total, 1),
        "avg_rewrite_diversity": round(diversity_total / total, 1),
        "avg_human_intervention": round(intervention_total / total, 1),
        "acceptance_rate": round((accepted_without_edit / total) * 100, 1),
    }


def export_sessions_csv() -> str:
    """Flatten stored sessions into a CSV string for quick analysis."""
    sessions = _read_sessions()
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "session_id",
            "timestamp",
            "domain",
            "compare_mode",
            "target_model",
            "chairman_model",
            "quality",
            "improvement",
            "trust",
            "control",
            "raw_query",
            "optimised_prompt",
            "final_prompt",
            "baseline_response",
            "llm_response",
            "safety_risk_level",
            "safety_check_count",
            "safety_acknowledged",
            "winner_label",
            "winner_candidate",
            "winner_average_rank",
            "consensus_label",
            "user_edited_prompt",
            "intervention_labels",
            "consensus_strength_pct",
            "reviewer_agreement_pct",
            "winner_first_place_support_pct",
            "rewrite_diversity_pct",
            "optimization_shift_pct",
            "human_edit_shift_pct",
            "human_edit_level",
            "consensus_response",
            "accepted_without_edit",
            "response_length_delta_pct",
            "comment",
        ],
    )
    writer.writeheader()

    for session in sessions:
        aggregate = session.get("aggregate_rankings") or []
        winner = aggregate[0] if aggregate else {}
        optimised_prompt = session.get("optimised_prompt", "")
        final_prompt = session.get("final_prompt", "")
        safety_report = session.get("safety_report") or {}
        safety_checks = safety_report.get("checks") or []
        insights = session.get("research_insights") or build_research_insights(session)

        writer.writerow(
            {
                "session_id": session.get("session_id", ""),
                "timestamp": session.get("timestamp", ""),
                "domain": session.get("domain", ""),
                "compare_mode": session.get("compare_mode", False),
                "target_model": session.get("target_model", ""),
                "chairman_model": session.get("chairman_model", ""),
                "quality": session.get("quality", ""),
                "improvement": session.get("improvement", ""),
                "trust": session.get("trust", ""),
                "control": session.get("control", ""),
                "raw_query": session.get("raw_query", ""),
                "optimised_prompt": optimised_prompt,
                "final_prompt": final_prompt,
                "baseline_response": session.get("baseline_response", ""),
                "llm_response": session.get("llm_response", ""),
                "safety_risk_level": safety_report.get("risk_level", "none"),
                "safety_check_count": len(safety_checks),
                "safety_acknowledged": session.get("safety_acknowledged", False),
                "winner_label": winner.get("label", ""),
                "winner_candidate": winner.get("candidate", ""),
                "winner_average_rank": winner.get("average_rank", ""),
                "consensus_label": insights.get("consensus_label", ""),
                "user_edited_prompt": optimised_prompt != final_prompt,
                "intervention_labels": " | ".join(session.get("intervention_labels") or []),
                "consensus_strength_pct": insights.get("consensus_strength_pct", ""),
                "reviewer_agreement_pct": insights.get("reviewer_agreement_pct", ""),
                "winner_first_place_support_pct": insights.get("winner_first_place_support_pct", ""),
                "rewrite_diversity_pct": insights.get("rewrite_diversity_pct", ""),
                "optimization_shift_pct": insights.get("optimization_shift_pct", ""),
                "human_edit_shift_pct": insights.get("human_edit_shift_pct", ""),
                "human_edit_level": insights.get("human_edit_level", ""),
                "consensus_response": insights.get("consensus_response", ""),
                "accepted_without_edit": insights.get("accepted_without_edit", ""),
                "response_length_delta_pct": insights.get("response_length_delta_pct", ""),
                "comment": session.get("text", ""),
            }
        )

    return output.getvalue()
