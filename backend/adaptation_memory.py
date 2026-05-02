"""
adaptation_memory.py
Summaries of how humans accepted, refined, or overrode council consensus.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List


SESSIONS_FILE = os.path.join(os.path.dirname(__file__), "sessions.json")


def _read_sessions() -> List[Dict[str, Any]]:
    if not os.path.exists(SESSIONS_FILE):
        return []
    try:
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def build_adaptation_context(limit: int = 8, topic_domain: str | None = None) -> str:
    """
    Summarize recent acceptance vs override behavior so the chairman can learn
    what kinds of consensus humans tend to keep or correct.
    """
    sessions = list(reversed(_read_sessions()))
    if not sessions:
        return ""

    normalized_domain = (topic_domain or "").strip().lower()

    accepted = 0
    refined = 0
    overrode = 0
    accepted_reasons: Dict[str, int] = {}
    override_reasons: Dict[str, int] = {}

    filtered_sessions: List[Dict[str, Any]] = []
    for session in sessions:
        session_domain = (
            session.get("domain")
            or (session.get("intent") or {}).get("topic_domain")
            or "general"
        )
        session_domain = str(session_domain).strip().lower()
        if normalized_domain and session_domain != normalized_domain:
            continue
        filtered_sessions.append(session)
        if len(filtered_sessions) >= limit:
            break

    for session in filtered_sessions:
        insights = session.get("research_insights") or {}
        outcome = (insights.get("consensus_response") or "").strip().lower()
        labels = session.get("intervention_labels") or []

        if outcome == "accepted":
            accepted += 1
            for label in labels or ["Accepted council output as-is"]:
                accepted_reasons[label] = accepted_reasons.get(label, 0) + 1
        elif outcome == "overrode":
            overrode += 1
            for label in labels:
                override_reasons[label] = override_reasons.get(label, 0) + 1
        elif outcome:
            refined += 1
            for label in labels:
                override_reasons[label] = override_reasons.get(label, 0) + 1

    def top_items(counts: Dict[str, int]) -> str:
        if not counts:
            return "none recorded"
        ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:3]
        return ", ".join(f"{label} ({count})" for label, count in ordered)

    scope = f" for the '{normalized_domain}' domain" if normalized_domain else ""
    return (
        f"Human consensus adaptation memory{scope}:\n"
        f"- Accepted without override: {accepted}\n"
        f"- Accepted with refinement: {refined}\n"
        f"- Overrode consensus: {overrode}\n"
        f"- Common acceptance patterns: {top_items(accepted_reasons)}\n"
        f"- Common override/refinement patterns: {top_items(override_reasons)}\n"
        "Use this to prefer consensus traits humans usually keep, and to correct traits humans repeatedly revise."
    )
