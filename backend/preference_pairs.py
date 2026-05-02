"""
preference_pairs.py
Extracts DPO-format preference pairs from ConsensusPrompt session data.

Grounded in course readings:
  - "Direct Preference Optimization" (DPO) — preference pair format
  - "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT)
  - "Constitutional AI: Harmlessness from AI Feedback"

Every time a user edits the council output before approval, a natural
(chosen, rejected) pair is created. This module extracts, structures,
and exports these pairs in standard JSONL format for downstream alignment.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional




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


# ── Preference Pair Extraction ────────────────────────────────────────────────

def extract_preference_pairs(
    domain_filter: Optional[str] = None,
    min_edit_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Extract DPO-format preference pairs from stored sessions.

    A preference pair is generated when:
      - The user edited the council output (chosen = final_prompt, rejected = optimised_prompt)
      - OR the user accepted without edits (recorded as a positive signal, type = 'accepted')

    Returns a list of structured pairs suitable for JSONL export.
    """
    sessions = _read_sessions()
    pairs: List[Dict[str, Any]] = []

    for session in sessions:
        raw_query = (session.get("raw_query") or "").strip()
        optimised_prompt = (session.get("optimised_prompt") or "").strip()
        final_prompt = (session.get("final_prompt") or "").strip()

        if not raw_query or not optimised_prompt or not final_prompt:
            continue

        session_domain = (
            session.get("domain")
            or (session.get("intent") or {}).get("topic_domain")
            or "general"
        )
        session_domain = str(session_domain).strip().lower()

        if domain_filter and session_domain != domain_filter.strip().lower():
            continue

        # Determine edit type
        was_edited = optimised_prompt != final_prompt
        insights = session.get("research_insights") or {}
        human_edit_shift = float(insights.get("human_edit_shift_pct", 0) or 0)

        if min_edit_threshold > 0 and human_edit_shift < min_edit_threshold:
            continue

        consensus_response = (insights.get("consensus_response") or "").strip()

        pair: Dict[str, Any] = {
            "prompt": raw_query,
            "chosen": final_prompt,
            "rejected": optimised_prompt if was_edited else "",
            "pair_type": "edited" if was_edited else "accepted",
            "domain": session_domain,
            "edit_shift_pct": human_edit_shift,
            "consensus_response": consensus_response,
            "metadata": {
                "session_id": session.get("session_id", ""),
                "timestamp": session.get("timestamp", ""),
                "quality_rating": session.get("quality", 0),
                "trust_rating": session.get("trust", 0),
                "improvement_rating": session.get("improvement", 0),
                "control_rating": session.get("control", 0),
                "consensus_strength_pct": insights.get("consensus_strength_pct", 0),
                "target_model": session.get("target_model", ""),
                "chairman_model": session.get("chairman_model", ""),
                "intervention_labels": session.get("intervention_labels", []),
            },
        }
        pairs.append(pair)

    return pairs


def export_preferences_jsonl(
    domain_filter: Optional[str] = None,
) -> str:
    """Export preference pairs as JSONL string (standard DPO format)."""
    pairs = extract_preference_pairs(domain_filter=domain_filter)
    lines = [json.dumps(pair, ensure_ascii=True) for pair in pairs]
    return "\n".join(lines) + ("\n" if lines else "")


# ── Preference Statistics ─────────────────────────────────────────────────────

def get_preference_stats(domain_filter: Optional[str] = None) -> Dict[str, Any]:
    """Compute aggregate statistics about the preference dataset."""
    pairs = extract_preference_pairs(domain_filter=domain_filter)
    total = len(pairs)

    if total == 0:
        return {
            "total_pairs": 0,
            "edited_pairs": 0,
            "accepted_pairs": 0,
            "edit_rate_pct": 0.0,
            "avg_edit_shift_pct": 0.0,
            "domain_distribution": {},
            "avg_quality_rating": 0.0,
            "avg_trust_rating": 0.0,
        }

    edited = [p for p in pairs if p["pair_type"] == "edited"]
    accepted = [p for p in pairs if p["pair_type"] == "accepted"]

    domain_counts: Dict[str, int] = {}
    for pair in pairs:
        d = pair.get("domain", "general")
        domain_counts[d] = domain_counts.get(d, 0) + 1

    def avg_field(field: str) -> float:
        vals = [float(p.get(field, 0) or 0) for p in pairs]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    def avg_meta_field(field: str) -> float:
        vals = [float(p.get("metadata", {}).get(field, 0) or 0) for p in pairs]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    return {
        "total_pairs": total,
        "edited_pairs": len(edited),
        "accepted_pairs": len(accepted),
        "edit_rate_pct": round((len(edited) / total) * 100, 1),
        "avg_edit_shift_pct": avg_field("edit_shift_pct"),
        "domain_distribution": domain_counts,
        "avg_quality_rating": avg_meta_field("quality_rating"),
        "avg_trust_rating": avg_meta_field("trust_rating"),
    }
