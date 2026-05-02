"""
feedback_memory.py
Utilities for persisting user feedback and turning high-signal sessions
into lightweight chairman guidance examples.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List


FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), "feedback.json")


def _read_feedback_entries() -> List[Dict[str, Any]]:
    """Load feedback entries from disk, returning an empty list on failure."""
    if not os.path.exists(FEEDBACK_FILE):
        return []

    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def append_feedback_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Persist one feedback entry and return the stored payload."""
    entries = _read_feedback_entries()
    stored = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **entry,
    }
    entries.append(stored)

    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=True)

    return stored


def build_chairman_feedback_context(limit: int = 3, topic_domain: str | None = None) -> str:
    """
    Build a few-shot style memory block from recent high-signal sessions.

    Preference is given to sessions where users actively edited the council
    output or left a positive/high-trust rating, because those are the most
    useful signals for future synthesis.
    """
    entries = _read_feedback_entries()
    if not entries:
        return ""

    scored_entries = []
    normalized_domain = (topic_domain or "").strip().lower()

    for entry in entries:
        optimised_prompt = (entry.get("optimised_prompt") or "").strip()
        final_prompt = (entry.get("final_prompt") or "").strip()
        if not optimised_prompt or not final_prompt:
            continue

        entry_domain = (
            entry.get("domain")
            or (entry.get("intent") or {}).get("topic_domain")
            or "general"
        )
        entry_domain = str(entry_domain).strip().lower()
        if normalized_domain and entry_domain != normalized_domain:
            continue

        quality = int(entry.get("quality", 0) or 0)
        improvement = int(entry.get("improvement", 0) or 0)
        trust = int(entry.get("trust", 0) or 0)
        control = int(entry.get("control", 0) or 0)
        comment = (entry.get("text") or "").strip()
        changed = optimised_prompt != final_prompt

        signal_score = quality + improvement + trust + control
        if changed:
            signal_score += 4
        if comment:
            signal_score += 1

        if signal_score < 12:
            continue

        scored_entries.append((signal_score, entry))

    if not scored_entries:
        return ""

    scored_entries.sort(key=lambda item: item[0], reverse=True)
    selected = [entry for _, entry in scored_entries[:limit]]

    examples = []
    for idx, entry in enumerate(selected, start=1):
        optimised_prompt = entry.get("optimised_prompt", "").strip()
        final_prompt = entry.get("final_prompt", "").strip()
        comment = (entry.get("text") or "").strip()
        changed = optimised_prompt != final_prompt
        preference = "User edited the synthesised prompt before approval." if changed else "User approved the synthesised prompt without edits."

        ratings = (
            f"quality={entry.get('quality', '?')}, "
            f"improvement={entry.get('improvement', '?')}, "
            f"trust={entry.get('trust', '?')}, "
            f"control={entry.get('control', '?')}"
        )

        parts = [
            f"EXAMPLE {idx}",
            f"Original query: {entry.get('raw_query', '')}",
            f"Council output: {optimised_prompt}",
            f"Human-approved prompt: {final_prompt}",
            f"Preference signal: {preference}",
            f"Ratings: {ratings}",
        ]
        if comment:
            parts.append(f"User comment: {comment}")
        examples.append("\n".join(parts))

    scope = f" for the '{normalized_domain}' domain" if normalized_domain else ""
    return (
        f"Human preference memory from prior sessions{scope}. Use these examples to prefer "
        "prompt structures that users trusted, approved, or explicitly edited toward:\n\n"
        + "\n\n".join(examples)
    )
