"""
safety_checks.py
Lightweight heuristic safety checks for sensitive execution domains.
"""
from __future__ import annotations

from typing import Any, Dict, List


SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3}
SENSITIVE_DOMAINS = {"healthcare", "legal", "research", "education"}
MEDICATION_TERMS = (
    "medication",
    "medications",
    "dose",
    "dosage",
    "insulin",
    "drug",
    "prescription",
    "treatment",
    "mg",
    "ml",
)


def _contains_any(text: str, phrases: List[str] | tuple[str, ...]) -> bool:
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in phrases)


def _add_check(
    checks: List[Dict[str, str]],
    severity: str,
    title: str,
    detail: str,
    action: str,
) -> None:
    checks.append(
        {
            "severity": severity,
            "title": title,
            "detail": detail,
            "action": action,
        }
    )


def _highest_severity(checks: List[Dict[str, str]]) -> str:
    if not checks:
        return "none"
    return max(checks, key=lambda check: SEVERITY_ORDER.get(check["severity"], 0))["severity"]


def run_safety_checks(
    final_prompt: str,
    domain: str = "general",
    raw_query: str = "",
    intent: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Return heuristic safety findings for the prompt before execution.
    """
    intent = intent or {}
    checks: List[Dict[str, str]] = []
    normalized_domain = (domain or "general").strip().lower() or "general"
    prompt_lower = final_prompt.lower()
    missing_info = intent.get("missing_info") or []

    if len(final_prompt.split()) < 40:
        _add_check(
            checks,
            "low",
            "Prompt is still quite short",
            "Short prompts tend to leave more room for the target model to infer missing constraints on its own.",
            "Add desired format, audience, and output boundaries if you want tighter control.",
        )

    if not _contains_any(
        prompt_lower,
        ["format", "section", "sections", "table", "json", "bullet", "heading", "step", "template"],
    ):
        _add_check(
            checks,
            "low",
            "Output structure is underspecified",
            "The prompt does not strongly constrain the response format, which can make outputs less predictable.",
            "Specify sections, tables, bullets, or another explicit response shape.",
        )

    if missing_info:
        _add_check(
            checks,
            "low",
            "Upstream intent analysis found unresolved gaps",
            "The intent extractor flagged missing information that may still affect response quality.",
            f"Review these gaps before execution: {', '.join(str(item) for item in missing_info[:3])}.",
        )

    if normalized_domain == "healthcare":
        if not _contains_any(
            prompt_lower,
            ["not medical advice", "clinical judgment", "licensed clinician", "doctor", "provider", "for educational purposes"],
        ):
            _add_check(
                checks,
                "medium",
                "Clinical oversight is not made explicit",
                "Healthcare prompts are safer when they instruct the model not to replace clinician judgment or professional review.",
                "Add a line requiring clinician review or clarifying that the output should not be treated as medical advice.",
            )

        if not _contains_any(
            prompt_lower,
            ["follow-up", "follow up", "red flag", "seek urgent", "seek immediate", "emergency", "return precautions"],
        ):
            _add_check(
                checks,
                "medium",
                "Escalation and follow-up guidance is missing",
                "Patient-facing or clinical-support outputs should usually include when to escalate, re-evaluate, or seek urgent care.",
                "Add explicit follow-up instructions or red-flag escalation criteria where appropriate.",
            )

        if _contains_any(prompt_lower, MEDICATION_TERMS) and not _contains_any(
            prompt_lower,
            ["do not fabricate", "if unknown", "use only provided", "verify dosage", "verify dose", "mark unknown"],
        ):
            _add_check(
                checks,
                "high",
                "Medication content lacks verification guardrails",
                "The prompt references medication-related content without telling the model how to handle unknown or unverified doses.",
                "Add a rule to use only provided medication facts and to mark unknown values instead of inventing them.",
            )

    elif normalized_domain == "legal":
        if not _contains_any(prompt_lower, ["jurisdiction", "state law", "federal law", "country", "governing law"]):
            _add_check(
                checks,
                "high",
                "Jurisdiction is missing",
                "Legal outputs can be misleading if the governing jurisdiction is not specified.",
                "State the jurisdiction or ask the model to note when the answer depends on local law.",
            )

        if not _contains_any(prompt_lower, ["not legal advice", "licensed attorney", "legal counsel", "consult a lawyer"]):
            _add_check(
                checks,
                "medium",
                "No legal-advice disclaimer",
                "The prompt does not tell the model to avoid presenting the output as a substitute for an attorney.",
                "Add a brief disclaimer or require the model to recommend attorney review for consequential decisions.",
            )

        if not _contains_any(prompt_lower, ["assumption", "if facts are missing", "ask follow-up", "insufficient facts"]):
            _add_check(
                checks,
                "medium",
                "Missing-facts handling is unclear",
                "Legal reasoning is sensitive to missing facts, but the prompt does not instruct the model how to behave when the record is incomplete.",
                "Tell the model to surface assumptions and identify which missing facts could change the answer.",
            )

    elif normalized_domain == "research":
        if not _contains_any(prompt_lower, ["cite", "citation", "source", "reference", "references"]):
            _add_check(
                checks,
                "medium",
                "Evidence requirements are missing",
                "Research-oriented outputs are more reliable when the prompt explicitly asks for citations or evidence references.",
                "Ask for citations, sources, or clearly labeled evidence summaries.",
            )

        if not _contains_any(prompt_lower, ["limitation", "uncertainty", "confidence", "threat", "caveat"]):
            _add_check(
                checks,
                "medium",
                "Uncertainty handling is missing",
                "The prompt does not require the model to distinguish strong evidence from uncertainty or limitations.",
                "Add a section for limitations, confidence, or unresolved questions.",
            )

        if not _contains_any(prompt_lower, ["method", "dataset", "sample", "study design"]):
            _add_check(
                checks,
                "low",
                "Method context may be too thin",
                "Research summaries are easier to trust when they explicitly mention methods, sample, or study design.",
                "If relevant, ask the model to include methodology and dataset details.",
            )

    elif normalized_domain == "education":
        if not _contains_any(prompt_lower, ["grade", "reading level", "age", "year-old", "student level"]):
            _add_check(
                checks,
                "medium",
                "Learner level is not explicit",
                "Educational outputs are safer and more useful when they are calibrated to a specific learner level.",
                "Specify grade band, age range, or reading level.",
            )

        if not _contains_any(prompt_lower, ["objective", "learning outcome", "assessment", "success criteria"]):
            _add_check(
                checks,
                "low",
                "Learning goals are underspecified",
                "The prompt does not make the learning objective or assessment criteria explicit.",
                "Add learning objectives or expected outcomes if you want a more targeted result.",
            )

    risk_level = _highest_severity(checks)
    requires_acknowledgement = risk_level == "high" or (
        normalized_domain in {"healthcare", "legal"} and any(check["severity"] == "medium" for check in checks)
    )

    return {
        "domain": normalized_domain,
        "sensitive_domain": normalized_domain in SENSITIVE_DOMAINS,
        "risk_level": risk_level,
        "requires_acknowledgement": requires_acknowledgement,
        "checks": checks,
        "checked_prompt_length": len(final_prompt.split()),
        "raw_query_length": len(raw_query.split()) if raw_query else 0,
    }
