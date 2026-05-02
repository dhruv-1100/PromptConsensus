"""
live_mode_utils.py
Helpers for clearer live-mode failures and more robust JSON extraction from LLMs.
"""
from __future__ import annotations

import json
import os
from typing import Any, Sequence


def require_openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured. Add it to backend/.env before running non-demo mode."
        )
    return api_key


def parse_json_response(content: str) -> Any:
    text = (content or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) > 1:
            text = parts[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError("Model returned invalid JSON for a structured pipeline step.")


def _is_retryable_openrouter_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in [
            "429",
            "rate-limit",
            "rate limit",
            "temporarily rate-limited",
            "provider returned error",
        ]
    )


def build_model_fallback_chain(preferred_model: str, *, allow_router: bool = False) -> list[str]:
    from config import TARGET_MODELS

    ordered = [preferred_model, *TARGET_MODELS]
    seen: set[str] = set()
    chain: list[str] = []
    for model in ordered:
        cleaned = (model or "").strip()
        if cleaned == "openrouter/free" and not allow_router and cleaned != preferred_model:
            continue
        if cleaned and cleaned not in seen:
            chain.append(cleaned)
            seen.add(cleaned)
    return chain


def invoke_openrouter_with_fallback(
    messages: Sequence[Any],
    preferred_model: str,
    *,
    allow_router: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 1000,
) -> tuple[str, str]:
    from langchain_openai import ChatOpenAI

    api_key = require_openrouter_api_key()
    errors: list[str] = []

    for model_name in build_model_fallback_chain(preferred_model, allow_router=allow_router):
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            response = llm.invoke(list(messages))
            return response.content.strip(), model_name
        except Exception as exc:
            if _is_retryable_openrouter_error(exc):
                errors.append(f"{model_name}: {exc}")
                continue
            raise

    raise RuntimeError(
        "All configured free OpenRouter models failed. "
        + " | ".join(errors[-4:])
    )
