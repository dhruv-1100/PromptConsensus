"""
live_mode_utils.py
Helpers for clearer live-mode failures and more robust JSON extraction from LLMs.
"""
from __future__ import annotations

import ast
import datetime
import json
import os
import re
from typing import Any, Sequence


PARSE_FAILURE_LOG = os.path.join(os.path.dirname(__file__), "structured_parse_failures.json")


def require_openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured. Add it to backend/.env before running non-demo mode."
        )
    return api_key


def coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
                elif item.get("type") in {"output_text", "text"} and isinstance(item.get("content"), str):
                    parts.append(item["content"])
                else:
                    for key in ("content", "value", "reasoning"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            parts.append(value)
                            break
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()

    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return json.dumps(content, ensure_ascii=True)

    return str(content or "").strip()


def parse_json_response(content: str) -> Any:
    text = (content or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) > 1:
            text = parts[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    candidates: list[str] = [text]

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        cleaned = candidate.strip()
        if not cleaned:
            continue

        try:
            return json.loads(cleaned)
        except Exception:
            pass

        repaired = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        repaired = repaired.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
        try:
            return json.loads(repaired)
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(repaired)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            pass

    raise ValueError("Model returned invalid JSON for a structured pipeline step.")


def log_structured_parse_failure(step: str, model_name: str, content: str, error: str) -> None:
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "step": step,
        "model": model_name,
        "error": error,
        "content_preview": (content or "")[:4000],
        "content_length": len(content or ""),
    }
    try:
        logs = []
        if os.path.exists(PARSE_FAILURE_LOG):
            with open(PARSE_FAILURE_LOG, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    logs = loaded
        logs.append(entry)
        with open(PARSE_FAILURE_LOG, "w", encoding="utf-8") as f:
            json.dump(logs[-50:], f, indent=2, ensure_ascii=True)
    except Exception:
        pass


def extract_prompt_and_perspective(
    content: str,
    default_perspective: str,
    *,
    model_name: str = "",
    step: str = "",
) -> tuple[str, str]:
    try:
        data = parse_json_response(content)
        if isinstance(data, dict):
            prompt = str(data.get("optimised_prompt") or content).strip()
            perspective = str(data.get("perspective_used") or "").strip() or default_perspective
            return prompt, perspective
    except Exception as exc:
        log_structured_parse_failure(step or "rewriter", model_name or "unknown", content, str(exc))

    cleaned = (content or "").strip()
    cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).replace("```", "").strip()
    if cleaned:
        return cleaned, default_perspective
    return content, default_perspective


def invoke_openrouter_model(
    messages: Sequence[Any],
    model_name: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> tuple[str, str]:
    from langchain_openai import ChatOpenAI

    api_key = require_openrouter_api_key()
    resolved_model = (model_name or "").strip()
    if not resolved_model:
        raise RuntimeError("No OpenRouter model was configured for this pipeline step.")

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=120,
    )
    try:
        response = llm.invoke(list(messages))
        content = coerce_message_content(response.content)
        if not content.strip():
            additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
            content = coerce_message_content(additional_kwargs)
        return content.strip(), resolved_model
    except Exception as exc:
        raise RuntimeError(f"OpenRouter call failed for configured model '{resolved_model}': {exc}") from exc
