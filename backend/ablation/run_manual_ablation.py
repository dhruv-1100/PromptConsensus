"""
Manual ablation runner for ConsensusPrompt.

Edit the placeholder values in STUDY_RUNS and the model constants below,
then execute this file directly.

Example:
    cd backend
    python3 ablation/run_manual_ablation.py
    python3 ablation/run_manual_ablation.py --output-csv ablation/output/my_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import MODELS, TARGET_MODELS  # noqa: E402
from pipeline.graph import build_rewriter_specs_for_models, execute_prompt, run_pipeline  # noqa: E402


DEFAULT_INTENT_MODEL = MODELS["intent_extractor"]
DEFAULT_REVIEWER_MODELS = [
    MODELS["reviewer_a"],
    MODELS["reviewer_b"],
    MODELS["reviewer_c"],
]
DEFAULT_CHAIRMAN_MODEL = MODELS["chairman"]
DEFAULT_EXECUTION_TARGETS = [TARGET_MODELS[0]]
DEFAULT_REPEAT_COUNT = 1


STUDY_RUNS: list[dict[str, Any]] = [
    {
        "run_name": "Model3_ab1",
        "input_domain": "business",
        "raw_query": "I want to start investment in stock markets, how should I start?",
        "rewriter_models": [
            "openai/gpt-5.4-nano",
            "google/gemini-2.5-flash",
            "deepseek/deepseek-v3.2"
        ],
        "repeat": DEFAULT_REPEAT_COUNT,
    },
    {
        "run_name": "Model5_ab1",
        "input_domain": "business",
        "raw_query": "I want to start investment in stock markets, how should I start?",
        "rewriter_models": [
            "openai/gpt-5.4-nano",
            "google/gemini-2.5-flash",
            "deepseek/deepseek-v3.2",
            "qwen/qwen3.6-flash",
            "x-ai/grok-4.3"
        ],
        "repeat": DEFAULT_REPEAT_COUNT,
    },
    {
        "run_name": "Model7_ab1",
        "input_domain": "business",
        "raw_query": "I want to start investment in stock markets, how should I start?",
        "rewriter_models": [
            "openai/gpt-5.4-nano",
            "google/gemini-2.5-flash",
            "deepseek/deepseek-v3.2",
            "qwen/qwen3.6-flash",
            "x-ai/grok-4.3",
            "mistralai/mistral-nemo",
            "meta-llama/llama-3.3-70b-instruct",
        ],
        "repeat": DEFAULT_REPEAT_COUNT,
    },
    {
        "run_name": "Model3_ab2",
        "input_domain": "general",
        "raw_query": "How many stars are there in the sky?",
        "rewriter_models": [
            "openai/gpt-5.4-nano",
            "google/gemini-2.5-flash",
            "deepseek/deepseek-v3.2"
        ],
        "repeat": DEFAULT_REPEAT_COUNT,
    },
    {
        "run_name": "Model5_ab2",
        "input_domain": "general",
        "raw_query": "How many stars are there in the sky?",
        "rewriter_models": [
            "openai/gpt-5.4-nano",
            "google/gemini-2.5-flash",
            "deepseek/deepseek-v3.2",
            "qwen/qwen3.6-flash",
            "x-ai/grok-4.3"
        ],
        "repeat": DEFAULT_REPEAT_COUNT,
    },
    {
        "run_name": "Model7_ab2",
        "input_domain": "general",
        "raw_query": "How many stars are there in the sky?",
        "rewriter_models": [
            "openai/gpt-5.4-nano",
            "google/gemini-2.5-flash",
            "deepseek/deepseek-v3.2",
            "qwen/qwen3.6-flash",
            "x-ai/grok-4.3",
            "mistralai/mistral-nemo",
            "meta-llama/llama-3.3-70b-instruct",
        ],
        "repeat": DEFAULT_REPEAT_COUNT,
    },
    {
        "run_name": "Model3_ab3",
        "input_domain": "technology",
        "raw_query": "How can I make my own claude code using open source models and end point apis?",
        "rewriter_models": [
            "openai/gpt-5.4-nano",
            "google/gemini-2.5-flash",
            "deepseek/deepseek-v3.2"
        ],
        "repeat": DEFAULT_REPEAT_COUNT,
    },
    {
        "run_name": "Model5_ab3",
        "input_domain": "technology",
        "raw_query": "How can I make my own claude code using open source models and end point apis?",
        "rewriter_models": [
            "openai/gpt-5.4-nano",
            "google/gemini-2.5-flash",
            "deepseek/deepseek-v3.2",
            "qwen/qwen3.6-flash",
            "x-ai/grok-4.3"
        ],
        "repeat": DEFAULT_REPEAT_COUNT,
    },
    {
        "run_name": "Model7_ab3",
        "input_domain": "technology",
        "raw_query": "How can I make my own claude code using open source models and end point apis?",
        "rewriter_models": [
            "openai/gpt-5.4-nano",
            "google/gemini-2.5-flash",
            "deepseek/deepseek-v3.2",
            "qwen/qwen3.6-flash",
            "x-ai/grok-4.3",
            "mistralai/mistral-nemo",
            "meta-llama/llama-3.3-70b-instruct",
        ],
        "repeat": DEFAULT_REPEAT_COUNT,
    },
]


FIELDNAMES = [
    "case_id",
    "run_name",
    "repeat_index",
    "raw_query",
    "input_domain",
    "intent_model",
    "rewriter_count",
    "rewriter_models",
    "reviewer_models",
    "chairman_model",
    "target_model",
    "demo_mode",
    "pipeline_elapsed_seconds",
    "execution_elapsed_seconds",
    "status",
    "error",
    "winner_label",
    "winner_candidate",
    "winner_average_rank",
    "winning_rewriter_model",
    "winner_prompt_text",
    "hybrid_prompt",
    "original_query_response",
    "hybrid_prompt_response",
    "intent",
    "candidate_models",
    "rewriter_outputs",
    "reviewer_votes",
    "peer_reviews",
    "aggregate_rankings",
    "label_map",
    "perspectives",
    "chairman_info",
    "consensus_label",
    "consensus_strength_pct",
    "reviewer_agreement_pct",
    "first_place_support_pct",
]


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _validate_existing_header(output_path: Path) -> None:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        existing_header = next(reader, [])

    if existing_header != FIELDNAMES:
        raise RuntimeError(
            "Existing CSV header does not match the current manual ablation schema. "
            f"Use a new output file or migrate the old one first: {output_path}"
        )


def _next_case_number(output_path: Path) -> int:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return 1

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        existing_rows = sum(1 for _ in reader)
    return existing_rows + 1


def _build_candidate_model_map(active_specs: list[dict[str, Any]]) -> dict[str, str]:
    return {spec["candidate_name"]: spec["model_name"] for spec in active_specs}


def _build_reviewer_vote_summary(peer_reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "reviewer": review.get("reviewer", ""),
            "model": review.get("model", ""),
            "ranking": review.get("parsed_ranking", []),
        }
        for review in peer_reviews
    ]


def _resolve_winner_prompt(state: dict[str, Any], winner_label: str) -> str:
    label_map = state.get("label_map") or {}
    all_candidates = state.get("all_candidates") or {}
    winning_candidate_name = label_map.get(winner_label, "")
    return all_candidates.get(winning_candidate_name, "")


def _validate_study_runs(runs: list[dict[str, Any]]) -> None:
    if not runs:
        raise RuntimeError("STUDY_RUNS is empty. Add at least one manual ablation run.")

    seen_names: set[str] = set()
    placeholder_errors: list[str] = []

    for run in runs:
        run_name = str(run.get("run_name", "")).strip()
        raw_query = str(run.get("raw_query", "")).strip()
        input_domain = str(run.get("input_domain", "general")).strip() or "general"
        rewriter_models = [str(item).strip() for item in run.get("rewriter_models", []) if str(item).strip()]

        if not run_name:
            raise RuntimeError("Each manual ablation run must define a non-empty run_name.")
        if run_name in seen_names:
            raise RuntimeError(f"Duplicate run_name in STUDY_RUNS: {run_name}")
        seen_names.add(run_name)

        if not raw_query:
            raise RuntimeError(f"Run '{run_name}' is missing raw_query.")
        if not rewriter_models:
            raise RuntimeError(f"Run '{run_name}' must define at least one rewriter model.")
        if int(run.get("repeat", DEFAULT_REPEAT_COUNT)) < 1:
            raise RuntimeError(f"Run '{run_name}' must have repeat >= 1.")

        if "REPLACE_ME" in raw_query:
            placeholder_errors.append(f"{run_name}: raw_query")
        if "REPLACE_ME" in run_name:
            placeholder_errors.append(f"{run_name}: run_name")
        if "REPLACE_ME" in input_domain:
            placeholder_errors.append(f"{run_name}: input_domain")
        for idx, model_name in enumerate(rewriter_models, start=1):
            if "REPLACE_ME" in model_name:
                placeholder_errors.append(f"{run_name}: rewriter_models[{idx}]")

    if placeholder_errors:
        raise RuntimeError(
            "Replace the manual ablation placeholders before running this script:\n- "
            + "\n- ".join(placeholder_errors)
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run manual ablation studies from in-file placeholder configs.")
    parser.add_argument(
        "--output-csv",
        default=str(BACKEND_ROOT / "ablation" / "output" / "ablation_results.csv"),
        help="CSV path for detailed ablation rows.",
    )
    parser.add_argument(
        "--run-name",
        action="append",
        default=[],
        help="Optional run_name filter. Repeat to execute only selected manual runs.",
    )
    parser.add_argument("--demo-mode", action="store_true", help="Run the pipeline in demo mode.")
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Skip target-model execution and store only the pipeline outputs.",
    )
    return parser


def main() -> None:
    load_dotenv(BACKEND_ROOT / ".env")
    args = _build_arg_parser().parse_args()

    selected_runs = STUDY_RUNS
    if args.run_name:
        requested = set(args.run_name)
        selected_runs = [run for run in STUDY_RUNS if str(run.get("run_name")) in requested]
        missing = requested - {str(run.get("run_name")) for run in selected_runs}
        if missing:
            raise RuntimeError("Unknown run_name values requested: " + ", ".join(sorted(missing)))

    _validate_study_runs(selected_runs)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _validate_existing_header(output_path)
    should_write_header = not output_path.exists() or output_path.stat().st_size == 0
    case_counter = _next_case_number(output_path)

    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if should_write_header:
            writer.writeheader()

        for run in selected_runs:
            run_name = str(run["run_name"]).strip()
            raw_query = str(run["raw_query"]).strip()
            input_domain = str(run.get("input_domain", "general")).strip() or "general"
            repeat_count = int(run.get("repeat", DEFAULT_REPEAT_COUNT))
            rewriter_models = [str(item).strip() for item in run["rewriter_models"]]
            rewriter_specs = build_rewriter_specs_for_models(rewriter_models)
            candidate_model_map = _build_candidate_model_map(rewriter_specs)
            intent_model = str(run.get("intent_model", DEFAULT_INTENT_MODEL)).strip()
            reviewer_models = [
                str(item).strip()
                for item in run.get("reviewer_models", DEFAULT_REVIEWER_MODELS)
                if str(item).strip()
            ]
            chairman_model = str(run.get("chairman_model", DEFAULT_CHAIRMAN_MODEL)).strip()
            target_models = [
                str(item).strip()
                for item in run.get("target_models", DEFAULT_EXECUTION_TARGETS)
                if str(item).strip()
            ]
            do_execute = not args.skip_execution and bool(target_models)

            for repeat_index in range(1, repeat_count + 1):
                pipeline_start = time.perf_counter()
                state: dict[str, Any] = {}
                pipeline_elapsed = 0.0
                status = "ok"
                error = ""

                _log(
                    f"[{run_name}] starting repeat {repeat_index}/{repeat_count} "
                    f"with {len(rewriter_models)} rewriters"
                )

                try:
                    state = run_pipeline(
                        raw_query=raw_query,
                        domain=input_domain,
                        demo_mode=args.demo_mode,
                        intent_model=intent_model,
                        rewriter_specs=rewriter_specs,
                        reviewer_models=reviewer_models,
                        chairman_model=chairman_model,
                        record_analytics=False,
                    )
                    pipeline_elapsed = round(time.perf_counter() - pipeline_start, 4)
                except Exception as exc:
                    pipeline_elapsed = round(time.perf_counter() - pipeline_start, 4)
                    status = "error"
                    error = str(exc)
                    _log(f"[{run_name}] pipeline error on repeat {repeat_index}: {error}")

                winner = (state.get("aggregate_rankings") or [{}])[0] if state else {}
                winner_label = winner.get("label", "")
                winner_candidate_name = (state.get("label_map") or {}).get(winner_label, "") if state else ""
                winner_prompt_text = _resolve_winner_prompt(state, winner_label) if state else ""
                reviewer_vote_summary = _build_reviewer_vote_summary(state.get("peer_reviews", []) if state else [])
                target_models_to_record = target_models if do_execute and status == "ok" else [""]

                for target_model in target_models_to_record:
                    execution_elapsed = 0.0
                    original_query_response = ""
                    hybrid_prompt_response = ""
                    execution_status = status
                    execution_error = error

                    if target_model and status == "ok":
                        execute_start = time.perf_counter()
                        try:
                            original_query_response = execute_prompt(
                                raw_query,
                                target_model=target_model,
                                demo_mode=args.demo_mode,
                            )
                            hybrid_prompt_response = execute_prompt(
                                state.get("optimised_prompt", ""),
                                target_model=target_model,
                                demo_mode=args.demo_mode,
                            )
                            execution_elapsed = round(time.perf_counter() - execute_start, 4)
                        except Exception as exc:
                            execution_elapsed = round(time.perf_counter() - execute_start, 4)
                            execution_status = "error"
                            execution_error = str(exc)
                            _log(
                                f"[{run_name}] execution error on repeat {repeat_index} "
                                f"for target {target_model}: {execution_error}"
                            )

                    case_id = f"case_{case_counter:05d}"
                    case_counter += 1

                    writer.writerow(
                        {
                            "case_id": case_id,
                            "run_name": run_name,
                            "repeat_index": repeat_index,
                            "raw_query": raw_query,
                            "input_domain": input_domain,
                            "intent_model": intent_model,
                            "rewriter_count": len(rewriter_models),
                            "rewriter_models": ",".join(rewriter_models),
                            "reviewer_models": ",".join(reviewer_models),
                            "chairman_model": chairman_model,
                            "target_model": target_model,
                            "demo_mode": args.demo_mode,
                            "pipeline_elapsed_seconds": pipeline_elapsed,
                            "execution_elapsed_seconds": execution_elapsed,
                            "status": execution_status,
                            "error": execution_error,
                            "winner_label": winner_label,
                            "winner_candidate": winner.get("candidate", ""),
                            "winner_average_rank": winner.get("average_rank", ""),
                            "winning_rewriter_model": candidate_model_map.get(winner_candidate_name, ""),
                            "winner_prompt_text": winner_prompt_text,
                            "hybrid_prompt": state.get("optimised_prompt", "") if state else "",
                            "original_query_response": original_query_response,
                            "hybrid_prompt_response": hybrid_prompt_response,
                            "intent": _json_cell(state.get("intent", {}) if state else {}),
                            "candidate_models": _json_cell(candidate_model_map),
                            "rewriter_outputs": _json_cell(state.get("all_candidates", {}) if state else {}),
                            "reviewer_votes": _json_cell(reviewer_vote_summary),
                            "peer_reviews": _json_cell(state.get("peer_reviews", []) if state else []),
                            "aggregate_rankings": _json_cell(state.get("aggregate_rankings", []) if state else []),
                            "label_map": _json_cell(state.get("label_map", {}) if state else {}),
                            "perspectives": _json_cell(state.get("perspectives", {}) if state else {}),
                            "chairman_info": _json_cell(state.get("chairman", {}) if state else {}),
                            "consensus_label": (state.get("consensus_diagnostics") or {}).get("consensus_label", "") if state else "",
                            "consensus_strength_pct": (state.get("consensus_diagnostics") or {}).get("consensus_strength_pct", "") if state else "",
                            "reviewer_agreement_pct": (state.get("consensus_diagnostics") or {}).get("reviewer_agreement_pct", "") if state else "",
                            "first_place_support_pct": (state.get("consensus_diagnostics") or {}).get("first_place_support_pct", "") if state else "",
                        }
                    )

    _log(f"Manual ablation results appended to {output_path}")


if __name__ == "__main__":
    main()
