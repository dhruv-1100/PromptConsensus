"""
CLI entrypoint for non-UI ablation studies over the ConsensusPrompt pipeline.

python3 ablation_study.py \
  --raw-query "I want to start investment in stock markets, how should I start?" \
  --input-domain business \
  --execute \
  --output-csv ./ablation_results.csv \
  --intent-models "openai/gpt-5.4-nano" \
  --rewriter-model-set "google/gemini-2.5-flash,openai/gpt-5.4-nano,deepseek/deepseek-v3.2,qwen/qwen3.6-flash,x-ai/grok-4.3" \
  --reviewer-model-set "google/gemini-2.5-flash,openai/gpt-5.4-nano,deepseek/deepseek-v3.2" \
  --chairman-models "nvidia/nemotron-3-super-120b-a12b"

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

from config import MODELS, TARGET_MODELS
from pipeline.graph import build_rewriter_specs_for_models, execute_prompt, run_pipeline


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_queries(raw_queries: list[str], queries_file: str | None, default_domain: str) -> list[dict[str, str]]:
    queries = [{"raw_query": query.strip(), "domain": default_domain} for query in raw_queries if query.strip()]
    if not queries_file:
        return queries

    path = Path(queries_file)
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_query = (row.get("raw_query") or row.get("query") or "").strip()
                if not raw_query:
                    continue
                queries.append({
                    "raw_query": raw_query,
                    "domain": (row.get("domain") or default_domain).strip() or default_domain,
                })
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw_query = line.strip()
                if raw_query:
                    queries.append({"raw_query": raw_query, "domain": default_domain})

    return queries


def _build_rewriter_spec_set(model_set: list[str]) -> list[dict[str, Any]]:
    if not model_set:
        raise ValueError("Each --rewriter-model-set must define at least one model.")
    return build_rewriter_specs_for_models(model_set)


def _next_case_number(output_path: Path) -> int:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return 1

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        existing_rows = sum(1 for _ in reader)
    return existing_rows + 1


def _validate_existing_header(output_path: Path, expected_fieldnames: list[str]) -> None:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        existing_header = next(reader, [])

    if existing_header != expected_fieldnames:
        raise RuntimeError(
            "Existing CSV header does not match the current ablation output schema. "
            f"Use a new output file or migrate the existing CSV first: {output_path}"
        )


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def _print_run_message(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _build_candidate_model_map(active_specs: list[dict[str, Any]]) -> dict[str, str]:
    return {
        spec["candidate_name"]: spec["model_name"]
        for spec in active_specs
    }


def _build_reviewer_vote_summary(peer_reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for review in peer_reviews:
        summary.append({
            "reviewer": review.get("reviewer", ""),
            "model": review.get("model", ""),
            "ranking": review.get("parsed_ranking", []),
        })
    return summary


def _resolve_winner_prompt(state: dict[str, Any], winner_label: str) -> str:
    label_map = state.get("label_map") or {}
    all_candidates = state.get("all_candidates") or {}
    winning_candidate_name = label_map.get(winner_label, "")
    return all_candidates.get(winning_candidate_name, "")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ConsensusPrompt ablation studies without the frontend.")
    parser.add_argument("--raw-query", action="append", default=[], help="Raw query to evaluate. Repeat for multiple queries.")
    parser.add_argument("--queries-file", help="Optional .txt or .csv file containing queries. CSV should contain raw_query and optional domain columns.")
    parser.add_argument(
        "--domain",
        "--input-domain",
        dest="domain",
        default="general",
        help="Input domain for the run, for example general, research, finance, healthcare, or education. This is the default domain for queries that do not specify one.",
    )
    parser.add_argument("--output-csv", required=True, help="Path to the CSV file that will store ablation results.")
    parser.add_argument("--demo-mode", action="store_true", help="Run the study in demo mode.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat each ablation case this many times.")
    parser.add_argument("--execute", action="store_true", help="Also execute the synthesized prompt against target models.")
    parser.add_argument("--intent-models", default=MODELS["intent_extractor"], help="Comma-separated list of intent models to sweep.")
    parser.add_argument(
        "--rewriter-model-set",
        action="append",
        dest="rewriter_model_sets",
        help="Comma-separated ordered models for active rewriters. Additional models reuse the existing A/B/C prompt families cyclically.",
    )
    parser.add_argument(
        "--reviewer-model-set",
        action="append",
        dest="reviewer_model_sets",
        help="Comma-separated reviewer model list. Repeat to sweep multiple reviewer sets.",
    )
    parser.add_argument("--chairman-models", default=MODELS["chairman"], help="Comma-separated chairman models to sweep.")
    parser.add_argument(
        "--target-models",
        default="",
        help="Comma-separated target models used only with --execute. Defaults to configured target models.",
    )
    return parser


def main() -> None:
    load_dotenv()
    args = _build_arg_parser().parse_args()

    queries = _load_queries(args.raw_query, args.queries_file, args.domain)
    if not queries:
        raise RuntimeError("Provide at least one query via --raw-query or --queries-file.")
    if args.repeat < 1:
        raise RuntimeError("--repeat must be at least 1.")

    intent_models = _parse_csv_list(args.intent_models)
    chairman_models = _parse_csv_list(args.chairman_models)

    rewriter_model_sets = args.rewriter_model_sets or [
        ",".join([MODELS["rewriter_a"], MODELS["rewriter_b"], MODELS["rewriter_c"]])
    ]
    rewriter_spec_sets = [_build_rewriter_spec_set(_parse_csv_list(model_set)) for model_set in rewriter_model_sets]

    reviewer_model_sets = args.reviewer_model_sets or [
        ",".join([MODELS["reviewer_a"], MODELS["reviewer_b"], MODELS["reviewer_c"]])
    ]
    parsed_reviewer_sets = [_parse_csv_list(model_set) for model_set in reviewer_model_sets]

    target_models = _parse_csv_list(args.target_models) if args.target_models else list(TARGET_MODELS)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "run_index",
        "raw_query",
        "domain",
        "intent_model",
        "rewriter_count",
        "rewriter_strategy_subset",
        "rewriter_models_all",
        "rewriter_models_active",
        "reviewer_models",
        "chairman_model",
        "target_model",
        "demo_mode",
        "optimize_elapsed_seconds",
        "execute_elapsed_seconds",
        "status",
        "error",
        "winner_label",
        "winner_candidate",
        "winner_average_rank",
        "consensus_label",
        "consensus_strength_pct",
        "reviewer_agreement_pct",
        "first_place_support_pct",
        "winning_rewriter_model",
        "winner_prompt_text",
        "hybrid_prompt",
        "optimised_prompt",
        "original_query_response",
        "hybrid_prompt_response",
        "llm_response",
        "intent",
        "candidate_models",
        "rewriter_outputs",
        "candidates",
        "perspectives",
        "reviewer_votes",
        "peer_reviews",
        "aggregate_rankings",
        "label_map",
        "chairman_info",
    ]

    _validate_existing_header(output_path, fieldnames)
    should_write_header = not output_path.exists() or output_path.stat().st_size == 0
    case_counter = _next_case_number(output_path)

    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()

        for query in queries:
            for intent_model in intent_models:
                for rewriter_specs in rewriter_spec_sets:
                    active_specs = rewriter_specs
                    for reviewer_models in parsed_reviewer_sets:
                        for chairman_model in chairman_models:
                            execution_models = target_models if args.execute else [""]
                            for target_model in execution_models:
                                for run_index in range(1, args.repeat + 1):
                                    case_id = f"case_{case_counter:05d}"
                                    case_counter += 1
                                    optimize_start = time.perf_counter()
                                    execute_elapsed = 0.0
                                    original_query_response = ""
                                    hybrid_prompt_response = ""
                                    llm_response = ""
                                    error = ""
                                    status = "ok"
                                    state: dict[str, Any] = {}
                                    active_candidate_names = [spec["candidate_name"] for spec in active_specs]
                                    active_candidate_models = [spec["model_name"] for spec in active_specs]
                                    candidate_model_map = _build_candidate_model_map(active_specs)

                                    try:
                                        state = run_pipeline(
                                            raw_query=query["raw_query"],
                                            domain=query["domain"],
                                            demo_mode=args.demo_mode,
                                            intent_model=intent_model,
                                            rewriter_specs=active_specs,
                                            reviewer_models=reviewer_models,
                                            chairman_model=chairman_model,
                                            record_analytics=False,
                                        )
                                        optimize_elapsed = round(time.perf_counter() - optimize_start, 4)

                                        if args.execute and target_model:
                                            execute_start = time.perf_counter()
                                            original_query_response = execute_prompt(
                                                query["raw_query"],
                                                target_model=target_model,
                                                demo_mode=args.demo_mode,
                                            )
                                            hybrid_prompt_response = execute_prompt(
                                                state.get("optimised_prompt", ""),
                                                target_model=target_model,
                                                demo_mode=args.demo_mode,
                                            )
                                            llm_response = hybrid_prompt_response
                                            execute_elapsed = round(time.perf_counter() - execute_start, 4)
                                    except Exception as exc:
                                        optimize_elapsed = round(time.perf_counter() - optimize_start, 4)
                                        status = "error"
                                        error = str(exc)
                                        _print_run_message(
                                            f"[{case_id}] pipeline error | "
                                            f"domain={query['domain']} | "
                                            f"intent_model={intent_model} | "
                                            f"rewriters={','.join(active_candidate_models)} | "
                                            f"reviewers={','.join(reviewer_models)} | "
                                            f"chairman={chairman_model} | "
                                            f"target={target_model or 'none'} | "
                                            f"error={error}"
                                        )

                                    winner = (state.get("aggregate_rankings") or [{}])[0] if state else {}
                                    winner_label = winner.get("label", "")
                                    winner_candidate_name = (state.get("label_map") or {}).get(winner_label, "") if state else ""
                                    winner_prompt_text = _resolve_winner_prompt(state, winner_label) if state else ""
                                    reviewer_vote_summary = _build_reviewer_vote_summary(state.get("peer_reviews", []) if state else [])

                                    writer.writerow({
                                        "case_id": case_id,
                                        "run_index": run_index,
                                        "raw_query": query["raw_query"],
                                        "domain": query["domain"],
                                        "intent_model": intent_model,
                                        "rewriter_count": len(active_specs),
                                        "rewriter_strategy_subset": ",".join(active_candidate_names),
                                        "rewriter_models_all": ",".join(active_candidate_models),
                                        "rewriter_models_active": ",".join(active_candidate_models),
                                        "reviewer_models": ",".join(reviewer_models),
                                        "chairman_model": chairman_model,
                                        "target_model": target_model,
                                        "demo_mode": args.demo_mode,
                                        "optimize_elapsed_seconds": optimize_elapsed,
                                        "execute_elapsed_seconds": execute_elapsed,
                                        "status": status,
                                        "error": error,
                                        "winner_label": winner_label,
                                        "winner_candidate": winner.get("candidate", ""),
                                        "winner_average_rank": winner.get("average_rank", ""),
                                        "consensus_label": (state.get("consensus_diagnostics") or {}).get("consensus_label", "") if state else "",
                                        "consensus_strength_pct": (state.get("consensus_diagnostics") or {}).get("consensus_strength_pct", "") if state else "",
                                        "reviewer_agreement_pct": (state.get("consensus_diagnostics") or {}).get("reviewer_agreement_pct", "") if state else "",
                                        "first_place_support_pct": (state.get("consensus_diagnostics") or {}).get("first_place_support_pct", "") if state else "",
                                        "winning_rewriter_model": candidate_model_map.get(winner_candidate_name, ""),
                                        "winner_prompt_text": winner_prompt_text,
                                        "hybrid_prompt": state.get("optimised_prompt", "") if state else "",
                                        "optimised_prompt": state.get("optimised_prompt", "") if state else "",
                                        "original_query_response": original_query_response,
                                        "hybrid_prompt_response": hybrid_prompt_response,
                                        "llm_response": llm_response,
                                        "intent": _json_cell(state.get("intent", {}) if state else {}),
                                        "candidate_models": _json_cell(candidate_model_map),
                                        "rewriter_outputs": _json_cell(state.get("all_candidates", {}) if state else {}),
                                        "candidates": _json_cell(state.get("all_candidates", {}) if state else {}),
                                        "perspectives": _json_cell(state.get("perspectives", {}) if state else {}),
                                        "reviewer_votes": _json_cell(reviewer_vote_summary),
                                        "peer_reviews": _json_cell(state.get("peer_reviews", []) if state else []),
                                        "aggregate_rankings": _json_cell(state.get("aggregate_rankings", []) if state else []),
                                        "label_map": _json_cell(state.get("label_map", {}) if state else {}),
                                        "chairman_info": _json_cell(state.get("chairman", {}) if state else {}),
                                    })


if __name__ == "__main__":
    main()
