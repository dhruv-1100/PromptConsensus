"""
Evaluate manual ablation CSV results and write aggregated summaries.

Example:
    cd backend
    python3 ablation/evaluate_ablation.py
    python3 ablation/evaluate_ablation.py --input-csv ablation/output/ablation_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = BACKEND_ROOT / "ablation" / "output" / "ablation_results.csv"
DEFAULT_SUMMARY_CSV = BACKEND_ROOT / "ablation" / "output" / "ablation_summary.csv"
DEFAULT_COUNT_SUMMARY_CSV = BACKEND_ROOT / "ablation" / "output" / "ablation_summary_by_model_count.csv"
DEFAULT_REPORT_JSON = BACKEND_ROOT / "ablation" / "output" / "ablation_report.json"


def _to_float(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: str) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def _safe_mean(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def _top_counter_value(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return counter.most_common(1)[0][0]


def _unique_errors(rows: list[dict[str, str]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        error = str(row.get("error", "")).strip()
        if error and error not in seen:
            seen.append(error)
    return seen


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate manual ablation results.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV), help="Detailed ablation CSV to evaluate.")
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV), help="Output CSV for grouped summary rows.")
    parser.add_argument(
        "--count-summary-csv",
        default=str(DEFAULT_COUNT_SUMMARY_CSV),
        help="Output CSV for statistics combined by rewriter/model count.",
    )
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON), help="Output JSON for the full evaluation report.")
    return parser


def _build_summary_row(
    group_rows: list[dict[str, str]],
    *,
    run_name: str,
    target_model: str,
) -> dict[str, Any]:
    pipeline_times = [value for value in (_to_float(row.get("pipeline_elapsed_seconds", "")) for row in group_rows) if value is not None]
    execution_times = [value for value in (_to_float(row.get("execution_elapsed_seconds", "")) for row in group_rows) if value is not None]
    consensus_strengths = [value for value in (_to_float(row.get("consensus_strength_pct", "")) for row in group_rows) if value is not None]
    reviewer_agreements = [value for value in (_to_float(row.get("reviewer_agreement_pct", "")) for row in group_rows) if value is not None]
    first_place_supports = [value for value in (_to_float(row.get("first_place_support_pct", "")) for row in group_rows) if value is not None]

    success_count = sum(1 for row in group_rows if row.get("status") == "ok")
    failure_count = len(group_rows) - success_count

    winning_model_counts = Counter(
        str(row.get("winning_rewriter_model", "")).strip()
        for row in group_rows
        if str(row.get("winning_rewriter_model", "")).strip()
    )
    winner_candidate_counts = Counter(
        str(row.get("winner_candidate", "")).strip()
        for row in group_rows
        if str(row.get("winner_candidate", "")).strip()
    )
    reviewer_vote_counts = Counter()
    for row in group_rows:
        raw_votes = row.get("reviewer_votes", "")
        if not raw_votes:
            continue
        try:
            votes = json.loads(raw_votes)
        except Exception:
            continue
        if isinstance(votes, list):
            for vote in votes:
                if isinstance(vote, dict):
                    ranking = vote.get("ranking", [])
                    if ranking:
                        reviewer_vote_counts[str(ranking[0])] += 1

    return {
        "run_name": run_name,
        "target_model": target_model,
        "rewriter_count": group_rows[0].get("rewriter_count", ""),
        "row_count": len(group_rows),
        "success_count": success_count,
        "failure_count": failure_count,
        "avg_pipeline_elapsed_seconds": _safe_mean(pipeline_times),
        "avg_execution_elapsed_seconds": _safe_mean(execution_times),
        "avg_consensus_strength_pct": _safe_mean(consensus_strengths),
        "avg_reviewer_agreement_pct": _safe_mean(reviewer_agreements),
        "avg_first_place_support_pct": _safe_mean(first_place_supports),
        "most_frequent_winning_rewriter_model": _top_counter_value(winning_model_counts),
        "most_frequent_winner_candidate": _top_counter_value(winner_candidate_counts),
        "most_common_first_place_vote": _top_counter_value(reviewer_vote_counts),
        "non_empty_original_query_responses": sum(1 for row in group_rows if str(row.get("original_query_response", "")).strip()),
        "non_empty_hybrid_prompt_responses": sum(1 for row in group_rows if str(row.get("hybrid_prompt_response", "")).strip()),
        "sample_errors": " | ".join(_unique_errors(group_rows)[:3]),
        "winning_rewriter_model_counts": dict(winning_model_counts),
        "winner_candidate_counts": dict(winner_candidate_counts),
        "first_place_vote_counts": dict(reviewer_vote_counts),
        "unique_errors": _unique_errors(group_rows),
    }


def main() -> None:
    args = _build_arg_parser().parse_args()
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise RuntimeError(f"Ablation results CSV not found: {input_path}")

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError(f"Ablation results CSV is empty: {input_path}")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("run_name", ""), row.get("target_model", ""))
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    report_runs: list[dict[str, Any]] = []

    for (run_name, target_model), group_rows in sorted(grouped.items()):
        summary_row = _build_summary_row(group_rows, run_name=run_name, target_model=target_model)
        summary_rows.append(summary_row)

        report_runs.append(
            {
                **summary_row,
            }
        )

    count_grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("rewriter_count", ""), row.get("target_model", ""))
        count_grouped[key].append(row)

    count_summary_rows: list[dict[str, Any]] = []
    report_counts: list[dict[str, Any]] = []
    for (rewriter_count, target_model), group_rows in sorted(
        count_grouped.items(),
        key=lambda item: (_to_int(item[0][0]), item[0][1]),
    ):
        summary_row = _build_summary_row(
            group_rows,
            run_name=f"combined_{rewriter_count}_models",
            target_model=target_model,
        )
        summary_row["combined_run_count"] = len({row.get("run_name", "") for row in group_rows})
        summary_row["source_run_names"] = ",".join(sorted({row.get("run_name", "") for row in group_rows if row.get("run_name", "")}))
        count_summary_rows.append(summary_row)
        report_counts.append(dict(summary_row))

    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[key for key in summary_rows[0].keys() if key not in {
                "winning_rewriter_model_counts",
                "winner_candidate_counts",
                "first_place_vote_counts",
                "unique_errors",
            }],
        )
        writer.writeheader()
        writer.writerows(
            {
                key: value
                for key, value in row.items()
                if key not in {
                    "winning_rewriter_model_counts",
                    "winner_candidate_counts",
                    "first_place_vote_counts",
                    "unique_errors",
                }
            }
            for row in summary_rows
        )

    count_summary_path = Path(args.count_summary_csv)
    count_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with count_summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[key for key in count_summary_rows[0].keys() if key not in {
                "winning_rewriter_model_counts",
                "winner_candidate_counts",
                "first_place_vote_counts",
                "unique_errors",
            }],
        )
        writer.writeheader()
        writer.writerows(
            {
                key: value
                for key, value in row.items()
                if key not in {
                    "winning_rewriter_model_counts",
                    "winner_candidate_counts",
                    "first_place_vote_counts",
                    "unique_errors",
                }
            }
            for row in count_summary_rows
        )

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "input_csv": str(input_path),
        "summary_csv": str(summary_path),
        "count_summary_csv": str(count_summary_path),
        "total_rows": len(rows),
        "total_groups": len(summary_rows),
        "total_count_groups": len(count_summary_rows),
        "groups": report_runs,
        "groups_by_model_count": report_counts,
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)

    print(f"Summary CSV written to {summary_path}")
    print(f"Model-count summary CSV written to {count_summary_path}")
    print(f"JSON report written to {report_path}")


if __name__ == "__main__":
    main()
