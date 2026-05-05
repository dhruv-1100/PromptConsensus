"""
Generate research-style ablation figures from the evaluator outputs.

This script reads the summary CSVs produced by evaluate_ablation.py and writes
publication-style PNG/PDF figures. It does not generate HTML.

Example:
    cd backend
    python3 ablation/visualize_ablation.py
    python3 ablation/visualize_ablation.py \
      --summary-csv ablation/output/ablation_summary.csv \
      --count-summary-csv ablation/output/ablation_summary_by_model_count.csv \
      --output-dir ablation/output/figures
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_CSV = BACKEND_ROOT / "ablation" / "output" / "ablation_summary.csv"
DEFAULT_COUNT_SUMMARY_CSV = BACKEND_ROOT / "ablation" / "output" / "ablation_summary_by_model_count.csv"
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "ablation" / "output" / "figures"


def _to_float(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _to_int(value: Any) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _configure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for figure generation. Install it in the backend environment first."
        ) from exc

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })
    return plt


def _save_figure(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def _plot_per_run_summary(plt, summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    if not summary_rows:
        return

    sorted_rows = sorted(
        summary_rows,
        key=lambda row: (_to_int(row.get("rewriter_count", 0)), row.get("run_name", ""), row.get("target_model", "")),
    )
    labels = [
        f"{row.get('run_name', '')}\n[{row.get('target_model', '') or 'no target'}]"
        for row in sorted_rows
    ]
    consensus = [_to_float(row.get("avg_consensus_strength_pct", 0)) for row in sorted_rows]
    agreement = [_to_float(row.get("avg_reviewer_agreement_pct", 0)) for row in sorted_rows]
    first_place = [_to_float(row.get("avg_first_place_support_pct", 0)) for row in sorted_rows]

    x_positions = list(range(len(sorted_rows)))
    width = 0.24

    fig, ax = plt.subplots(figsize=(max(10, len(sorted_rows) * 1.2), 5.5))
    ax.bar([x - width for x in x_positions], consensus, width=width, color="#2878B5", label="Consensus strength")
    ax.bar(x_positions, agreement, width=width, color="#4DAA57", label="Reviewer agreement")
    ax.bar([x + width for x in x_positions], first_place, width=width, color="#C97C1A", label="First-place support")

    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage")
    ax.set_title("Per-Run Council Metrics")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))

    _save_figure(fig, output_dir, "per_run_council_metrics")
    plt.close(fig)


def _plot_combined_by_model_count(plt, count_rows: list[dict[str, str]], output_dir: Path) -> None:
    if not count_rows:
        return

    sorted_rows = sorted(
        count_rows,
        key=lambda row: (_to_int(row.get("rewriter_count", 0)), row.get("target_model", "")),
    )
    labels = [str(row.get("rewriter_count", "")) for row in sorted_rows]
    x_positions = list(range(len(sorted_rows)))

    consensus = [_to_float(row.get("avg_consensus_strength_pct", 0)) for row in sorted_rows]
    agreement = [_to_float(row.get("avg_reviewer_agreement_pct", 0)) for row in sorted_rows]
    pipeline_time = [_to_float(row.get("avg_pipeline_elapsed_seconds", 0)) for row in sorted_rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=False)

    axes[0].plot(x_positions, consensus, marker="o", linewidth=2.2, color="#2878B5")
    axes[0].set_title("Consensus Strength by Model Count")
    axes[0].set_ylabel("Percentage")
    axes[0].set_ylim(0, 100)
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("Number of rewriter models")

    axes[1].plot(x_positions, agreement, marker="o", linewidth=2.2, color="#4DAA57")
    axes[1].set_title("Reviewer Agreement by Model Count")
    axes[1].set_ylabel("Percentage")
    axes[1].set_ylim(0, 100)
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("Number of rewriter models")

    axes[2].plot(x_positions, pipeline_time, marker="o", linewidth=2.2, color="#C97C1A")
    axes[2].set_title("Pipeline Time by Model Count")
    axes[2].set_ylabel("Seconds")
    axes[2].set_xticks(x_positions)
    axes[2].set_xticklabels(labels)
    axes[2].set_xlabel("Number of rewriter models")
    if pipeline_time:
        y_min = min(pipeline_time)
        y_max = max(pipeline_time)
        if y_min == y_max:
            padding = max(0.5, y_max * 0.1 if y_max else 1.0)
        else:
            padding = max(0.25, (y_max - y_min) * 0.15)
        axes[2].set_ylim(max(0, y_min - padding), y_max + padding)

    _save_figure(fig, output_dir, "combined_by_model_count")
    plt.close(fig)


def _plot_success_failure_by_model_count(plt, count_rows: list[dict[str, str]], output_dir: Path) -> None:
    if not count_rows:
        return

    sorted_rows = sorted(
        count_rows,
        key=lambda row: (_to_int(row.get("rewriter_count", 0)), row.get("target_model", "")),
    )
    labels = [str(row.get("rewriter_count", "")) for row in sorted_rows]
    x_positions = list(range(len(sorted_rows)))
    success = [_to_int(row.get("success_count", 0)) for row in sorted_rows]
    failure = [_to_int(row.get("failure_count", 0)) for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x_positions, success, color="#4DAA57", label="Success")
    ax.bar(x_positions, failure, bottom=success, color="#C44E52", label="Failure")
    ax.set_title("Success / Failure Counts by Model Count")
    ax.set_ylabel("Rows")
    ax.set_xlabel("Number of rewriter models")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")

    _save_figure(fig, output_dir, "success_failure_by_model_count")
    plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate research-style ablation figures.")
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV), help="Per-run summary CSV from evaluate_ablation.py")
    parser.add_argument("--count-summary-csv", default=str(DEFAULT_COUNT_SUMMARY_CSV), help="Combined-by-model-count summary CSV from evaluate_ablation.py")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for PNG/PDF figure outputs")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary_path = Path(args.summary_csv)
    count_summary_path = Path(args.count_summary_csv)
    output_dir = Path(args.output_dir)

    if not summary_path.exists():
        raise RuntimeError(f"Summary CSV not found: {summary_path}")
    if not count_summary_path.exists():
        raise RuntimeError(f"Count summary CSV not found: {count_summary_path}")

    summary_rows = _load_csv(summary_path)
    count_summary_rows = _load_csv(count_summary_path)
    if not summary_rows:
        raise RuntimeError(f"Summary CSV is empty: {summary_path}")
    if not count_summary_rows:
        raise RuntimeError(f"Count summary CSV is empty: {count_summary_path}")

    plt = _configure_matplotlib()
    _plot_per_run_summary(plt, summary_rows, output_dir)
    _plot_combined_by_model_count(plt, count_summary_rows, output_dir)
    _plot_success_failure_by_model_count(plt, count_summary_rows, output_dir)

    print(f"Figures written to {output_dir}")


if __name__ == "__main__":
    main()
