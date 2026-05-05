# Ablation Study Tools

This folder contains the standalone ablation workflow for the backend pipeline.

Files:
- `run_manual_ablation.py`: runs manual ablation experiments from in-file placeholder configs.
- `evaluate_ablation.py`: aggregates the detailed CSV output into a summary CSV and JSON report.
- `visualize_ablation.py`: generates research-style PNG/PDF figures from the evaluation outputs.
- `output/`: default location for generated ablation artifacts.

## Manual Setup

Edit the placeholders near the top of `run_manual_ablation.py`:
- `STUDY_RUNS[*].run_name`
- `STUDY_RUNS[*].raw_query`
- `STUDY_RUNS[*].rewriter_models`

Optional global defaults:
- `DEFAULT_INTENT_MODEL`
- `DEFAULT_REVIEWER_MODELS`
- `DEFAULT_CHAIRMAN_MODEL`
- `DEFAULT_EXECUTION_TARGETS`
- `DEFAULT_REPEAT_COUNT`

The script fails fast if any `REPLACE_ME` placeholder is still present.

## Run Ablation

```bash
cd backend
python3 ablation/run_manual_ablation.py
```

Optional flags:
- `--output-csv`: write to a different detailed results file
- `--run-name`: execute only selected manual runs
- `--demo-mode`: run the pipeline in demo mode
- `--skip-execution`: skip target-model execution and record only pipeline outputs

## Evaluate Results

```bash
cd backend
python3 ablation/evaluate_ablation.py
```

## Visualize Results

```bash
cd backend
python3 ablation/visualize_ablation.py
```

Outputs:
- `ablation/output/ablation_results.csv`
- `ablation/output/ablation_summary.csv`
- `ablation/output/ablation_summary_by_model_count.csv`
- `ablation/output/ablation_report.json`
- `ablation/output/figures/per_run_council_metrics.png`
- `ablation/output/figures/per_run_council_metrics.pdf`
- `ablation/output/figures/combined_by_model_count.png`
- `ablation/output/figures/combined_by_model_count.pdf`
- `ablation/output/figures/success_failure_by_model_count.png`
- `ablation/output/figures/success_failure_by_model_count.pdf`
