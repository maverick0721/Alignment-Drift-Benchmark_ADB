# Alignment-Drift-Benchmark_ADB
ADB is a research framework that measures how model compression (like INT8/INT4 quantization) degrades safety alignment in large language models compared to their general capabilities

## Quick Start

```bash
pip install -r requirements.txt
```

Run evaluation for selected models and precisions:

```bash
python evaluation/evaluate_alignment_drift.py --precision fp16
python evaluation/evaluate_alignment_drift.py --precision int8
python evaluation/evaluate_alignment_drift.py --precision int4
```

Run analysis and regenerate all summary tables and figures:

```bash
python analysis/run_analysis.py
```

## Key Outputs

- Logs: `evaluation/logs/results_*.csv`
- Summaries:
	- `analysis/refusal_summary.csv`
	- `analysis/drift_summary.csv`
	- `analysis/margin_summary.csv`
- Coverage diagnostics:
	- `analysis/refusal_coverage.csv`
	- `analysis/drift_coverage.csv`
	- `analysis/margin_coverage.csv`
- Figures:
	- `figures/refusal_plot.pdf`
	- `figures/drift_plot.pdf`
	- `figures/margin_plot.pdf`
	- `figures/refusal_margin_overlay.pdf`
	- `figures/data_completeness_panel.pdf`

## Metric Coverage Notes

- `refusal_summary.csv` and `margin_summary.csv` are reported for all available precisions (`fp16`, `int8`, `int4`) per model.
- `drift_summary.csv` is reported only for quantized precisions (`int8`, `int4`).
- Drift is defined relative to each model's `fp16` baseline, so there is no standalone `fp16` drift row by design.
- Coverage CSVs (`refusal_coverage.csv`, `drift_coverage.csv`, `margin_coverage.csv`) explicitly mark each model/precision pair as `computed`, `skipped`, or `not_applicable`.
- In `drift_coverage.csv`, `fp16` appears as `not_applicable` because it is the baseline reference precision.

## Paper Assets

- Manuscript source: `paper/paper.tex`
- Headline metrics: `paper/results_headline_summary.txt`
- Stage 8 checklist: `paper/STAGE8_SUBMISSION_VISIBILITY.md`
- arXiv package builder: `scripts/package_arxiv.sh`

Create arXiv upload package:

```bash
bash scripts/package_arxiv.sh
```

This generates:

- `paper/arxiv_submission/arxiv_source.tar.gz`

## Release and Visibility

Use `paper/STAGE8_SUBMISSION_VISIBILITY.md` as the final runbook for:

- arXiv submission steps
- GitHub release preparation
- public launch messaging and distribution
