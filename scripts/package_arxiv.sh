#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/paper/arxiv_submission"

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/arxiv_source.tar.gz

# Copy paper and figures to an arXiv-friendly source directory.
cp "$ROOT_DIR/paper/paper.tex" "$OUT_DIR/paper.tex"
cp "$ROOT_DIR/figures/refusal_plot.pdf" "$OUT_DIR/refusal_plot.pdf"
cp "$ROOT_DIR/figures/drift_plot.pdf" "$OUT_DIR/drift_plot.pdf"
cp "$ROOT_DIR/figures/margin_plot.pdf" "$OUT_DIR/margin_plot.pdf"
cp "$ROOT_DIR/figures/refusal_margin_overlay.pdf" "$OUT_DIR/refusal_margin_overlay.pdf"
cp "$ROOT_DIR/figures/data_completeness_panel.pdf" "$OUT_DIR/data_completeness_panel.pdf"

# Rewrite figure paths for arXiv package layout.
sed -i 's#../figures/refusal_plot.pdf#refusal_plot.pdf#g' "$OUT_DIR/paper.tex"
sed -i 's#../figures/drift_plot.pdf#drift_plot.pdf#g' "$OUT_DIR/paper.tex"
sed -i 's#../figures/margin_plot.pdf#margin_plot.pdf#g' "$OUT_DIR/paper.tex"
sed -i 's#../figures/refusal_margin_overlay.pdf#refusal_margin_overlay.pdf#g' "$OUT_DIR/paper.tex"
sed -i 's#../figures/data_completeness_panel.pdf#data_completeness_panel.pdf#g' "$OUT_DIR/paper.tex"

(
  cd "$OUT_DIR"
  tar -czf arxiv_source.tar.gz \
    paper.tex \
    refusal_plot.pdf \
    drift_plot.pdf \
    margin_plot.pdf \
    refusal_margin_overlay.pdf \
    data_completeness_panel.pdf
)

echo "Created: $OUT_DIR/arxiv_source.tar.gz"
