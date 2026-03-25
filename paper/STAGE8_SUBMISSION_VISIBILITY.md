# Stage 8 - Paper Finalization, Submission, and Visibility

## 1. Paper Finalization Checklist

- [ ] Compile paper locally from `paper/paper.tex`
- [ ] Confirm all figures render and captions match claims
- [ ] Freeze final numbers from:
  - `analysis/refusal_summary.csv`
  - `analysis/drift_summary.csv`
  - `analysis/margin_summary.csv`
- [ ] Confirm completeness tables:
  - `analysis/refusal_coverage.csv`
  - `analysis/drift_coverage.csv`
  - `analysis/margin_coverage.csv`
- [ ] Proofread abstract, title, and conclusion for consistency

## 2. arXiv Submission (Source Upload)

1. Build source package:
   - Run `bash scripts/package_arxiv.sh`
2. Verify package content:
   - `paper/paper.tex`
   - `figures/refusal_plot.pdf`
   - `figures/drift_plot.pdf`
   - `figures/margin_plot.pdf`
   - `figures/refusal_margin_overlay.pdf`
   - `figures/data_completeness_panel.pdf`
3. Upload `paper/arxiv_submission/arxiv_source.tar.gz` to arXiv.
4. In arXiv metadata:
   - Primary category: `cs.CL`
   - Add links to repository and reproducibility instructions.
5. Compile preview on arXiv and fix any TeX warnings/errors.
6. Submit and record arXiv ID in repo README.

## 3. GitHub Repository Release Prep

- [ ] Ensure `README.md` includes:
  - project overview
  - setup instructions
  - run commands for evaluation and analysis
  - figure list and interpretation notes
- [ ] Tag release: `v1.0.0`
- [ ] Add release notes:
  - benchmark scope
  - key findings
  - limitations

Suggested commands:

```bash
git add .
git commit -m "Stage 8: final paper, figures, and release assets"
git tag -a v1.0.0 -m "ADB v1.0.0"
git push origin main --tags
```

## 4. Public Visibility Plan

### Day 0 (submission day)
- Post summary thread on X/LinkedIn with main figure and one key table.
- Publish repository with arXiv preprint link.
- Share in relevant communities:
  - Hugging Face community
  - r/MachineLearning
  - alignment/safety Slack or Discord groups

### Day 1-3
- Release a short technical blog post that explains:
  - benchmark design
  - drift metric definition
  - surprising model-specific trends
- Share reproducibility command snippets.

### Week 1
- Submit to Papers With Code (if task framing fits).
- Reach out directly to researchers working on quantization/safety.
- Record first-wave engagement metrics (stars, forks, citations, discussions).

## 5. Visibility Assets To Publish

- `figures/refusal_plot.pdf`
- `figures/drift_plot.pdf`
- `figures/margin_plot.pdf`
- `figures/refusal_margin_overlay.pdf`
- `figures/data_completeness_panel.pdf`
- `paper/results_headline_summary.txt`

## 6. Messaging Template

"We benchmarked alignment drift under quantization across Gemma-2B, Mistral-7B, and Llama-3-8B for FP16/INT8/INT4. The repository includes full logs, coverage diagnostics, and reproducible plots."
