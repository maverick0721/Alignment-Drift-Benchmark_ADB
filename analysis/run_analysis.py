from analyze_results import (
	load_all_results,
	save_summary,
	compute_refusal_rate,
	compute_drift_ratio,
	compute_margin_stats,
	compute_refusal_coverage,
	compute_drift_ratio_with_coverage,
	compute_margin_coverage,
)
from plot_results import (
	plot_refusal,
	plot_drift,
	plot_margin,
	plot_refusal_margin_overlay,
	plot_data_completeness,
)

df = load_all_results()

save_summary(df)

refusal_df = compute_refusal_rate(df)
drift_df = compute_drift_ratio(df)
margin_df = compute_margin_stats(df)
refusal_coverage_df = compute_refusal_coverage(df)
_, drift_coverage_df = compute_drift_ratio_with_coverage(df)
margin_coverage_df = compute_margin_coverage(df)

plot_refusal(refusal_df)
plot_drift(drift_df)

if not margin_df.empty:
	plot_margin(margin_df)
	plot_refusal_margin_overlay(
		refusal_df,
		margin_df,
		refusal_coverage_df,
		drift_coverage_df,
		margin_coverage_df,
	)

plot_data_completeness(refusal_coverage_df, drift_coverage_df, margin_coverage_df)