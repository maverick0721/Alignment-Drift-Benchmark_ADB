from analyze_results import load_all_results, save_summary
from plot_results import plot_refusal, plot_drift

df = load_all_results()

save_summary(df)

refusal_df = compute_refusal_rate(df)
drift_df = compute_drift_ratio(df)

plot_refusal(refusal_df)
plot_drift(drift_df)