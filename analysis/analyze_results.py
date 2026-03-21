import pandas as pd
import glob

# LOAD & COMBINE LOGS
def load_all_results():

    files = glob.glob("logs/*.csv")

    dfs = [pd.read_csv(f) for f in files]

    df = pd.concat(dfs, ignore_index=True)

    return df

# COMPUTE REFUSAL RATES
def compute_refusal_rate(df):

    grouped = df.groupby(["model", "precision"])["refusal"].mean()

    return grouped.reset_index(name="refusal_rate")

# COMPUTE CATEGORY-WISE METRICS
def compute_category_stats(df):

    stats = df.groupby(
        ["model", "precision", "category"]
    )["refusal"].mean()

    return stats.reset_index(name="refusal_rate")

# COMPUTE DRIFT RATIO
def compute_drift_ratio(df):

    results = []

    for model in df["model"].unique():

        model_df = df[df["model"] == model]

        baseline = model_df[model_df["precision"] == "fp16"]

        base_refusal = baseline["refusal"].mean()
        base_cap = 1 - baseline["refusal"].mean()

        for precision in ["int8", "int4"]:

            current = model_df[model_df["precision"] == precision]

            align_drop = abs(current["refusal"].mean() - base_refusal)
            cap_drop = abs((1 - current["refusal"].mean()) - base_cap)

            drift = align_drop / (cap_drop + 1e-6)

            results.append({
                "model": model,
                "precision": precision,
                "drift_ratio": drift
            })

    return pd.DataFrame(results)

#SAVE SUMMARY TABLES
def save_summary(df):

    refusal = compute_refusal_rate(df)
    drift = compute_drift_ratio(df)

    refusal.to_csv("analysis/refusal_summary.csv", index=False)
    drift.to_csv("analysis/drift_summary.csv", index=False)

    print("Saved summary tables.")