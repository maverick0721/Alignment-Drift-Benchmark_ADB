import pandas as pd
import glob
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "evaluation" / "logs"
ANALYSIS_DIR = ROOT_DIR / "analysis"

# LOAD & COMBINE LOGS
def load_all_results():

    files = glob.glob(str(LOG_DIR / "*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV logs found in {LOG_DIR}")

    required_columns = {"model", "precision", "refusal"}
    dfs = []
    lfs_pointer_files = []

    for file_path in files:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
            first_line = handle.readline().strip()

        if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
            lfs_pointer_files.append(file_path)
            continue

        df = pd.read_csv(file_path)

        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(
                f"Missing required columns {sorted(missing)} in {file_path}. "
                f"Found columns: {list(df.columns)}"
            )

        dfs.append(df)

    if not dfs:
        if lfs_pointer_files:
            raise RuntimeError(
                "No usable CSV result logs were found. The files in evaluation/logs "
                "appear to be Git LFS pointers. Run 'git lfs pull' in the repository "
                "to download the real CSV contents."
            )
        raise ValueError("No valid CSV logs were loaded from evaluation/logs.")

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

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    refusal.to_csv(ANALYSIS_DIR / "refusal_summary.csv", index=False)
    drift.to_csv(ANALYSIS_DIR / "drift_summary.csv", index=False)

    print("Saved summary tables.")