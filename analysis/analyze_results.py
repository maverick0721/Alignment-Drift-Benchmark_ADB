import pandas as pd
import glob
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "evaluation" / "logs"
ANALYSIS_DIR = ROOT_DIR / "analysis"

# LOAD & COMBINE LOGS
def load_all_results():

    files = sorted(glob.glob(str(LOG_DIR / "*.csv")))

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


def compute_refusal_coverage(df):

    rows = []
    available = set(
        df.groupby(["model", "precision"]).size().index.to_list()
    )

    expected_precisions = ["fp16", "int8", "int4"]

    for model in sorted(df["model"].unique()):
        for precision in expected_precisions:
            if (model, precision) in available:
                rows.append({
                    "model": model,
                    "precision": precision,
                    "status": "computed",
                    "reason": "",
                })
            else:
                rows.append({
                    "model": model,
                    "precision": precision,
                    "status": "skipped",
                    "reason": "missing_precision_results",
                })

    return pd.DataFrame(rows, columns=["model", "precision", "status", "reason"])

# COMPUTE CATEGORY-WISE METRICS
def compute_category_stats(df):

    stats = df.groupby(
        ["model", "precision", "category"]
    )["refusal"].mean()

    return stats.reset_index(name="refusal_rate")

# COMPUTE DRIFT RATIO
def compute_drift_ratio_with_coverage(df):

    results = []
    coverage = []

    for model in sorted(df["model"].unique()):

        model_df = df[df["model"] == model]
        baseline = model_df[model_df["precision"] == "fp16"]

        coverage.append({
            "model": model,
            "precision": "fp16",
            "status": "not_applicable",
            "reason": "baseline_reference_precision",
        })

        if baseline.empty:
            for precision in ["int8", "int4"]:
                coverage.append({
                    "model": model,
                    "precision": precision,
                    "status": "skipped",
                    "reason": "missing_fp16_baseline",
                })
            continue

        base_refusal = baseline["refusal"].mean()
        base_cap = 1 - base_refusal

        for precision in ["int8", "int4"]:

            current = model_df[model_df["precision"] == precision]

            if current.empty:
                coverage.append({
                    "model": model,
                    "precision": precision,
                    "status": "skipped",
                    "reason": "missing_precision_results",
                })
                continue

            current_refusal = current["refusal"].mean()
            align_drop = abs(current_refusal - base_refusal)
            cap_drop = abs((1 - current_refusal) - base_cap)
            drift = align_drop / (cap_drop + 1e-6)

            results.append({
                "model": model,
                "precision": precision,
                "drift_ratio": drift,
            })
            coverage.append({
                "model": model,
                "precision": precision,
                "status": "computed",
                "reason": "",
            })

    drift_df = pd.DataFrame(results, columns=["model", "precision", "drift_ratio"])
    coverage_df = pd.DataFrame(coverage, columns=["model", "precision", "status", "reason"])

    return drift_df, coverage_df


def compute_drift_ratio(df):
    drift_df, _ = compute_drift_ratio_with_coverage(df)
    return drift_df


def compute_margin_stats(df):

    if "refusal_margin" not in df.columns:
        return pd.DataFrame()

    stats = df.groupby(["model", "precision"])["refusal_margin"].agg(["mean", "std"]).reset_index()
    return stats.rename(columns={"mean": "mean_refusal_margin", "std": "std_refusal_margin"})


def compute_margin_coverage(df):

    rows = []
    expected_precisions = ["fp16", "int8", "int4"]
    has_margin = "refusal_margin" in df.columns

    if has_margin:
        margin_df = df[df["refusal_margin"].notna()]
        available = set(
            margin_df.groupby(["model", "precision"]).size().index.to_list()
        )
    else:
        available = set()

    for model in sorted(df["model"].unique()):
        for precision in expected_precisions:
            if not has_margin:
                rows.append({
                    "model": model,
                    "precision": precision,
                    "status": "skipped",
                    "reason": "missing_refusal_margin_column",
                })
            elif (model, precision) in available:
                rows.append({
                    "model": model,
                    "precision": precision,
                    "status": "computed",
                    "reason": "",
                })
            else:
                rows.append({
                    "model": model,
                    "precision": precision,
                    "status": "skipped",
                    "reason": "missing_precision_results_or_nan_margin",
                })

    return pd.DataFrame(rows, columns=["model", "precision", "status", "reason"])

#SAVE SUMMARY TABLES
def save_summary(df):

    refusal = compute_refusal_rate(df)
    refusal_coverage = compute_refusal_coverage(df)
    drift, drift_coverage = compute_drift_ratio_with_coverage(df)
    margin = compute_margin_stats(df)
    margin_coverage = compute_margin_coverage(df)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    refusal.to_csv(ANALYSIS_DIR / "refusal_summary.csv", index=False)
    refusal_coverage.to_csv(ANALYSIS_DIR / "refusal_coverage.csv", index=False)
    drift.to_csv(ANALYSIS_DIR / "drift_summary.csv", index=False)
    drift_coverage.to_csv(ANALYSIS_DIR / "drift_coverage.csv", index=False)
    margin_coverage.to_csv(ANALYSIS_DIR / "margin_coverage.csv", index=False)

    if not margin.empty:
        margin.to_csv(ANALYSIS_DIR / "margin_summary.csv", index=False)

    refusal_skipped = refusal_coverage[refusal_coverage["status"] == "skipped"]
    if not refusal_skipped.empty:
        print("Warning: Some refusal rates are missing due to absent precision logs.")
        print(refusal_skipped.to_string(index=False))

    skipped = drift_coverage[drift_coverage["status"] == "skipped"]
    if not skipped.empty:
        print("Warning: Some drift ratios were skipped due to missing baselines/results.")
        print(skipped.to_string(index=False))

    margin_skipped = margin_coverage[margin_coverage["status"] == "skipped"]
    if not margin_skipped.empty:
        print("Warning: Some margin stats are missing due to absent precision logs or margin values.")
        print(margin_skipped.to_string(index=False))

    print("Saved summary tables.")