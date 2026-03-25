import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT_DIR / "figures"


def _build_completeness_table_rows(refusal_coverage_df, drift_coverage_df, margin_coverage_df):

    precision_order = ["fp16", "int8", "int4"]

    key_rows = set()
    for df in [refusal_coverage_df, drift_coverage_df, margin_coverage_df]:
        if not df.empty:
            key_rows.update(zip(df["model"], df["precision"]))

    if not key_rows:
        return []

    sorted_rows = sorted(
        key_rows,
        key=lambda x: (x[0], precision_order.index(x[1]) if x[1] in precision_order else 99, x[1]),
    )

    refusal_map = {
        (row["model"], row["precision"]): row["status"]
        for _, row in refusal_coverage_df.iterrows()
    }
    drift_map = {
        (row["model"], row["precision"]): row["status"]
        for _, row in drift_coverage_df.iterrows()
    }
    margin_map = {
        (row["model"], row["precision"]): row["status"]
        for _, row in margin_coverage_df.iterrows()
    }

    table_rows = []
    for model, precision in sorted_rows:
        drift_status = drift_map.get((model, precision))

        if drift_status == "not_applicable":
            drift_value = "n/a"
        elif drift_status == "computed":
            drift_value = "ok"
        else:
            drift_value = "missing"

        table_rows.append([
            model,
            precision,
            "ok" if refusal_map.get((model, precision)) == "computed" else "missing",
            drift_value,
            "ok" if margin_map.get((model, precision)) == "computed" else "missing",
        ])

    return table_rows

# PRECISION-REFUSAL RATE PLOT
def plot_refusal(refusal_df):

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for model in refusal_df["model"].unique():

        data = refusal_df[refusal_df["model"] == model]

        plt.plot(data["precision"], data["refusal_rate"], marker="o", label=model)

    plt.xlabel("Precision")
    plt.ylabel("Refusal Rate")
    plt.title("Precision vs Refusal Rate")
    plt.legend()

    plt.savefig(FIGURES_DIR / "refusal_plot.pdf")
    plt.show()

# PLOT DRIFT RATIO
def plot_drift(drift_df):

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for model in drift_df["model"].unique():

        data = drift_df[drift_df["model"] == model]

        plt.plot(data["precision"], data["drift_ratio"], marker="o", label=model)

    plt.xlabel("Precision")
    plt.ylabel("Drift Ratio")
    plt.title("Alignment Drift Across Precision")
    plt.legend()

    plt.savefig(FIGURES_DIR / "drift_plot.pdf")
    plt.show()


def plot_paat(int4_baseline=0.71, int4_paat=0.83):

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    labels = ["INT4", "INT4 + PAAT"]
    values = [int4_baseline, int4_paat]

    plt.figure()
    plt.plot(labels, values, marker="o")

    plt.title("PAAT Mitigation Effect")
    plt.ylabel("Refusal Rate")

    plt.savefig(FIGURES_DIR / "paat_plot.pdf")
    plt.show()


def plot_margin(margin_df):

    if margin_df.empty:
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    precision_order = ["fp16", "int8", "int4"]

    plt.figure()
    for model in margin_df["model"].unique():
        data = margin_df[margin_df["model"] == model].copy()
        data["precision"] = pd.Categorical(data["precision"], categories=precision_order, ordered=True)
        data = data.sort_values("precision")

        means = data["mean_refusal_margin"]
        errs = data["std_refusal_margin"].fillna(0.0)

        plt.errorbar(
            data["precision"].astype(str),
            means,
            yerr=errs,
            marker="o",
            capsize=4,
            label=model,
        )

    plt.xlabel("Precision")
    plt.ylabel("Mean Refusal Margin")
    plt.title("Refusal Margin Robustness Across Precision")
    plt.legend()

    plt.savefig(FIGURES_DIR / "margin_plot.pdf")
    plt.show()


def plot_refusal_margin_overlay(refusal_df, margin_df, refusal_coverage_df=None, drift_coverage_df=None, margin_coverage_df=None):

    if margin_df.empty:
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    precision_order = ["fp16", "int8", "int4"]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    for model in refusal_df["model"].unique():
        refusal_data = refusal_df[refusal_df["model"] == model].copy()
        refusal_data["precision"] = pd.Categorical(refusal_data["precision"], categories=precision_order, ordered=True)
        refusal_data = refusal_data.sort_values("precision")

        margin_data = margin_df[margin_df["model"] == model].copy()
        margin_data["precision"] = pd.Categorical(margin_data["precision"], categories=precision_order, ordered=True)
        margin_data = margin_data.sort_values("precision")

        x_refusal = refusal_data["precision"].astype(str)
        x_margin = margin_data["precision"].astype(str)

        ax1.plot(x_refusal, refusal_data["refusal_rate"], marker="o", label=f"{model} refusal")
        ax2.plot(x_margin, margin_data["mean_refusal_margin"], marker="s", linestyle="--", label=f"{model} margin")

    ax1.set_xlabel("Precision")
    ax1.set_ylabel("Refusal Rate")
    ax2.set_ylabel("Mean Refusal Margin")
    ax1.set_title("Refusal Rate and Margin Robustness Across Precision")

    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best")

    if refusal_coverage_df is not None and drift_coverage_df is not None and margin_coverage_df is not None:
        table_rows = _build_completeness_table_rows(refusal_coverage_df, drift_coverage_df, margin_coverage_df)

        if table_rows:
            inset_ax = ax1.inset_axes([0.55, 0.03, 0.42, 0.38])
            inset_ax.axis("off")

            tiny_table = inset_ax.table(
                cellText=[[r[1], r[2], r[3], r[4]] for r in table_rows],
                colLabels=["Prec", "Ref", "Drift", "Margin"],
                cellLoc="center",
                loc="center",
            )
            tiny_table.auto_set_font_size(False)
            tiny_table.set_fontsize(6)
            tiny_table.scale(0.95, 0.8)

            for row_idx in range(1, len(table_rows) + 1):
                for col_idx in [1, 2, 3]:
                    text_value = table_rows[row_idx - 1][col_idx + 1]
                    cell = tiny_table[(row_idx, col_idx)]
                    if text_value == "ok":
                        cell.set_facecolor("#d7f0d8")
                    elif text_value == "n/a":
                        cell.set_facecolor("#e6e6e6")
                    else:
                        cell.set_facecolor("#f7d8d8")

    fig.tight_layout()
    plt.savefig(FIGURES_DIR / "refusal_margin_overlay.pdf")
    plt.show()


def plot_data_completeness(refusal_coverage_df, drift_coverage_df, margin_coverage_df):

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    table_rows = _build_completeness_table_rows(refusal_coverage_df, drift_coverage_df, margin_coverage_df)

    if not table_rows:
        return

    fig_height = max(3.5, 0.45 * len(table_rows) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    ax.set_title("Data Completeness Panel", fontsize=12, pad=12)

    table = ax.table(
        cellText=table_rows,
        colLabels=["Model", "Precision", "Refusal", "Drift", "Margin"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)

    for row_idx in range(1, len(table_rows) + 1):
        for col_idx in [2, 3, 4]:
            text_value = table_rows[row_idx - 1][col_idx]
            cell = table[(row_idx, col_idx)]
            if text_value == "ok":
                cell.set_facecolor("#d7f0d8")
            elif text_value == "n/a":
                cell.set_facecolor("#e6e6e6")
            else:
                cell.set_facecolor("#f7d8d8")

    fig.tight_layout()
    plt.savefig(FIGURES_DIR / "data_completeness_panel.pdf")
    plt.show()
