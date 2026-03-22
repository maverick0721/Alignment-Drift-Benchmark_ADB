import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT_DIR / "figures"

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


def plot_refusal_margin_overlay(refusal_df, margin_df):

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

    fig.tight_layout()
    plt.savefig(FIGURES_DIR / "refusal_margin_overlay.pdf")
    plt.show()
