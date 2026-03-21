import matplotlib.pyplot as plt

# PRECISION-REFUSAL RATE PLOT
def plot_refusal(refusal_df):

    for model in refusal_df["model"].unique():

        data = refusal_df[refusal_df["model"] == model]

        plt.plot(data["precision"], data["refusal_rate"], marker="o", label=model)

    plt.xlabel("Precision")
    plt.ylabel("Refusal Rate")
    plt.title("Precision vs Refusal Rate")
    plt.legend()

    plt.savefig("figures/refusal_plot.pdf")
    plt.show()

# PLOT DRIFT RATIO
def plot_drift(drift_df):

    for model in drift_df["model"].unique():

        data = drift_df[drift_df["model"] == model]

        plt.plot(data["precision"], data["drift_ratio"], marker="o", label=model)

    plt.xlabel("Precision")
    plt.ylabel("Drift Ratio")
    plt.title("Alignment Drift Across Precision")
    plt.legend()

    plt.savefig("figures/drift_plot.pdf")
    plt.show()
