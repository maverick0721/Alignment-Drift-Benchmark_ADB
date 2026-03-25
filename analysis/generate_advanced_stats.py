import pandas as pd
import glob
from pathlib import Path
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "evaluation" / "logs"
FIGURES_DIR = ROOT_DIR / "figures"
ANALYSIS_DIR = ROOT_DIR / "analysis"

sns.set_theme(style="whitegrid", font_scale=1.2)
sns.set_palette("colorblind")

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    files = glob.glob(str(LOG_DIR / "*.csv"))
    
    dfs = []
    for f in files:
        if "bak" in f: continue
        df = pd.read_csv(f)
        if "refusal_margin" in df.columns and "refusal" in df.columns:
            dfs.append(df)
            
    if not dfs:
        print("No raw logs found.")
        return
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.dropna(subset=["refusal_margin"])
    
    # 1. Violin Plot for Margins
    plt.figure(figsize=(10, 6))
    precision_order = ["fp16", "int8", "int4"]
    sns.violinplot(
        data=full_df, 
        x="precision", 
        y="refusal_margin", 
        hue="model", 
        order=precision_order, 
        split=False,
        inner="quartile",
        linewidth=1.5
    )
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title("Distribution of Refusal Margins by Precision")
    plt.ylabel("Log-Prob Refusal Margin")
    plt.xlabel("Precision")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "violin_margin.pdf")
    plt.close()
    
    # 2. Chi-Squared P-values for FP16 vs INT4 Drift
    # We want to save a p-value table
    results = []
    for model in full_df["model"].unique():
        mdf = full_df[full_df["model"] == model]
        
        fp16 = mdf[mdf["precision"] == "fp16"]
        int4 = mdf[mdf["precision"] == "int4"]
        
        if len(fp16) > 0 and len(int4) > 0:
            fp16_refusals = fp16["refusal"].sum()
            fp16_comps = len(fp16) - fp16_refusals
            
            int4_refusals = int4["refusal"].sum()
            int4_comps = len(int4) - int4_refusals
            
            # Contingency table
            table = [
                [fp16_refusals, fp16_comps],
                [int4_refusals, int4_comps]
            ]
            
            chi2, p, dof, ex = chi2_contingency(table)
            results.append({
                "model": model,
                "fp16_refusal_rate": fp16["refusal"].mean(),
                "int4_refusal_rate": int4["refusal"].mean(),
                "p_value": p,
                "significant_at_05": p < 0.05
            })
            
    p_df = pd.DataFrame(results)
    p_df.to_csv(ANALYSIS_DIR / "p_values.csv", index=False)
    
    # 3. Heatmap for Refusal Rates
    heatmap_data = full_df.groupby(["model", "precision"])["refusal"].mean().unstack()[precision_order]
    # Shorten model names for clean axis labels
    heatmap_data.index = [n.split("/")[-1] for n in heatmap_data.index]
    plt.figure(figsize=(8, 5))
    sns.heatmap(heatmap_data, annot=True, cmap="viridis", fmt=".3f",
                cbar_kws={'label': 'Refusal Rate'},
                xticklabels=["FP16", "INT8", "INT4"],
                linewidths=0.5, linecolor="white")
    plt.xlabel("Precision")
    plt.ylabel("Model")
    plt.title("Refusal Rate Heatmap Across Models and Precisions")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "heatmap_drift.pdf")
    plt.close()

if __name__ == "__main__":
    main()
