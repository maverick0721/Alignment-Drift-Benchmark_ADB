import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT_DIR / "analysis"

st.set_page_config(page_title="Alignment Drift Explorer", layout="wide")

st.title("🛡️ Alignment Drift Benchmark Explorer")
st.markdown("Interactive dashboard analyzing how INT8/INT4 quantization degrades the safety guardrails of LLMs.")

@st.cache_data
def load_data():
    refusal = pd.read_csv(ANALYSIS_DIR / "refusal_summary.csv")
    category = pd.read_csv(ANALYSIS_DIR / "category_summary.csv")
    drift = pd.read_csv(ANALYSIS_DIR / "drift_summary.csv")
    return refusal, category, drift

try:
    refusal_df, category_df, drift_df = load_data()
    
    st.header("Refusal Rates by Precision")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(refusal_df.style.format({"refusal_rate": "{:.2%}", "refusal_rate_se": "{:.4f}"}))
        
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=refusal_df, x="model", y="refusal_rate", hue="precision", ax=ax)
        ax.set_title("Safety Retention Across Quantization")
        st.pyplot(fig)
        
    st.header("Drift Disaggregated by Category")
    model_filter = st.selectbox("Select Model to Inspect:", category_df["model"].unique())
    
    filtered_cat = category_df[category_df["model"] == model_filter]
    
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=filtered_cat, x="precision", y="refusal_rate", hue="category", marker="o", ax=ax2)
    ax2.set_title(f"Ablation for {model_filter}")
    st.pyplot(fig2)

    st.success("App successfully loaded data!")
    
except Exception as e:
    st.error(f"Could not load analysis data. Run `bash run.sh` first. Error: {e}")
