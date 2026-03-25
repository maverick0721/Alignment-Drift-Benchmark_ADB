#!/bin/bash
set -e

echo "================================================="
echo " Alignment Drift Benchmark Pipeline              "
echo "================================================="

# Set random seeds for reproducibility
export PYTHONHASHSEED=42
export CUDA_SEED=42
echo "Setting random seeds (PYTHONHASHSEED=42, CUDA_SEED=42)"

echo "1. Generating Synthetic Prompts..."
cd benchmark
python generate_prompts.py
cd ..

echo "2. Running Model Evaluations (Requires GPU)..."
echo "Note: If you want to run fresh inference, remove 'evaluation/logs/*.csv' first."
cd evaluation
# python evaluate_alignment_drift.py --model meta-llama/Meta-Llama-3-8B-Instruct
cd ..

echo "3. Parsing Baseline Refusals..."
cd scripts
python reparse_refusals.py
cd ..

echo "4. Generating Advanced Data Analysis..."
cd analysis
python run_analysis.py
python generate_advanced_stats.py
cd ..

echo "5. Compiling the Research Paper PDF..."
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
cd ..

echo "Done! The final manuscript is at paper/paper.pdf."
echo "You can view the interactive dashboard with: streamlit run app.py"
