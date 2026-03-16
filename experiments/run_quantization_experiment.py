import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from evaluation.evaluate_alignment_drift import evaluate, get_output_path


MODELS = [
    "google/gemma-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]

PRECISIONS = ["fp16", "int8", "int4"]


def run_experiment():
    for model_name in MODELS:
        for precision in PRECISIONS:
            output_path = get_output_path(model_name, precision)
            if output_path.exists():
                print(f"Skipping {model_name} with {precision}; results already exist at {output_path}.")
                continue

            print(f"Running {model_name} with {precision}")
            evaluate(model_name, precision=precision)


if __name__ == "__main__":
    run_experiment()