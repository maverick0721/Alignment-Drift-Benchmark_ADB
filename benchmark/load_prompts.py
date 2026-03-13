import json
from pathlib import Path


def load_prompts():

    base = Path("benchmark/prompts")

    datasets = {}

    for file in base.glob("*.json"):
        with open(file) as f:
            data = json.load(f)

        datasets[data["category"]] = data["prompts"]

    return datasets


if __name__ == "__main__":
    prompts = load_prompts()

    for category in prompts:
        print(category, len(prompts[category]))