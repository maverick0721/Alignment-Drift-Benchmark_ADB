import json
from pathlib import Path


def load_prompts():
    base = Path(__file__).resolve().parent / "prompts"

    datasets = {}

    for file in base.glob("*.json"):
        with file.open(encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "category" in data and "prompts" in data:
            datasets[data["category"]] = data["prompts"]
        elif isinstance(data, dict):
            for category, prompts in data.items():
                if isinstance(prompts, list):
                    datasets[category] = prompts

    return datasets


if __name__ == "__main__":
    prompts = load_prompts()

    for category in prompts:
        print(category, len(prompts[category]))