import argparse
import gc
import os
import time

import torch
from dotenv import load_dotenv


MODELS = [
    "google/gemma-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Hugging Face model loading for benchmark models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="One or more model IDs to test (default: all 3 benchmark models)",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to from_pretrained (default: auto)",
    )
    return parser.parse_args()


def test_model(model_id: str, token: str, device_map: str) -> bool:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    transformers.logging.set_verbosity_error()

    try:
        print(f"  [tokenizer] Loading...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        _ = tokenizer

        print(f"  [weights]   Loading...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token,
            device_map=device_map,
        )
        _ = model

        # Free GPU memory before next model
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True
    except Exception as e:
        print(f"  [FAILED] {e}")
        return False


def main() -> None:
    args = parse_args()
    load_dotenv()
    token = os.getenv("HF_TOKEN")

    print("Importing transformers modules...")
    start = time.perf_counter()
    import transformers  # noqa: F401
    print(f"Done in {time.perf_counter() - start:.2f}s\n")

    results = {}
    for i, model_id in enumerate(args.models, 1):
        print(f"[{i}/{len(args.models)}] Testing: {model_id}")
        t0 = time.perf_counter()
        ok = test_model(model_id, token, args.device_map)
        elapsed = time.perf_counter() - t0
        results[model_id] = ok
        status = "OK" if ok else "FAILED"
        print(f"  -> {status} ({elapsed:.1f}s)\n")

    print("=" * 50)
    print("Summary:")
    for model_id, ok in results.items():
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {model_id}")

    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
