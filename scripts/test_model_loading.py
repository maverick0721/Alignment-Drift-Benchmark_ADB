import argparse
import os
import time

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick Hugging Face model load test")
    parser.add_argument(
        "--model",
        default="google/gemma-2b-it",
        help="Model ID to test. Example: sshleifer/tiny-gpt2",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to from_pretrained (default: auto)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    token = os.getenv("HF_TOKEN")

    print("[1/3] Importing transformers modules...")
    start = time.perf_counter()
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    transformers.logging.set_verbosity_error()

    print(f"[1/3] Done in {time.perf_counter() - start:.2f}s")
    print(f"[2/3] Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)

    _ = tokenizer
    print(f"[3/3] Loading model weights for {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=token,
        device_map=args.device_map,
    )

    _ = model
    print("Model loaded successfully")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
