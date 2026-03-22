import argparse
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.load_prompts import load_prompts


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_quantization_noise(model, noise_level=0.01):
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad and torch.is_floating_point(param):
                noise = torch.randn_like(param) * noise_level
                param.add_(noise)


def _single_token_ids(tokenizer, terms):
    ids = []
    for term in terms:
        token_ids = tokenizer.encode(term, add_special_tokens=False)
        if token_ids:
            ids.append(token_ids[0])
    if not ids:
        raise ValueError("Could not derive any token ids for refusal/comply terms.")
    return sorted(set(ids))


def refusal_margin_loss(logits, refusal_token_ids, comply_token_ids, margin_threshold=1.0):
    # Use next-token logits at the end of the prompt.
    final_logits = logits[:, -1, :].float()

    refusal_scores = final_logits[:, refusal_token_ids]
    comply_scores = final_logits[:, comply_token_ids]

    # Aggregate each class with log-sum-exp to approximate class log-probability.
    refusal_logp = torch.logsumexp(refusal_scores, dim=-1)
    comply_logp = torch.logsumexp(comply_scores, dim=-1)

    # Enforce refusal_logp - comply_logp >= margin_threshold.
    margin = refusal_logp - comply_logp
    return F.relu(margin_threshold - margin).mean()


def train_step(
    prompt,
    tokenizer,
    model,
    optimizer,
    device,
    refusal_token_ids,
    comply_token_ids,
    margin_weight=0.1,
    margin_threshold=1.0,
    max_length=256,
):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model(**inputs, labels=inputs["input_ids"])

    base_loss = outputs.loss
    margin_loss = refusal_margin_loss(
        outputs.logits,
        refusal_token_ids,
        comply_token_ids,
        margin_threshold=margin_threshold,
    )
    total_loss = base_loss + margin_weight * margin_loss

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item(), base_loss.item(), margin_loss.item()


def parse_args():
    parser = argparse.ArgumentParser(description="Train PAAT prototype for quantization-robust refusal behavior.")
    parser.add_argument("--model-name", default="google/gemma-2b-it")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "paat_model"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-prompts", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--noise-level", type=float, default=0.01)
    parser.add_argument("--margin-weight", type=float, default=0.1)
    parser.add_argument("--margin-threshold", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=HF_TOKEN,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    prompts = load_prompts().get("harmful", [])
    if not prompts:
        raise ValueError("No harmful prompts found in benchmark prompts.")

    refusal_terms = [" I cannot", " I can't", " Sorry", " I refuse", " cannot"]
    comply_terms = [" Sure", " Here", " First", " To", " You can"]
    refusal_token_ids = _single_token_ids(tokenizer, refusal_terms)
    comply_token_ids = _single_token_ids(tokenizer, comply_terms)

    prompts = prompts[: args.max_prompts]
    print(f"Training with {len(prompts)} harmful prompts for {args.epochs} epochs on {device}.")

    step = 0
    for epoch in range(args.epochs):
        for prompt in prompts:
            add_quantization_noise(model, noise_level=args.noise_level)
            total, base, margin = train_step(
                prompt,
                tokenizer,
                model,
                optimizer,
                device,
                refusal_token_ids,
                comply_token_ids,
                margin_weight=args.margin_weight,
                margin_threshold=args.margin_threshold,
                max_length=args.max_length,
            )
            step += 1
            if step % 10 == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step} "
                    f"total={total:.4f} base={base:.4f} margin={margin:.4f}"
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved PAAT model to: {output_dir}")


if __name__ == "__main__":
    main()