"""
Generate text from a trained checkpoint.

Examples:
  python generate.py --prompt "Once upon a time"
  python generate.py --prompt "User: Hello\\nAssistant:" --max-tokens 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import config
from checkpoint_io import load_model
from tokenizer import END_OF_TEXT


def generate_text(
    model,
    tok,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    stop_on_eot: bool = True,
) -> str:
    ids = tok.encode(prompt)
    if not ids:
        # Start from the end-of-text token so the model begins a new document.
        ids = tok.encode(END_OF_TEXT)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    stop_id = tok.stoi.get(END_OF_TEXT) if stop_on_eot else None
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_id=stop_id,
    )
    return tok.decode(out[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample text from the mini GPT.")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-tokens", type=int, default=config.DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=config.DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model.pt or a full ckpt.")
    parser.add_argument("--n", type=int, default=1, help="How many samples to draw.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cpu")
    model, tok, ckpt, path = load_model(Path(args.checkpoint) if args.checkpoint else None, device)
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    print(
        f"# checkpoint={path}  step={ckpt.get('step') if isinstance(ckpt, dict) else '?'}  "
        f"val={meta.get('val_loss', ckpt.get('best_val', '?') if isinstance(ckpt, dict) else '?')}  "
        f"params={meta.get('n_params', '?')}",
        file=sys.stderr,
    )

    for i in range(args.n):
        text = generate_text(
            model,
            tok,
            args.prompt,
            args.max_tokens,
            args.temperature,
            args.top_k,
            device,
        )
        if args.n > 1:
            print(f"----- sample {i + 1} -----")
        print(text)
        if args.n > 1:
            print()


if __name__ == "__main__":
    main()
