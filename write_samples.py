"""
Write a few prompt completions and chat exchanges to samples/latest.md.

Used by the GitHub Actions training workflow so you can see what the
latest model says without downloading anything.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch

import config
from chat import build_prompt, extract_reply
from checkpoint_io import load_model
from generate import generate_text


PROMPTS = [
    "Once upon a time",
    "Once upon a time, a little girl",
    "There was a small dog",
    "The little boy found",
]

CHAT_TURNS = [
    "Hello",
    "What is 2 plus 2?",
    "Tell me a short story about a cat",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write sample generations to samples/latest.md")
    parser.add_argument("--output", type=str, default=str(config.SAMPLES_DIR / "latest.md"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=config.DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    args = parser.parse_args()

    device = torch.device("cpu")
    model, tok, ckpt, path = load_model(None, device)
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    step = ckpt.get("step") if isinstance(ckpt, dict) else "?"
    val = meta.get("val_loss", ckpt.get("best_val", "?") if isinstance(ckpt, dict) else "?")
    n_params = meta.get("n_params", model.count_parameters())

    lines: list[str] = []
    lines.append("# Latest sample outputs")
    lines.append("")
    try:
        rel = path.resolve().relative_to(config.ROOT)
    except ValueError:
        rel = path.name
    lines.append(f"Generated after a training run. Checkpoint: `{rel}` (step {step}).")
    lines.append("")
    lines.append(f"- Parameters: {n_params}")
    lines.append(f"- Val loss / best val: {val}")
    lines.append(f"- Sampling: temperature {args.temperature}, top-k {args.top_k}, max tokens {args.max_tokens}")
    lines.append(f"- Time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z")
    lines.append("")
    lines.append("The writing is often rough. That is expected for a small model.")
    lines.append("")
    lines.append("## Prompt completions")
    lines.append("")

    for i, prompt in enumerate(PROMPTS, start=1):
        torch.manual_seed(args.seed + i)
        text = generate_text(
            model, tok, prompt, args.max_tokens, args.temperature, args.top_k, device
        )
        lines.append(f"### {i}. `{prompt}`")
        lines.append("")
        lines.append("```")
        lines.append(text.strip())
        lines.append("```")
        lines.append("")

    lines.append("## Chat exchanges")
    lines.append("")
    torch.manual_seed(args.seed + 20)
    history: list[tuple[str, str]] = []
    for msg in CHAT_TURNS:
        prompt = build_prompt(history, msg)
        full = generate_text(
            model, tok, prompt, args.max_tokens, args.temperature, args.top_k, device, stop_on_eot=True
        )
        reply = extract_reply(full, prompt) or "..."
        history.append((msg, reply))
        lines.append(f"**User:** {msg}")
        lines.append("")
        lines.append(f"**Assistant:** {reply}")
        lines.append("")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
