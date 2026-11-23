"""
Chat with the trained mini GPT in a simple User / Assistant format.

Interactive:
  python chat.py

One-shot (useful for tests, scripts, and the GitHub Actions chat workflow):
  python chat.py --message "Hello"
  python chat.py --turns "Hi" "What is 2 plus 2?" "Tell me a short story about a cat"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

import config
from checkpoint_io import load_model
from generate import generate_text
from tokenizer import END_OF_TEXT


def read_user_line(prompt: str = "User: ") -> str:
    """Read a line, flushing prompts so pipe/file input works cleanly."""
    if sys.stdin.isatty():
        return input(prompt).strip()
    print(prompt, end="", flush=True)
    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    # Echo the typed line when stdin is not a terminal, so logs are readable.
    print(line, end="", flush=True)
    return line.strip()


STOP_RE = re.compile(r"\nUser:|\n\nUser:|" + re.escape(END_OF_TEXT))


def build_prompt(history: list[tuple[str, str]], user_message: str) -> str:
    """
    history is a list of (user, assistant) turns already completed.
    We always end the prompt on 'Assistant:' so the model writes the reply.
    """
    parts: list[str] = []
    for u, a in history:
        parts.append(f"User: {u}")
        parts.append(f"Assistant: {a}")
    parts.append(f"User: {user_message}")
    parts.append("Assistant:")
    return "\n".join(parts)


def extract_reply(full_text: str, prompt: str) -> str:
    """Take the model's continuation after the prompt and stop at the next turn."""
    if full_text.startswith(prompt):
        reply = full_text[len(prompt) :]
    else:
        # Decode/encode can occasionally change whitespace at the seam.
        reply = full_text.split("Assistant:")[-1]
    reply = STOP_RE.split(reply, maxsplit=1)[0]
    reply = reply.strip()
    # The model sometimes repeats "Assistant:" at the start.
    if reply.lower().startswith("assistant:"):
        reply = reply[len("assistant:") :].strip()
    return reply


def reply_once(model, tok, history, user_message, args, device) -> str:
    prompt = build_prompt(history, user_message)
    full = generate_text(
        model,
        tok,
        prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        stop_on_eot=True,
    )
    return extract_reply(full, prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the from-scratch mini GPT.")
    parser.add_argument("--message", type=str, default=None, help="Send one message and exit.")
    parser.add_argument("--turns", nargs="+", default=None, help="A list of user messages, in order.")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=config.DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cpu")
    model, tok, ckpt, path = load_model(Path(args.checkpoint) if args.checkpoint else None, device)
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    step = ckpt.get("step") if isinstance(ckpt, dict) else "?"
    banner = (
        f"own-gpt chat  (step {step}, "
        f"{meta.get('n_params', '?')} params, val {meta.get('val_loss', ckpt.get('best_val', '?') if isinstance(ckpt, dict) else '?')}, "
        f"file {path.name})"
    )

    history: list[tuple[str, str]] = []

    def handle(user_message: str) -> str:
        assistant = reply_once(model, tok, history, user_message, args, device)
        if not assistant:
            assistant = "..."
        history.append((user_message, assistant))
        # Keep the rolling context short so we stay inside block_size.
        # A few turns is plenty for this small model.
        if len(history) > 4:
            del history[0]
        return assistant

    if args.turns:
        print(banner, flush=True)
        for msg in args.turns:
            print(f"User: {msg}")
            print(f"Assistant: {handle(msg)}")
            print()
        return

    if args.message is not None:
        print(handle(args.message))
        return

    print(banner)
    print("Type a message and press Enter. Commands: /reset  /exit", flush=True)
    print()
    while True:
        try:
            user_message = read_user_line("User: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.", flush=True)
            break
        if not user_message:
            continue
        if user_message in {"/exit", "/quit", "exit", "quit"}:
            print("Goodbye.", flush=True)
            break
        if user_message == "/reset":
            history.clear()
            print("(chat history cleared)", flush=True)
            continue
        try:
            assistant = handle(user_message)
        except Exception as exc:  # keep the REPL alive
            print(f"(error: {exc})", file=sys.stderr)
            continue
        print(f"Assistant: {assistant}", flush=True)
        print()


if __name__ == "__main__":
    main()
