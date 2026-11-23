"""
Train the mini GPT from scratch on our corpus.

Steps this script takes:
  1. Train (or load) our own BPE tokenizer.
  2. Encode the whole corpus into a compact token file.
  3. Train the transformer with AdamW, cosine learning rate, and eval checks.
  4. Stop when we hit the wall-clock limit or max_iters, and save weights.

The file committed to git is model/model.pt (fp16 weights only, ~25MB).
Full optimizer checkpoints, if saved, stay in checkpoints/ and are gitignored.

Run:
  python train.py
  python train.py --resume                  # continue from model/model.pt or ckpt_last.pt
  python train.py --max-wall-seconds 600
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

import config
from checkpoint_io import (
    default_gpt_config,
    load_raw,
    save_full_checkpoint,
    save_weights,
)
from model import GPT
from tokenizer import BPETokenizer


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def cosine_lr(step: int, warmup: int, max_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / max(warmup, 1)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (lr - min_lr)


def train_or_load_tokenizer(corpus_text: str, force: bool) -> BPETokenizer:
    if config.TOKENIZER_PATH.exists() and not force:
        print(f"Loading tokenizer from {config.TOKENIZER_PATH}")
        tok = BPETokenizer.load(config.TOKENIZER_PATH)
        print(f"  vocab size = {tok.vocab_size}")
        return tok
    if config.LEGACY_TOKENIZER_PATH.exists() and not force:
        print(f"Loading tokenizer from {config.LEGACY_TOKENIZER_PATH}")
        tok = BPETokenizer.load(config.LEGACY_TOKENIZER_PATH)
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        tok.save(config.TOKENIZER_PATH)
        print(f"  vocab size = {tok.vocab_size} (copied to {config.TOKENIZER_PATH})")
        return tok
    print("Training a new BPE tokenizer from scratch (no pretrained vocab)...")
    tok = BPETokenizer()
    tok.train([corpus_text], vocab_size=config.VOCAB_SIZE, verbose=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tok.save(config.TOKENIZER_PATH)
    print(f"Saved tokenizer to {config.TOKENIZER_PATH}")
    return tok


def encode_corpus(tok: BPETokenizer, corpus_text: str, force: bool) -> tuple[np.ndarray, np.ndarray]:
    if config.TRAIN_BIN.exists() and config.VAL_BIN.exists() and not force:
        train_ids = np.memmap(config.TRAIN_BIN, dtype=np.uint16, mode="r")
        val_ids = np.memmap(config.VAL_BIN, dtype=np.uint16, mode="r")
        print(f"Loaded cached tokens: train={len(train_ids):,}  val={len(val_ids):,}")
        return train_ids, val_ids

    print("Encoding corpus with our tokenizer...")
    t0 = time.time()
    ids = tok.encode(corpus_text)
    arr = np.array(ids, dtype=np.uint16)
    n = len(arr)
    n_val = max(int(n * config.VAL_FRACTION), config.BLOCK_SIZE * 4)
    n_val = min(n_val, n // 10)
    # Use the end of the shuffled corpus as a simple hold-out split.
    train = arr[:-n_val]
    val = arr[-n_val:]
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.tofile(config.TRAIN_BIN)
    val.tofile(config.VAL_BIN)
    print(
        f"Encoded {n:,} tokens in {time.time() - t0:.1f}s  "
        f"(train={len(train):,}, val={len(val):,})"
    )
    # Re-open as memmap so we do not keep two copies in RAM.
    train_ids = np.memmap(config.TRAIN_BIN, dtype=np.uint16, mode="r")
    val_ids = np.memmap(config.VAL_BIN, dtype=np.uint16, mode="r")
    return train_ids, val_ids


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: GPT, train_ids, val_ids, batch_size, block_size, device, eval_iters: int) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split, data in (("train", train_ids), ("val", val_ids)):
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def pick_resume_path() -> Path | None:
    """Prefer a full local checkpoint (has optimizer state), else the published weights."""
    if config.CKPT_LAST.exists():
        return config.CKPT_LAST
    if config.MODEL_WEIGHTS.exists():
        return config.MODEL_WEIGHTS
    return None


def resume_into(model: GPT, optimizer: torch.optim.Optimizer, path: Path, device: torch.device) -> tuple[int, float]:
    print(f"Resuming from {path}")
    ckpt = load_raw(path, device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    # Upcast fp16 published weights back to fp32 for training.
    state = {
        k: (v.float() if v.is_floating_point() and v.dtype != torch.float32 else v)
        for k, v in state.items()
    }
    model.load_state_dict(state)
    start_step = int(ckpt.get("step", 0)) if isinstance(ckpt, dict) else 0
    best_val = float(ckpt.get("best_val", float("inf"))) if isinstance(ckpt, dict) else float("inf")
    opt_state = ckpt.get("optimizer") if isinstance(ckpt, dict) else None
    if opt_state:
        try:
            optimizer.load_state_dict(opt_state)
            print(f"  loaded optimizer state, step={start_step}, best_val={best_val:.4f}")
        except Exception as exc:
            print(f"  optimizer state not usable ({exc}); starting the optimizer fresh.")
    else:
        print(f"  weights-only file, step={start_step}, best_val={best_val:.4f}; optimizer starts fresh.")
    return start_step, best_val


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the from-scratch mini GPT.")
    parser.add_argument("--max-wall-seconds", type=int, default=config.MAX_WALL_SECONDS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--max-iters", type=int, default=config.MAX_ITERS)
    parser.add_argument("--fresh-tokenizer", action="store_true", help="Retrain the BPE tokenizer.")
    parser.add_argument("--fresh-tokens", action="store_true", help="Re-encode the corpus.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from checkpoints/ckpt_last.pt or model/model.pt if present.",
    )
    parser.add_argument(
        "--skip-full-ckpt",
        action="store_true",
        help="Do not write the 142MB optimizer checkpoints (recommended on GitHub Actions).",
    )
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "8")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    set_seed(config.SEED)

    if not config.CORPUS_PATH.exists():
        raise SystemExit("No corpus found. Run prepare_data.py first.")

    device = torch.device("cpu")
    print(f"Device: {device}  threads={torch.get_num_threads()}")

    corpus_text = config.CORPUS_PATH.read_text(encoding="utf-8")
    tok = train_or_load_tokenizer(corpus_text, force=args.fresh_tokenizer)
    # If we just retrained the tokenizer, the cached .bin files are stale.
    train_ids, val_ids = encode_corpus(tok, corpus_text, force=args.fresh_tokens or args.fresh_tokenizer)

    gpt_conf = default_gpt_config(vocab_size=tok.vocab_size)
    model = GPT(gpt_conf).to(device)
    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        betas=config.BETAS,
        weight_decay=config.WEIGHT_DECAY,
    )

    start_step = 0
    best_val = float("inf")
    if args.resume:
        resume_path = pick_resume_path()
        if resume_path is None:
            print("No existing model found; starting a fresh run.")
        else:
            start_step, best_val = resume_into(model, optimizer, resume_path, device)

    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_exists = config.LOSS_LOG.exists()
    log_file = open(config.LOSS_LOG, "a", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    if not log_exists:
        writer.writerow(["step", "wall_s", "lr", "train_loss", "val_loss", "tokens"])

    batch_size = args.batch_size
    block_size = config.BLOCK_SIZE
    tokens_per_step = batch_size * block_size
    max_iters = args.max_iters
    wall_limit = args.max_wall_seconds

    print(
        f"Training: batch={batch_size} context={block_size} "
        f"tokens/step={tokens_per_step:,} max_iters={max_iters} "
        f"wall_limit={wall_limit}s"
    )

    t_start = time.time()
    model.train()
    running = 0.0
    running_n = 0
    last_val = None

    def persist(step: int, meta: dict) -> None:
        save_weights(config.MODEL_WEIGHTS, model, step, best_val, meta)
        if not args.skip_full_ckpt:
            save_full_checkpoint(config.CKPT_LAST, model, optimizer, step, best_val, meta)

    def maybe_eval_and_save(step: int, always: bool = False) -> float | None:
        nonlocal best_val, last_val
        if not always and step % config.EVAL_INTERVAL != 0:
            return last_val
        losses = estimate_loss(
            model, train_ids, val_ids, batch_size, block_size, device, config.EVAL_ITERS
        )
        last_val = losses["val"]
        wall = time.time() - t_start
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"step {step:5d}  wall {wall:6.1f}s  lr {lr_now:.2e}  "
            f"train {losses['train']:.4f}  val {losses['val']:.4f}"
        )
        writer.writerow(
            [
                step,
                f"{wall:.1f}",
                f"{lr_now:.6e}",
                f"{losses['train']:.6f}",
                f"{losses['val']:.6f}",
                step * tokens_per_step,
            ]
        )
        log_file.flush()
        meta = {
            "n_params": n_params,
            "tokens_per_step": tokens_per_step,
            "vocab_size": tok.vocab_size,
            "wall_s": wall,
            "train_loss": losses["train"],
            "val_loss": losses["val"],
        }
        persist(step, meta)
        if losses["val"] < best_val:
            best_val = losses["val"]
            meta = {**meta, "best_val": best_val}
            save_weights(config.MODEL_WEIGHTS, model, step, best_val, meta)
            if not args.skip_full_ckpt:
                save_full_checkpoint(config.CKPT_BEST, model, optimizer, step, best_val, meta)
            print(f"  new best val {best_val:.4f} -> {config.MODEL_WEIGHTS}")
        return last_val

    maybe_eval_and_save(start_step, always=True)

    step = start_step
    try:
        while step < max_iters:
            if time.time() - t_start >= wall_limit:
                print(f"Reached wall-clock limit ({wall_limit}s) at step {step}.")
                break

            lr = cosine_lr(step, config.WARMUP_ITERS, max_iters, config.LEARNING_RATE, config.MIN_LR)
            for g in optimizer.param_groups:
                g["lr"] = lr

            x, y = get_batch(train_ids, batch_size, block_size, device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()

            running += float(loss.item())
            running_n += 1
            step += 1

            if step % config.LOG_INTERVAL == 0:
                wall = time.time() - t_start
                avg = running / max(running_n, 1)
                ms = (wall / max(step - start_step, 1)) * 1000 if step else 0
                print(
                    f"step {step:5d}  wall {wall:6.1f}s  {ms:.0f} ms/step  "
                    f"lr {lr:.2e}  loss {avg:.4f}  "
                    f"~{(step * tokens_per_step)/1e6:.2f}M tokens"
                )
                running = 0.0
                running_n = 0

            if step % config.EVAL_INTERVAL == 0 or step % config.CKPT_INTERVAL == 0:
                maybe_eval_and_save(step)
    except KeyboardInterrupt:
        print(f"Interrupted at step {step}. Saving...")

    maybe_eval_and_save(step, always=True)
    persist(
        step,
        {
            "n_params": n_params,
            "tokens_per_step": tokens_per_step,
            "vocab_size": tok.vocab_size,
            "wall_s": time.time() - t_start,
            "best_val": best_val,
            "val_loss": last_val if last_val is not None else best_val,
            "finished": True,
        },
    )
    log_file.close()
    wall = time.time() - t_start
    print(
        f"Done. steps={step}  wall={wall:.1f}s  best_val={best_val:.4f}  "
        f"tokens_seen={step * tokens_per_step:,}"
    )
    print(f"Published weights: {config.MODEL_WEIGHTS}")
    if not args.skip_full_ckpt:
        print(f"Full checkpoints: {config.CKPT_LAST}  {config.CKPT_BEST}")
    print(f"Loss log: {config.LOSS_LOG}")


if __name__ == "__main__":
    main()
