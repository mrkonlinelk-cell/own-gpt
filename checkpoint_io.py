"""
Load and save model files.

Two formats are used:

- Full training checkpoint (checkpoints/ckpt_*.pt): model + optimizer + step.
  These are around 142MB and are not committed to git (GitHub's file limit is 100MB).
- Weights-only file (model/model.pt): model weights in fp16, plus config and
  a little metadata. Around 25MB, safe to commit. train.py can fine-tune from it
  (the optimizer starts fresh). generate.py / chat.py / export_onnx.py load it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

import config
from model import GPT, GPTConfig
from tokenizer import BPETokenizer


def default_gpt_config(vocab_size: int | None = None) -> GPTConfig:
    return GPTConfig(
        vocab_size=vocab_size or config.VOCAB_SIZE,
        block_size=config.BLOCK_SIZE,
        n_layer=config.N_LAYER,
        n_head=config.N_HEAD,
        n_embd=config.N_EMBD,
        dropout=config.DROPOUT,
        bias=config.BIAS,
    )


def _float32_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            out[key] = tensor.float()
        else:
            out[key] = tensor
    return out


def fp16_state_dict(model: GPT) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, tensor in model.state_dict().items():
        cpu = tensor.detach().cpu().contiguous()
        if cpu.is_floating_point():
            out[key] = cpu.half()
        else:
            out[key] = cpu
    return out


def save_full_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    best_val: float,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "step": step,
        "best_val": best_val,
        "config": model.config.__dict__,
        "meta": meta,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def save_weights(path: Path, model: GPT, step: int, best_val: float, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": fp16_state_dict(model),
        "step": step,
        "best_val": best_val,
        "config": model.config.__dict__,
        "meta": meta,
        "weights_only": True,
        "dtype": "fp16",
    }
    torch.save(payload, path)
    write_meta(step=step, best_val=best_val, meta=meta, config_dict=model.config.__dict__)


def write_meta(*, step: int, best_val: float, meta: dict[str, Any], config_dict: dict[str, Any]) -> None:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "step": step,
        "best_val": best_val,
        "n_params": meta.get("n_params"),
        "val_loss": meta.get("val_loss", meta.get("best_val", best_val)),
        "train_loss": meta.get("train_loss"),
        "vocab_size": meta.get("vocab_size", config_dict.get("vocab_size")),
        "block_size": config_dict.get("block_size", config.BLOCK_SIZE),
        "n_layer": config_dict.get("n_layer", config.N_LAYER),
        "n_head": config_dict.get("n_head", config.N_HEAD),
        "n_embd": config_dict.get("n_embd", config.N_EMBD),
        "dtype": "fp16",
    }
    config.MODEL_META.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_raw(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def load_model_from_payload(ckpt: dict[str, Any], device: torch.device) -> tuple[GPT, dict[str, Any]]:
    if "config" in ckpt:
        conf = GPTConfig(**ckpt["config"])
    else:
        conf = default_gpt_config()
    if "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    model = GPT(conf).to(device)
    model.load_state_dict(_float32_state(state))
    return model, ckpt


def find_tokenizer_path() -> Path:
    if config.TOKENIZER_PATH.exists():
        return config.TOKENIZER_PATH
    if config.LEGACY_TOKENIZER_PATH.exists():
        return config.LEGACY_TOKENIZER_PATH
    raise FileNotFoundError(
        f"No tokenizer at {config.TOKENIZER_PATH}. Train first, or copy tokenizer.json into model/."
    )


def resolve_weights_path(preferred: str | Path | None = None) -> Path:
    if preferred is not None:
        path = Path(preferred)
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint at {path}")
        return path
    for path in (config.MODEL_WEIGHTS, config.CKPT_BEST, config.CKPT_LAST):
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No model weights at {config.MODEL_WEIGHTS}. Train the model first with train.py."
    )


def load_model(ckpt_path: Path | None, device: torch.device) -> tuple[GPT, BPETokenizer, dict[str, Any], Path]:
    path = resolve_weights_path(ckpt_path)
    tok = BPETokenizer.load(find_tokenizer_path())
    ckpt = load_raw(path, device)
    model, ckpt = load_model_from_payload(ckpt, device)
    model.eval()
    return model, tok, ckpt, path
