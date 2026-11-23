"""
Export the published weights (model/model.pt) to ONNX for the in-browser chat app.

The ONNX graph takes token ids of variable length (up to block_size) and
returns the next-token logits of shape (batch, vocab_size).

Run:
  python export_onnx.py
  python export_onnx.py --output model/model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import config
from checkpoint_io import load_model
from model import GPTLogits


def export_onnx(output: Path, opset: int = 17, check: bool = True) -> Path:
    device = torch.device("cpu")
    model, tok, ckpt, path = load_model(None, device)
    model.eval()
    # Use the same attention path the ONNX graph will use, so the check is fair.
    from model import CausalSelfAttention
    for module in model.modules():
        if isinstance(module, CausalSelfAttention):
            module.force_manual_attention = True
    wrapper = GPTLogits(model)
    wrapper.eval()

    block_size = model.config.block_size
    dummy_len = min(16, block_size)
    dummy = torch.randint(0, model.config.vocab_size, (1, dummy_len), dtype=torch.long)

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting ONNX from {path} -> {output} (opset {opset})")

    # Legacy exporter is more predictable for this small decoder-only model.
    export_kwargs = dict(
        input_names=["tokens"],
        output_names=["logits"],
        dynamic_axes={
            "tokens": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(wrapper, (dummy,), str(output), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(wrapper, (dummy,), str(output), **export_kwargs)

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Wrote {output} ({size_mb:.1f} MB)")

    if check:
        # Compare PyTorch vs ONNX Runtime on a couple of sequence lengths.
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        for length in (1, 8, dummy_len, min(64, block_size)):
            ids = torch.randint(0, model.config.vocab_size, (1, length), dtype=torch.long)
            with torch.no_grad():
                torch_logits = wrapper(ids).numpy()
            ort_logits = sess.run(None, {"tokens": ids.numpy()})[0]
            max_diff = float(np.max(np.abs(torch_logits - ort_logits)))
            print(f"  length {length:3d}: max |pt-onnx| = {max_diff:.5g}")
            if max_diff > 1e-3:
                raise SystemExit(f"ONNX mismatch too large at length {length}: {max_diff}")
        print("ONNX matches PyTorch closely enough.")
        _ = tok  # tokenizer is not needed for the graph, but must exist beside the weights
        _ = ckpt
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export own-gpt to ONNX for the browser.")
    parser.add_argument("--output", type=str, default=str(config.ONNX_PATH))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()
    export_onnx(Path(args.output), opset=args.opset, check=not args.no_check)


if __name__ == "__main__":
    main()
