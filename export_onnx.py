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

    import numpy as np
    import onnxruntime as ort

    def _compare(onnx_path: Path, tag: str, limit: float) -> None:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        for length in (1, 8, dummy_len, min(64, block_size)):
            ids = torch.randint(0, model.config.vocab_size, (1, length), dtype=torch.long)
            with torch.no_grad():
                torch_logits = wrapper(ids).numpy()
            ort_logits = sess.run(None, {"tokens": ids.numpy()})[0].astype(np.float32)
            max_diff = float(np.max(np.abs(torch_logits - ort_logits)))
            agree = float(np.mean(np.argmax(torch_logits, axis=-1) == np.argmax(ort_logits, axis=-1)))
            print(f"  {tag} length {length:3d}: max |pt-onnx| = {max_diff:.5g}  argmax_agree={agree:.3f}")
            if max_diff > limit:
                raise SystemExit(f"ONNX {tag} mismatch too large at length {length}: {max_diff}")

    if check:
        _compare(output, "fp32", 1e-3)
        print("ONNX fp32 matches PyTorch closely enough.")

    # Convert to fp16 for the browser/APK (~half the size, much easier on phones).
    # Keep int64 inputs and float32 logits so the JS runtime does not change.
    try:
        import onnx
        from onnxconverter_common import float16 as _f16
    except ImportError:
        print("onnxconverter-common not installed; leaving a float32 ONNX file.")
    else:
        fp32_model = onnx.load(str(output))
        fp16_model = _f16.convert_float_to_float16(fp32_model, keep_io_types=True)
        onnx.save(fp16_model, str(output))
        size_mb = output.stat().st_size / (1024 * 1024)
        print(f"Converted to fp16 -> {output} ({size_mb:.1f} MB)")
        if check:
            _compare(output, "fp16", 2e-2)
            print("ONNX fp16 is close enough to PyTorch (argmax should match).")

    _ = tok
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
