from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.screw_classifier import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ProbabilityWrapper,
    load_checkpoint,
)


def _step(message: str) -> float:
    print(message, flush=True)
    return time.perf_counter()


def _done(start: float, suffix: str = "OK") -> None:
    print(f"    {suffix} ({time.perf_counter() - start:.2f}s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the screw/empty classifier checkpoint to ONNX.")
    parser.add_argument("--checkpoint", default="artifacts/screw_classifier/best.pt")
    parser.add_argument("--output", default="artifacts/screw_classifier/screw_classifier.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-ort-check", action="store_true")
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export dynamic batch dimension. Default is fixed batch=1, which is simpler for WinForms single-ROI inference.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== ProductAlignInspector ONNX Export ===", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"Opset: {args.opset}", flush=True)
    print(f"Dynamic batch: {args.dynamic_batch}", flush=True)

    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    t = _step("[1/7] Loading checkpoint on CPU...")
    model, checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    wrapper = ProbabilityWrapper(model).eval()
    _done(t)

    input_size = int(checkpoint.get("input_size", 224))
    class_names = list(checkpoint.get("class_names", ["empty", "screw"]))
    dummy = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    t = _step("[2/7] Running a PyTorch dry forward pass...")
    with torch.inference_mode():
        torch_output = wrapper(dummy).cpu().numpy()
    if torch_output.shape != (1, len(class_names)):
        raise RuntimeError(f"Unexpected PyTorch output shape: {torch_output.shape}")
    _done(t, f"OK shape={torch_output.shape}")

    t = _step("[3/7] Exporting ONNX with TorchScript/legacy exporter...")
    export_kwargs: dict[str, object] = {
        "input_names": ["images"],
        "output_names": ["probabilities"],
        "opset_version": args.opset,
        "dynamo": False,
        "verbose": False,
    }
    if args.dynamic_batch:
        export_kwargs["dynamic_axes"] = {
            "images": {0: "batch"},
            "probabilities": {0: "batch"},
        }

    # For the production WinForms path we default to fixed batch=1. A fixed
    # graph is enough because one configured ROI is classified at a time and
    # avoids unnecessary dynamic-shape complexity during export/runtime.
    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        **export_kwargs,
    )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("torch.onnx.export returned but the ONNX file is missing/empty")
    _done(t, f"OK size={output_path.stat().st_size / (1024 * 1024):.2f} MB")

    t = _step("[4/7] Importing ONNX package...")
    import onnx
    _done(t)

    t = _step("[5/7] Loading and checking ONNX graph...")
    model_onnx = onnx.load(str(output_path))
    onnx.checker.check_model(model_onnx)
    _done(t)

    max_abs_diff = None
    if args.skip_ort_check:
        print("[6/7] ONNX Runtime parity check: SKIPPED by --skip-ort-check", flush=True)
    else:
        t = _step("[6/7] Creating ONNX Runtime CPU session and checking parity...")
        import onnxruntime as ort

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(output_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        ort_output = session.run(None, {input_name: dummy.cpu().numpy()})[0]
        max_abs_diff = float(np.max(np.abs(torch_output - ort_output)))
        if max_abs_diff > 1e-4:
            raise RuntimeError(
                f"PyTorch/ONNX parity check failed: max_abs_diff={max_abs_diff:.8f}"
            )
        _done(t, f"OK max_abs_diff={max_abs_diff:.8f}")

    t = _step("[7/7] Writing WinForms metadata JSON...")
    metadata = {
        "schema_version": 1,
        "model_type": "screw_empty_classifier",
        "architecture": str(checkpoint.get("architecture", "mobilenet_v3_small")),
        "onnx_file": output_path.name,
        "input_name": "images",
        "output_name": "probabilities",
        "input_shape": (["N", 3, input_size, input_size] if args.dynamic_batch else [1, 3, input_size, input_size]),
        "dynamic_batch": bool(args.dynamic_batch),
        "input_size": input_size,
        "input_dtype": "float32",
        "color_order": "RGB",
        "preprocess": {
            "pad_to_square": True,
            "pad_value_rgb": [255, 255, 255],
            "resize": [input_size, input_size],
            "scale": "uint8 / 255.0",
            "mean": list(checkpoint.get("mean", IMAGENET_MEAN)),
            "std": list(checkpoint.get("std", IMAGENET_STD)),
            "layout": "NCHW",
        },
        "classes": class_names,
        "output_semantics": "softmax probabilities in the same order as classes",
        "checkpoint_metrics": checkpoint.get("metrics", {}),
        "pytorch_onnx_max_abs_diff": max_abs_diff,
        "opset": args.opset,
        "exporter": "torchscript_legacy",
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _done(t)

    print("", flush=True)
    print(f"ONNX: {output_path}", flush=True)
    print(f"Metadata: {metadata_path}", flush=True)
    print(f"Classes: {class_names}", flush=True)
    if args.dynamic_batch:
        print(f"Input: [N, 3, {input_size}, {input_size}]", flush=True)
    else:
        print(f"Input: [1, 3, {input_size}, {input_size}]", flush=True)
    if max_abs_diff is not None:
        print(f"PyTorch/ONNX max abs diff: {max_abs_diff:.8f}", flush=True)
    print("Export OK.", flush=True)


if __name__ == "__main__":
    main()
