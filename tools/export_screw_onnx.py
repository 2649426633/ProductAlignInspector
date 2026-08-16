from __future__ import annotations

import argparse
import json
import sys
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the screw/empty classifier checkpoint to ONNX.")
    parser.add_argument("--checkpoint", default="artifacts/screw_classifier/best.pt")
    parser.add_argument("--output", default="artifacts/screw_classifier/screw_classifier.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-ort-check", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    wrapper = ProbabilityWrapper(model).eval()

    input_size = int(checkpoint.get("input_size", 224))
    class_names = list(checkpoint.get("class_names", ["empty", "screw"]))
    dummy = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    # MobileNetV3 is a simple graph. The legacy exporter is intentionally used
    # here because it provides a stable dynamic-batch graph for broad ONNX Runtime
    # compatibility; the exported model is then checked by ONNX and ONNX Runtime.
    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["images"],
        output_names=["probabilities"],
        dynamic_axes={
            "images": {0: "batch"},
            "probabilities": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx

    model_onnx = onnx.load(str(output_path))
    onnx.checker.check_model(model_onnx)

    max_abs_diff = None
    if not args.skip_ort_check:
        import onnxruntime as ort

        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        with torch.no_grad():
            torch_output = wrapper(dummy).cpu().numpy()
        ort_output = session.run(None, {input_name: dummy.cpu().numpy()})[0]
        max_abs_diff = float(np.max(np.abs(torch_output - ort_output)))
        if max_abs_diff > 1e-4:
            raise RuntimeError(
                f"PyTorch/ONNX parity check failed: max_abs_diff={max_abs_diff:.8f}"
            )

    metadata = {
        "schema_version": 1,
        "model_type": "screw_empty_classifier",
        "architecture": str(checkpoint.get("architecture", "mobilenet_v3_small")),
        "onnx_file": output_path.name,
        "input_name": "images",
        "output_name": "probabilities",
        "input_shape": ["N", 3, input_size, input_size],
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
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"ONNX: {output_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Classes: {class_names}")
    print(f"Input: [N, 3, {input_size}, {input_size}]")
    if max_abs_diff is not None:
        print(f"PyTorch/ONNX max abs diff: {max_abs_diff:.8f}")
    print("Export OK.")


if __name__ == "__main__":
    main()
