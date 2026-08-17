from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.anomaly.dinov2_adapter import IMAGENET_MEAN, IMAGENET_STD
from product_align_inspector.anomaly.roi_patchcore import load_roi_model, read_model_manifest
from product_align_inspector.decision_rules import load_decision_multipliers, roi_multiplier


def group_of(roi_id: str) -> str:
    rid = roi_id.upper()
    if rid.startswith("SPRING"):
        return "SPRING"
    if rid.startswith("E"):
        return "EMPTY"
    if rid.startswith("S"):
        return "SCREW"
    return "OTHER"


def resolve_from_repo(text: str | None) -> Path | None:
    if not text:
        return None
    path = Path(text)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> None:
    p = argparse.ArgumentParser(description="Export ONNX/BIN/JSON runtime bundle for C#/.NET inspection.")
    p.add_argument("--model-dir", required=True, help="Python ROI DINO/PatchCore model directory")
    p.add_argument("--onnx", required=True, help="Exported DINOv2 patch-token ONNX")
    p.add_argument("--reference", help="Override canonical aligned reference image")
    p.add_argument("--decision-rules", default="configs/brunei_decision_rules.json")
    p.add_argument("--threshold-scale", type=float, default=1.0)
    p.add_argument("--output", default="artifacts/runtime_bundle")
    args = p.parse_args()

    if args.threshold_scale <= 0:
        raise SystemExit("--threshold-scale must be > 0")

    model_dir = Path(args.model_dir).resolve()
    onnx_source = Path(args.onnx).resolve()
    output = Path(args.output).resolve()
    banks_dir = output / "banks"
    reference_dir = output / "reference"
    output.mkdir(parents=True, exist_ok=True)
    banks_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    if not onnx_source.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_source}")

    manifest = read_model_manifest(model_dir)
    dino_meta = manifest.get("dino", {})
    preprocess_meta = dino_meta.get("preprocess", {})
    image_size = int(dino_meta.get("image_size", 224))
    embedding_dim = int(dino_meta.get("embedding_dim", 384))
    patch_size = 14
    patch_grid = image_size // patch_size
    patch_tokens = patch_grid * patch_grid

    reference_source = Path(args.reference or manifest["reference_image"]).resolve()
    if not reference_source.is_file():
        raise FileNotFoundError(f"Reference image not found: {reference_source}")

    rules_path = resolve_from_repo(args.decision_rules)
    default_multiplier, roi_multipliers = load_decision_multipliers(rules_path)

    all_models = [load_roi_model(model_dir, str(item["id"])) for item in manifest.get("rois", [])]
    models = [m for m in all_models if group_of(m.roi_id) in {"SCREW", "EMPTY"}]
    if not models:
        raise RuntimeError("No S/E ROI models found")

    onnx_dest = output / "dinov2_vits14_patchtokens.onnx"
    if onnx_source != onnx_dest:
        shutil.copy2(onnx_source, onnx_dest)
    reference_dest = reference_dir / "reference.png"
    shutil.copy2(reference_source, reference_dest)

    rois: list[dict[str, object]] = []
    total_bank_bytes = 0
    for model in models:
        group = group_of(model.roi_id)
        multiplier = roi_multiplier(model.roi_id, default_multiplier, roi_multipliers)
        base_threshold = None if model.threshold is None else float(model.threshold)
        decision_threshold = None
        if base_threshold is not None:
            decision_threshold = base_threshold * float(args.threshold_scale) * float(multiplier)

        memory = np.asarray(model.memory, dtype="<f4", order="C")
        if memory.ndim != 2 or memory.shape[1] != model.feature_dim:
            raise RuntimeError(f"Unexpected memory shape for {model.roi_id}: {memory.shape}")
        if model.feature_dim != embedding_dim:
            raise RuntimeError(
                f"Feature dimension mismatch for {model.roi_id}: {model.feature_dim} != {embedding_dim}"
            )

        bank_file = banks_dir / f"{model.roi_id}.bin"
        memory.tofile(bank_file)
        bank_bytes = int(bank_file.stat().st_size)
        expected_bytes = int(memory.shape[0] * memory.shape[1] * 4)
        if bank_bytes != expected_bytes:
            raise RuntimeError(
                f"BIN byte-size mismatch for {model.roi_id}: {bank_bytes} != {expected_bytes}"
            )
        total_bank_bytes += bank_bytes

        expected = "SCREW" if group == "SCREW" else "EMPTY"
        defect_meaning = "MISSING_OR_WRONG_SCREW" if group == "SCREW" else "UNEXPECTED_SCREW_OR_OBJECT"
        rois.append(
            {
                "id": model.roi_id,
                "group": group,
                "expected": expected,
                "defect_meaning": defect_meaning,
                "roi": list(model.roi),
                "bank_file": f"banks/{model.roi_id}.bin",
                "bank_format": "float32_le_row_major",
                "bank_rows": int(memory.shape[0]),
                "feature_dim": int(memory.shape[1]),
                "base_threshold": base_threshold,
                "threshold_multiplier": float(multiplier),
                "global_threshold_scale": float(args.threshold_scale),
                "decision_threshold": decision_threshold,
                "score_top_fraction": float(model.score_top_fraction),
                "patch_grid": int(model.patch_grid),
            }
        )

    runtime = {
        "schema_version": 1,
        "runtime_type": "product_align_dinov2_patchcore",
        "product_rules": {
            "S": "must_have_screw",
            "E": "must_be_empty",
            "SPRING": "ignored_for_current_product_decision",
        },
        "reference_image": "reference/reference.png",
        "alignment": {
            "implementation": "SIFT_RANSAC_ECC_STAGED_RECOVERY",
            "note": "C# must reproduce ProductAlignInspector alignment.py behavior before ROI cropping.",
        },
        "dino": {
            "model_file": "dinov2_vits14_patchtokens.onnx",
            "input_name": "input",
            "output_name": "patch_tokens",
            "input_dtype": "float32",
            "input_layout": "NCHW",
            "input_shape": ["B", 3, image_size, image_size],
            "output_dtype": "float32",
            "output_shape": ["B", patch_tokens, embedding_dim],
            "patch_size": patch_size,
            "patch_grid": patch_grid,
            "embedding_dim": embedding_dim,
            "output_l2_normalized": True,
            "preprocess": {
                "source_color": "BGR",
                "gray_to_bgr": True,
                "pad_to_square": True,
                "pad_value": int(preprocess_meta.get("pad_value", dino_meta.get("pad_value", 255))),
                "resize": [image_size, image_size],
                "resize_interpolation": "INTER_LINEAR",
                "convert_color": "BGR_TO_RGB",
                "scale": 1.0 / 255.0,
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
                "layout": "NCHW",
                "dtype": "float32",
            },
        },
        "scoring": {
            "distance": "1 - max(query_token_dot_memory_row)",
            "query_l2_normalized": True,
            "memory_l2_normalized": True,
            "score": "mean(top ceil(patch_count * score_top_fraction) distances)",
            "decision": "score > decision_threshold => NG",
        },
        "decision_roi_count": len(rois),
        "rois": rois,
    }

    runtime_path = output / "runtime.json"
    runtime_path.write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Runtime Bundle Export ===", flush=True)
    print(f"Output:        {output}", flush=True)
    print(f"ONNX:          {onnx_dest.name}", flush=True)
    print(f"Reference:     {reference_dest.relative_to(output)}", flush=True)
    print(f"Decision ROIs: {len(rois)}", flush=True)
    print(f"Banks:         {total_bank_bytes / (1024.0 * 1024.0):.2f} MB", flush=True)
    print(f"Runtime JSON:  {runtime_path}", flush=True)
    for item in rois:
        print(
            f"  {item['id']:<4} rows={item['bank_rows']:<5} dim={item['feature_dim']} "
            f"threshold={item['decision_threshold']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
