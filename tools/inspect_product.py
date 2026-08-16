from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.io_utils import read_image, write_image, write_json
from product_align_inspector.roi import crop_roi, validate_roi
from product_align_inspector.screw_classifier import IMAGENET_MEAN, IMAGENET_STD, load_checkpoint


def _pad_to_square(image: np.ndarray, value: int = 255) -> np.ndarray:
    h, w = image.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(value, value, value),
    )


def preprocess_roi(
    image_bgr: np.ndarray,
    *,
    input_size: int,
    mean: list[float] | tuple[float, ...],
    std: list[float] | tuple[float, ...],
) -> np.ndarray:
    """OpenCV preprocessing intentionally matching the exported ONNX/C# contract."""
    image = _pad_to_square(image_bgr, 255)
    image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean_arr) / std_arr
    image = np.transpose(image, (2, 0, 1))
    return np.ascontiguousarray(image[None, ...], dtype=np.float32)


def _draw_label(
    image: np.ndarray,
    roi: list[int] | tuple[int, int, int, int],
    text: str,
    passed: bool,
    uncertain: bool,
) -> None:
    x, y, w, h = map(int, roi)
    if uncertain:
        color = (0, 215, 255)  # yellow/orange in BGR
    elif passed:
        color = (0, 200, 0)
    else:
        color = (0, 0, 255)

    thickness = max(2, int(round(min(image.shape[:2]) / 900)))
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

    font_scale = max(0.55, min(1.2, min(image.shape[:2]) / 1800.0))
    text_thickness = max(1, thickness - 1)
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
    tx = max(0, x)
    ty = max(th + baseline + 4, y - 6)
    cv2.rectangle(
        image,
        (tx, ty - th - baseline - 4),
        (min(image.shape[1] - 1, tx + tw + 8), ty + 3),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (tx + 4, ty - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end aligned screw/empty product inspection.")
    parser.add_argument("--input", required=True, help="Raw production image")
    parser.add_argument("--reference", required=True, help="Canonical reference_aligned.png")
    parser.add_argument("--config", required=True, help="Product ROI config JSON")
    parser.add_argument("--checkpoint", default="artifacts/screw_classifier/best.pt")
    parser.add_argument("--output", default="artifacts/inspection")
    parser.add_argument("--confidence", type=float, default=0.80, help="Below this confidence, slot is treated as uncertain/NG")
    parser.add_argument("--threshold", type=int, default=238, help="Foreground fallback threshold")
    parser.add_argument("--min-inlier-ratio", type=float, default=0.25, help="Minimum accepted SIFT inlier ratio")
    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1")

    input_path = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    reference = read_image(args.reference)
    raw = read_image(input_path)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    print("=== ProductAlignInspector Screw Inspection ===", flush=True)
    print(f"Input: {input_path}", flush=True)

    align_cfg = ProductLocatorConfig(foreground_threshold=args.threshold)
    alignment = align_to_reference(raw, reference, align_cfg)
    aligned = alignment.aligned

    print(
        f"Alignment: {alignment.method}, matches={alignment.feature_matches}, "
        f"inliers={alignment.feature_inliers}, ratio={alignment.feature_inlier_ratio:.1%}, "
        f"ECC={'-' if alignment.ecc_score is None else f'{alignment.ecc_score:.4f}'}",
        flush=True,
    )

    alignment_ok = True
    alignment_reason = "ok"
    if alignment.method.startswith("sift") and alignment.feature_inlier_ratio < args.min_inlier_ratio:
        alignment_ok = False
        alignment_reason = (
            f"weak_alignment: inlier_ratio={alignment.feature_inlier_ratio:.3f} "
            f"< {args.min_inlier_ratio:.3f}"
        )

    model, checkpoint = load_checkpoint(args.checkpoint, device="cpu")
    model.eval()
    input_size = int(checkpoint.get("input_size", 224))
    classes = list(checkpoint.get("class_names", ["empty", "screw"]))
    mean = list(checkpoint.get("mean", IMAGENET_MEAN))
    std = list(checkpoint.get("std", IMAGENET_STD))

    preview = aligned.copy()
    h, w = aligned.shape[:2]
    slot_results: list[dict[str, object]] = []
    all_slots_pass = True

    screw_slots = [slot for slot in config.get("screw_slots", []) if bool(slot.get("enabled", True))]
    if not screw_slots:
        raise RuntimeError("No enabled screw_slots found in product config.")

    print("", flush=True)
    print(f"{'ID':<10} {'Expected':<10} {'Actual':<10} {'Conf':>8} {'Result':<18}", flush=True)
    print("-" * 62, flush=True)

    with torch.inference_mode():
        for slot in screw_slots:
            slot_id = str(slot.get("id", "S?"))
            expected = str(slot.get("expected", "screw"))
            roi = slot.get("roi")
            if roi is None or not validate_roi(roi, w, h):
                raise RuntimeError(f"Invalid ROI {slot_id}: {roi} for aligned image {w}x{h}")

            crop = crop_roi(aligned, roi)
            crop_path = out / "crops" / f"{slot_id}.png"
            write_image(crop_path, crop)

            tensor_np = preprocess_roi(
                crop,
                input_size=input_size,
                mean=mean,
                std=std,
            )
            logits = model(torch.from_numpy(tensor_np))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

            best_index = int(np.argmax(probabilities))
            actual = classes[best_index]
            confidence = float(probabilities[best_index])
            uncertain = confidence < args.confidence
            passed = (actual == expected) and not uncertain and alignment_ok

            if uncertain:
                reason = "Low Confidence"
            elif not alignment_ok:
                reason = "Alignment Uncertain"
            elif expected == "screw" and actual == "empty":
                reason = "Missing Screw"
            elif expected == "empty" and actual == "screw":
                reason = "Extra Screw"
            elif actual != expected:
                reason = "State Mismatch"
            else:
                reason = "OK"

            if not passed:
                all_slots_pass = False

            _draw_label(
                preview,
                roi,
                f"{slot_id} {actual} {confidence:.2f} {reason}",
                passed=passed,
                uncertain=uncertain,
            )

            probs_dict = {name: float(probabilities[i]) for i, name in enumerate(classes)}
            slot_results.append(
                {
                    "id": slot_id,
                    "roi": list(map(int, roi)),
                    "expected": expected,
                    "actual": actual,
                    "confidence": confidence,
                    "probabilities": probs_dict,
                    "passed": passed,
                    "uncertain": uncertain,
                    "reason": reason,
                    "crop": str(crop_path),
                }
            )

            result_text = "PASS" if passed else f"NG:{reason}"
            print(f"{slot_id:<10} {expected:<10} {actual:<10} {confidence:>8.4f} {result_text:<18}", flush=True)

    final_pass = bool(alignment_ok and all_slots_pass)
    final_status = "PASS" if final_pass else "NG"

    banner_color = (0, 160, 0) if final_pass else (0, 0, 255)
    banner_h = max(50, int(round(preview.shape[0] * 0.06)))
    cv2.rectangle(preview, (0, 0), (preview.shape[1], banner_h), banner_color, -1)
    cv2.putText(
        preview,
        f"{final_status}  screw inspection  alignment={alignment.method}",
        (20, int(banner_h * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.8, banner_h / 65.0),
        (255, 255, 255),
        max(2, int(round(banner_h / 30))),
        cv2.LINE_AA,
    )

    write_image(out / "aligned.png", aligned)
    write_image(out / "inspection_preview.png", preview)

    report = {
        "schema_version": 1,
        "product": config.get("product", "unknown"),
        "input": str(input_path),
        "final_status": final_status,
        "passed": final_pass,
        "confidence_threshold": float(args.confidence),
        "alignment": {
            **alignment.to_dict(),
            "accepted": alignment_ok,
            "reason": alignment_reason,
            "min_inlier_ratio": float(args.min_inlier_ratio),
        },
        "screw_classifier": {
            "checkpoint": str(Path(args.checkpoint)),
            "architecture": checkpoint.get("architecture", "mobilenet_v3_small"),
            "input_size": input_size,
            "classes": classes,
            "preprocess_contract": {
                "pad_to_square": True,
                "pad_value_rgb": [255, 255, 255],
                "resize": [input_size, input_size],
                "resize_interpolation": "bilinear/OpenCV INTER_LINEAR",
                "color_order": "RGB",
                "scale": "uint8 / 255.0",
                "mean": mean,
                "std": std,
                "layout": "NCHW",
            },
        },
        "screw_slots": slot_results,
        "spring_regions": {
            "status": "not_evaluated_in_this_phase",
            "configured_count": len(config.get("spring_regions", [])),
        },
    }
    write_json(out / "inspection.json", report)

    print("", flush=True)
    print(f"FINAL: {final_status}", flush=True)
    print(f"Preview: {out / 'inspection_preview.png'}", flush=True)
    print(f"Report:  {out / 'inspection.json'}", flush=True)


if __name__ == "__main__":
    main()
