from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.anomaly.dinov2_adapter import DINOv2Adapter, DINOv2Config
from product_align_inspector.anomaly.roi_patchcore import (
    load_roi_model,
    read_model_manifest,
    score_patch_tokens,
)
from product_align_inspector.io_utils import read_image, write_image, write_json
from product_align_inspector.roi import crop_roi, validate_roi


def _draw_result(
    canvas: np.ndarray,
    roi: tuple[int, int, int, int],
    text: str,
    status: str,
) -> None:
    x, y, w, h = roi
    if status == "PASS":
        color = (0, 190, 0)
    elif status == "NG":
        color = (0, 0, 255)
    else:
        color = (0, 200, 255)

    thickness = max(2, int(round(min(canvas.shape[:2]) / 900)))
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
    font_scale = max(0.55, min(1.1, min(canvas.shape[:2]) / 1800.0))
    text_thickness = max(1, thickness - 1)
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
    tx = max(0, x)
    ty = max(th + baseline + 4, y - 6)
    cv2.rectangle(
        canvas,
        (tx, ty - th - baseline - 4),
        (min(canvas.shape[1] - 1, tx + tw + 8), ty + 3),
        color,
        -1,
    )
    cv2.putText(
        canvas,
        text,
        (tx + 4, ty - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def _save_heatmap(crop: np.ndarray, anomaly_map: np.ndarray, threshold: float | None, output: Path) -> None:
    h, w = crop.shape[:2]
    up = cv2.resize(anomaly_map.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    scale = float(threshold) if threshold is not None and threshold > 1e-12 else float(np.max(up))
    scale = max(scale, 1e-8)
    normalized = np.clip(up / scale, 0.0, 1.0)
    heat_u8 = np.round(normalized * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(crop, 0.58, heat, 0.42, 0.0)
    write_image(output, overlay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect fixed ROIs with GOOD-only DINOv2/PatchCore memory banks.")
    parser.add_argument("--input", required=True, help="Raw full-resolution production image")
    parser.add_argument("--model-dir", default="artifacts/roi_dino_patchcore")
    parser.add_argument("--reference", help="Override reference image path saved in model.json")
    parser.add_argument("--dino-repo", help="Override local DINOv2 repository")
    parser.add_argument("--dino-weights", help="Override DINOv2 weights")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="artifacts/roi_anomaly_test")
    parser.add_argument("--threshold-scale", type=float, default=1.0, help="Multiply all calibrated ROI thresholds")
    args = parser.parse_args()

    if args.threshold_scale <= 0:
        raise SystemExit("--threshold-scale must be > 0")

    input_path = Path(args.input)
    model_dir = Path(args.model_dir)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    manifest = read_model_manifest(model_dir)
    reference_path = Path(args.reference or manifest["reference_image"])
    dino_meta = manifest["dino"]
    align_meta = manifest.get("alignment", {})

    dino_cfg = DINOv2Config(
        model_name=str(dino_meta.get("model_name", "dinov2_vits14")),
        image_size=int(dino_meta.get("image_size", 224)),
        embedding_dim=int(dino_meta.get("embedding_dim", 384)),
        repo_dir=str(args.dino_repo or dino_meta.get("repo_dir", "third_party/dinov2")),
        weights_path=str(args.dino_weights or dino_meta.get("weights_path", "weights/dinov2_vits14_pretrain.pth")),
        pad_value=int(dino_meta.get("preprocess", {}).get("pad_value", 255)),
    )

    models = [load_roi_model(model_dir, str(item["id"])) for item in manifest.get("rois", [])]
    if not models:
        raise RuntimeError(f"No ROI banks listed in {model_dir / 'model.json'}")

    print("=== ROI DINOv2 / PatchCore Inspection ===", flush=True)
    print(f"Input:      {input_path}", flush=True)
    print(f"Model dir:  {model_dir}", flush=True)
    print(f"ROI IDs:    {[m.roi_id for m in models]}", flush=True)

    t_total = time.perf_counter()
    raw = read_image(input_path)
    reference = read_image(reference_path)

    t0 = time.perf_counter()
    alignment = align_to_reference(
        raw,
        reference,
        ProductLocatorConfig(
            foreground_threshold=int(align_meta.get("foreground_threshold", 238))
        ),
    )
    alignment_seconds = time.perf_counter() - t0
    min_inlier_ratio = float(align_meta.get("min_inlier_ratio", 0.25))
    alignment_ok = not (
        alignment.method.startswith("sift")
        and alignment.feature_inlier_ratio < min_inlier_ratio
    )
    print(
        f"Alignment: {alignment.method}, matches={alignment.feature_matches}, "
        f"inliers={alignment.feature_inliers}, ratio={alignment.feature_inlier_ratio:.1%}, "
        f"ECC={'-' if alignment.ecc_score is None else f'{alignment.ecc_score:.4f}'}, "
        f"time={alignment_seconds:.3f}s",
        flush=True,
    )

    aligned = alignment.aligned
    h, w = aligned.shape[:2]
    crops: list[np.ndarray] = []
    for model in models:
        if not validate_roi(model.roi, w, h):
            raise RuntimeError(f"Invalid saved ROI {model.roi_id}: {model.roi} for aligned {w}x{h}")
        crop = crop_roi(aligned, model.roi)
        crops.append(crop)
        write_image(out / "crops" / f"{model.roi_id}.png", crop)

    dino = DINOv2Adapter(device=args.device, config=dino_cfg, project_root=REPO_ROOT)
    dino.load()

    t0 = time.perf_counter()
    tokens_batch = dino.patch_tokens_batch(crops)
    dino_seconds = time.perf_counter() - t0

    preview = aligned.copy()
    results: list[dict[str, object]] = []
    scoring_start = time.perf_counter()

    print("", flush=True)
    print(f"{'ROI':<14} {'Score':>10} {'Threshold':>10} {'MaxPatch':>10} {'Result':<10}", flush=True)
    print("-" * 62, flush=True)

    any_ng = False
    any_unknown = False
    for model, crop, tokens in zip(models, crops, tokens_batch):
        score, anomaly_map, stats = score_patch_tokens(
            tokens,
            model.memory,
            patch_grid=model.patch_grid,
            top_fraction=model.score_top_fraction,
        )
        threshold = None if model.threshold is None else float(model.threshold) * args.threshold_scale

        if not alignment_ok:
            status = "NG"
            reason = "Alignment Uncertain"
            any_ng = True
        elif threshold is None:
            status = "UNKNOWN"
            reason = "Threshold Not Calibrated"
            any_unknown = True
        elif score > threshold:
            status = "NG"
            reason = "ROI Anomaly"
            any_ng = True
        else:
            status = "PASS"
            reason = "OK"

        threshold_text = "-" if threshold is None else f"{threshold:.6f}"
        print(
            f"{model.roi_id:<14} {score:>10.6f} {threshold_text:>10} "
            f"{stats['max']:>10.6f} {status:<10}",
            flush=True,
        )

        _draw_result(
            preview,
            model.roi,
            f"{model.roi_id} {status} s={score:.3f}" + ("" if threshold is None else f" t={threshold:.3f}"),
            status,
        )
        _save_heatmap(crop, anomaly_map, threshold, out / "heatmaps" / f"{model.roi_id}.png")

        results.append(
            {
                "id": model.roi_id,
                "source_group": model.source_group,
                "roi": list(model.roi),
                "score": score,
                "threshold": threshold,
                "base_threshold": model.threshold,
                "threshold_scale": args.threshold_scale,
                "status": status,
                "reason": reason,
                "patch_stats": stats,
                "memory_features": int(len(model.memory)),
                "calibration_scores": model.calibration_scores,
                "crop": str(out / "crops" / f"{model.roi_id}.png"),
                "heatmap": str(out / "heatmaps" / f"{model.roi_id}.png"),
            }
        )

    scoring_seconds = time.perf_counter() - scoring_start
    if any_ng:
        final_status = "NG"
    elif any_unknown:
        final_status = "UNKNOWN"
    else:
        final_status = "PASS"

    banner_color = (0, 160, 0) if final_status == "PASS" else ((0, 0, 255) if final_status == "NG" else (0, 180, 255))
    banner_h = max(50, int(round(preview.shape[0] * 0.06)))
    cv2.rectangle(preview, (0, 0), (preview.shape[1], banner_h), banner_color, -1)
    cv2.putText(
        preview,
        f"{final_status}  ROI DINOv2/PatchCore",
        (20, int(banner_h * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.8, banner_h / 65.0),
        (255, 255, 255),
        max(2, int(round(banner_h / 30))),
        cv2.LINE_AA,
    )

    total_seconds = time.perf_counter() - t_total
    write_image(out / "aligned.png", aligned)
    write_image(out / "inspection_preview.png", preview)
    report = {
        "schema_version": 1,
        "model_type": "roi_dinov2_patchcore",
        "input": str(input_path),
        "model_dir": str(model_dir),
        "final_status": final_status,
        "alignment": {
            **alignment.to_dict(),
            "accepted": alignment_ok,
            "min_inlier_ratio": min_inlier_ratio,
        },
        "timing_seconds": {
            "alignment": alignment_seconds,
            "dino_batch": dino_seconds,
            "memory_scoring": scoring_seconds,
            "total": total_seconds,
        },
        "rois": results,
    }
    write_json(out / "inspection.json", report)

    print("", flush=True)
    print(f"DINO batch time: {dino_seconds:.3f}s for {len(models)} ROI(s)", flush=True)
    print(f"Scoring time:    {scoring_seconds:.3f}s", flush=True)
    print(f"TOTAL:           {total_seconds:.3f}s", flush=True)
    print(f"FINAL:           {final_status}", flush=True)
    print(f"Preview:         {out / 'inspection_preview.png'}", flush=True)
    print(f"Report:          {out / 'inspection.json'}", flush=True)


if __name__ == "__main__":
    main()
