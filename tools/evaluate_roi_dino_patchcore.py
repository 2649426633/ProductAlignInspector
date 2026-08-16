from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.anomaly.dinov2_adapter import DINOv2Adapter, DINOv2Config
from product_align_inspector.anomaly.roi_patchcore import load_roi_model, read_model_manifest, score_patch_tokens
from product_align_inspector.io_utils import read_image
from product_align_inspector.roi import crop_roi, validate_roi

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _collect_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _safe_mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _safe_min(values: list[float]) -> float | None:
    return None if not values else float(np.min(values))


def _safe_max(values: list[float]) -> float | None:
    return None if not values else float(np.max(values))


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate independent GOOD/NG images against ROI DINOv2/PatchCore memory banks."
    )
    parser.add_argument("--test-root", required=True, help="Directory containing test/good and test/ng folders")
    parser.add_argument("--model-dir", default="artifacts/roi_dino_patchcore")
    parser.add_argument("--reference", help="Override reference image saved in model.json")
    parser.add_argument("--dino-repo", help="Override local DINOv2 repository")
    parser.add_argument("--dino-weights", help="Override DINOv2 weights")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="artifacts/roi_dino_evaluation")
    parser.add_argument("--threshold-scale", type=float, default=1.0)
    args = parser.parse_args()

    if args.threshold_scale <= 0:
        raise SystemExit("--threshold-scale must be > 0")

    test_root = Path(args.test_root)
    good_dir = test_root / "good"
    ng_dir = test_root / "ng"
    good_images = _collect_images(good_dir)
    ng_images = _collect_images(ng_dir)
    if not good_images and not ng_images:
        raise SystemExit(
            f"No images found. Expected {good_dir} and/or {ng_dir} with BMP/PNG/JPG/TIF images."
        )

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

    reference = read_image(reference_path)
    align_cfg = ProductLocatorConfig(
        foreground_threshold=int(align_meta.get("foreground_threshold", 238))
    )
    min_inlier_ratio = float(align_meta.get("min_inlier_ratio", 0.25))

    dino = DINOv2Adapter(device=args.device, config=dino_cfg, project_root=REPO_ROOT)
    dino.load()

    samples: list[tuple[str, Path]] = [("GOOD", p) for p in good_images] + [("NG", p) for p in ng_images]
    rows: list[dict[str, object]] = []
    per_roi: dict[str, dict[str, list[float]]] = {
        m.roi_id: {"GOOD": [], "NG": []} for m in models
    }
    image_predictions: list[dict[str, object]] = []

    print("=== ROI DINOv2 / PatchCore Batch Evaluation ===", flush=True)
    print(f"GOOD images: {len(good_images)}", flush=True)
    print(f"NG images:   {len(ng_images)}", flush=True)
    print(f"ROI IDs:     {[m.roi_id for m in models]}", flush=True)
    print(f"Threshold scale: {args.threshold_scale:.3f}", flush=True)
    print("", flush=True)

    total_start = time.perf_counter()
    total_dino = 0.0
    total_align = 0.0
    failed_images = 0

    for index, (truth, image_path) in enumerate(samples, 1):
        t_image = time.perf_counter()
        try:
            raw = read_image(image_path)
            t0 = time.perf_counter()
            alignment = align_to_reference(raw, reference, align_cfg)
            total_align += time.perf_counter() - t0
            alignment_ok = not (
                alignment.method.startswith("sift")
                and alignment.feature_inlier_ratio < min_inlier_ratio
            )
            if not alignment_ok:
                raise RuntimeError(
                    f"weak alignment inlier_ratio={alignment.feature_inlier_ratio:.3f} < {min_inlier_ratio:.3f}"
                )

            aligned = alignment.aligned
            h, w = aligned.shape[:2]
            crops = []
            for model in models:
                if not validate_roi(model.roi, w, h):
                    raise RuntimeError(f"invalid ROI {model.roi_id}: {model.roi}")
                crops.append(crop_roi(aligned, model.roi))

            t0 = time.perf_counter()
            tokens_batch = dino.patch_tokens_batch(crops)
            dino_seconds = time.perf_counter() - t0
            total_dino += dino_seconds

            image_any_ng = False
            image_roi_results = []
            for model, tokens in zip(models, tokens_batch):
                score, _anomaly_map, stats = score_patch_tokens(
                    tokens,
                    model.memory,
                    patch_grid=model.patch_grid,
                    top_fraction=model.score_top_fraction,
                )
                threshold = None if model.threshold is None else float(model.threshold) * args.threshold_scale
                pred = "UNKNOWN" if threshold is None else ("NG" if score > threshold else "GOOD")
                if pred == "NG":
                    image_any_ng = True

                per_roi[model.roi_id][truth].append(float(score))
                ratio = None if threshold is None or threshold <= 1e-12 else float(score / threshold)
                row = {
                    "truth": truth,
                    "source": str(image_path),
                    "roi_id": model.roi_id,
                    "score": float(score),
                    "threshold": threshold,
                    "score_over_threshold": ratio,
                    "roi_prediction": pred,
                    "max_patch": float(stats["max"]),
                    "mean_patch": float(stats["mean"]),
                    "alignment_method": alignment.method,
                    "feature_inlier_ratio": float(alignment.feature_inlier_ratio),
                    "ecc_score": "" if alignment.ecc_score is None else float(alignment.ecc_score),
                    "dino_seconds": dino_seconds,
                    "status": "ok",
                    "error": "",
                }
                rows.append(row)
                image_roi_results.append(row)

            image_pred = "NG" if image_any_ng else "GOOD"
            correct = image_pred == truth
            image_predictions.append(
                {
                    "truth": truth,
                    "prediction": image_pred,
                    "correct": correct,
                    "source": str(image_path),
                }
            )

            elapsed = time.perf_counter() - t_image
            print(
                f"[{index}/{len(samples)}] {truth:<4} {image_path.name:<28} -> {image_pred:<4} "
                f"{'OK' if correct else 'MISS'}  align={alignment.feature_inlier_ratio:.1%} "
                f"dino={dino_seconds:.3f}s total={elapsed:.3f}s",
                flush=True,
            )
            score_text = "  ".join(
                f"{r['roi_id']}={float(r['score']):.4f}/t={float(r['threshold']):.4f}" if r["threshold"] is not None
                else f"{r['roi_id']}={float(r['score']):.4f}/t=-"
                for r in image_roi_results
            )
            print(f"    {score_text}", flush=True)

        except Exception as exc:
            failed_images += 1
            rows.append(
                {
                    "truth": truth,
                    "source": str(image_path),
                    "roi_id": "",
                    "score": "",
                    "threshold": "",
                    "score_over_threshold": "",
                    "roi_prediction": "",
                    "max_patch": "",
                    "mean_patch": "",
                    "alignment_method": "",
                    "feature_inlier_ratio": "",
                    "ecc_score": "",
                    "dino_seconds": "",
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(samples)}] {truth:<4} {image_path.name} -> FAILED: {exc}", flush=True)

    total_seconds = time.perf_counter() - total_start

    csv_path = out / "scores.csv"
    fieldnames = [
        "truth", "source", "roi_id", "score", "threshold", "score_over_threshold",
        "roi_prediction", "max_patch", "mean_patch", "alignment_method",
        "feature_inlier_ratio", "ecc_score", "dino_seconds", "status", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    roi_summary: dict[str, object] = {}
    print("", flush=True)
    print("=== Per-ROI score separation ===", flush=True)
    print(f"{'ROI':<10} {'GOOD min':>10} {'GOOD mean':>10} {'GOOD max':>10} {'NG min':>10} {'NG mean':>10} {'NG max':>10} {'Threshold':>10}", flush=True)
    print("-" * 94, flush=True)

    for model in models:
        good_scores = per_roi[model.roi_id]["GOOD"]
        ng_scores = per_roi[model.roi_id]["NG"]
        threshold = None if model.threshold is None else float(model.threshold) * args.threshold_scale
        item = {
            "threshold": threshold,
            "good": {
                "count": len(good_scores),
                "min": _safe_min(good_scores),
                "mean": _safe_mean(good_scores),
                "max": _safe_max(good_scores),
            },
            "ng": {
                "count": len(ng_scores),
                "min": _safe_min(ng_scores),
                "mean": _safe_mean(ng_scores),
                "max": _safe_max(ng_scores),
            },
        }
        if good_scores and ng_scores:
            item["raw_gap_ng_min_minus_good_max"] = float(min(ng_scores) - max(good_scores))
            item["separated_without_overlap"] = bool(min(ng_scores) > max(good_scores))
        roi_summary[model.roi_id] = item
        print(
            f"{model.roi_id:<10} {_fmt(_safe_min(good_scores)):>10} {_fmt(_safe_mean(good_scores)):>10} "
            f"{_fmt(_safe_max(good_scores)):>10} {_fmt(_safe_min(ng_scores)):>10} "
            f"{_fmt(_safe_mean(ng_scores)):>10} {_fmt(_safe_max(ng_scores)):>10} {_fmt(threshold):>10}",
            flush=True,
        )

    valid_image_predictions = image_predictions
    total_valid = len(valid_image_predictions)
    correct_count = sum(1 for r in valid_image_predictions if bool(r["correct"]))
    good_total = sum(1 for r in valid_image_predictions if r["truth"] == "GOOD")
    ng_total = sum(1 for r in valid_image_predictions if r["truth"] == "NG")
    false_ng = sum(1 for r in valid_image_predictions if r["truth"] == "GOOD" and r["prediction"] == "NG")
    missed_ng = sum(1 for r in valid_image_predictions if r["truth"] == "NG" and r["prediction"] == "GOOD")

    summary = {
        "schema_version": 1,
        "model_dir": str(model_dir),
        "test_root": str(test_root),
        "threshold_scale": float(args.threshold_scale),
        "image_level": {
            "valid_images": total_valid,
            "failed_images": failed_images,
            "correct": correct_count,
            "accuracy": None if total_valid == 0 else correct_count / total_valid,
            "good_images": good_total,
            "ng_images": ng_total,
            "false_ng_on_good": false_ng,
            "missed_ng": missed_ng,
            "good_false_positive_rate": None if good_total == 0 else false_ng / good_total,
            "ng_miss_rate": None if ng_total == 0 else missed_ng / ng_total,
        },
        "timing_seconds": {
            "total": total_seconds,
            "alignment_total": total_align,
            "dino_total": total_dino,
            "mean_total_per_valid_image": None if total_valid == 0 else total_seconds / total_valid,
            "mean_dino_per_valid_image": None if total_valid == 0 else total_dino / total_valid,
        },
        "rois": roi_summary,
        "images": valid_image_predictions,
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("", flush=True)
    print("=== Image-level summary ===", flush=True)
    print(f"Valid images:       {total_valid}", flush=True)
    print(f"Failed images:      {failed_images}", flush=True)
    print(f"Correct:            {correct_count}/{total_valid}" if total_valid else "Correct:            -", flush=True)
    print(f"GOOD false alarms:  {false_ng}/{good_total}" if good_total else "GOOD false alarms:  -", flush=True)
    print(f"Missed NG:          {missed_ng}/{ng_total}" if ng_total else "Missed NG:          -", flush=True)
    print(f"Mean DINO/image:    {total_dino / total_valid:.3f}s" if total_valid else "Mean DINO/image:    -", flush=True)
    print(f"Scores CSV:         {csv_path}", flush=True)
    print(f"Summary JSON:       {summary_path}", flush=True)


if __name__ == "__main__":
    main()
