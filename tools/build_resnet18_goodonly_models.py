from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.io_utils import read_image, write_image
from product_align_inspector.resnet18_goodonly import (
    DEFAULT_RESNET18_WEIGHTS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_feature_extractor,
    calibrate_threshold,
    extract_features,
    robust_bank_mask,
)
from product_align_inspector.roi import crop_roi, validate_roi

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def group_of(roi_id: str) -> str | None:
    rid = roi_id.upper()
    if rid.startswith("S"):
        return "S"
    if rid.startswith("E"):
        return "E"
    return None


def draw_reference_preview(reference: np.ndarray, slots: list[dict[str, object]], path: Path) -> None:
    canvas = reference.copy()
    h, w = canvas.shape[:2]
    thickness = max(3, int(round(min(h, w) / 650)))
    font_scale = max(0.65, min(1.1, min(h, w) / 1800.0))
    for slot in slots:
        roi = slot.get("roi")
        if roi is None or not validate_roi(roi, w, h):
            continue
        x, y, rw, rh = [int(v) for v in roi]
        group = str(slot["group"])
        color = (0, 190, 0) if group == "S" else (0, 150, 220)
        cv2.rectangle(canvas, (x, y), (x + rw, y + rh), color, thickness)
        cv2.putText(
            canvas,
            str(slot["id"]),
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(2, thickness - 1),
            cv2.LINE_AA,
        )
    write_image(path, canvas)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Build two independent GOOD-only normal models from fixed ROIs using an offline "
            "ImageNet ResNet18 feature extractor. S learns normal screw appearance; "
            "E learns normal empty appearance. Test/NG images are never used."
        )
    )
    p.add_argument("--good-dir", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--weights", default=str(DEFAULT_RESNET18_WEIGHTS))
    p.add_argument("--output", default="artifacts/resnet18_goodonly")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--k-neighbors", type=int, default=3)
    p.add_argument("--threshold-quantile", type=float, default=0.98)
    p.add_argument("--threshold-margin", type=float, default=1.10)
    p.add_argument("--outlier-quantile", type=float, default=0.95)
    p.add_argument("--outlier-mad-scale", type=float, default=5.0)
    p.add_argument("--foreground-threshold", type=int, default=238)
    args = p.parse_args()

    good_dir = Path(args.good_dir).resolve()
    reference_path = Path(args.reference).resolve()
    config_path = Path(args.config).resolve()
    weights_path = Path(args.weights).resolve()
    output = Path(args.output).resolve()

    if not good_dir.is_dir():
        raise SystemExit(f"GOOD directory not found: {good_dir}")
    if not reference_path.is_file():
        raise SystemExit(f"Reference not found: {reference_path}")
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    if not weights_path.is_file():
        raise SystemExit(
            f"ResNet18 weights not found: {weights_path}\n"
            "Put the official resnet18-f37072fd.pth in D:\\Brunei\\weight, or pass --weights."
        )

    images = collect_images(good_dir)
    if not images:
        raise SystemExit(f"No GOOD images found: {good_dir}")

    reference = read_image(reference_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    slots: list[dict[str, object]] = []
    for item in config.get("screw_slots", []):
        if not bool(item.get("enabled", True)):
            continue
        roi_id = str(item.get("id", ""))
        group = group_of(roi_id)
        if group is None:
            continue
        slots.append({"id": roi_id, "group": group, "roi": item.get("roi")})

    s_ids = [s["id"] for s in slots if s["group"] == "S"]
    e_ids = [s["id"] for s in slots if s["group"] == "E"]
    if not s_ids or not e_ids:
        raise SystemExit(f"Need both S and E ROIs in config. S={s_ids}, E={e_ids}")

    output.mkdir(parents=True, exist_ok=True)
    draw_reference_preview(reference, slots, output / "reference_roi_preview.png")
    model, device = build_feature_extractor(weights_path, args.device)
    align_cfg = ProductLocatorConfig(foreground_threshold=args.foreground_threshold)

    crops_by_roi: dict[str, list[np.ndarray]] = {str(s["id"]): [] for s in slots}
    sources_by_roi: dict[str, list[str]] = {str(s["id"]): [] for s in slots}
    report_rows: list[dict[str, object]] = []

    print("=== ResNet18 GOOD-only S/E Builder ===")
    print(f"Device: {device}")
    print(f"Weights: {weights_path}")
    print(f"GOOD images: {len(images)}")
    print(f"S model ROIs: {s_ids}")
    print(f"E model ROIs: {e_ids}")
    print("NG/test images used for model building: 0")
    print(f"Reference ROI preview: {output / 'reference_roi_preview.png'}")

    for index, image_path in enumerate(images, 1):
        try:
            raw = read_image(image_path)
            result = align_to_reference(raw, reference, align_cfg)
            if result.feature_matrix is None:
                raise RuntimeError(
                    f"alignment method {result.method} has no feature_matrix; excluded from normal bank"
                )

            aligned = result.aligned
            h, w = aligned.shape[:2]
            for slot in slots:
                roi_id = str(slot["id"])
                roi = slot["roi"]
                if roi is None or not validate_roi(roi, w, h):
                    raise RuntimeError(f"invalid ROI {roi_id}: {roi}")
                crop = crop_roi(aligned, roi)
                crops_by_roi[roi_id].append(crop)
                sources_by_roi[roi_id].append(image_path.name)
                write_image(output / "debug_good_crops" / roi_id / f"{image_path.stem}.png", crop)

            report_rows.append(
                {
                    "image": str(image_path),
                    "status": "ok",
                    "alignment_method": result.method,
                    "feature_matches": result.feature_matches,
                    "feature_inliers": result.feature_inliers,
                    "feature_inlier_ratio": result.feature_inlier_ratio,
                    "ecc_score": "" if result.ecc_score is None else result.ecc_score,
                    "error": "",
                }
            )
            print(
                f"[{index}/{len(images)}] {image_path.name} -> OK "
                f"({result.method}, inliers={result.feature_inlier_ratio:.1%}, "
                f"ecc={'-' if result.ecc_score is None else format(result.ecc_score, '.4f')})"
            )
        except Exception as exc:
            report_rows.append(
                {
                    "image": str(image_path),
                    "status": "skipped",
                    "alignment_method": "",
                    "feature_matches": "",
                    "feature_inliers": "",
                    "feature_inlier_ratio": "",
                    "ecc_score": "",
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(images)}] {image_path.name} -> SKIP: {exc}")

    accepted = sum(1 for row in report_rows if row["status"] == "ok")
    if accepted < 8:
        raise SystemExit(f"Only {accepted} GOOD images survived alignment; need more reliable GOOD samples.")

    group_entries: dict[str, list[dict[str, object]]] = {"S": [], "E": []}
    quality_rows: list[dict[str, object]] = []
    print("")
    print("=== Building robust normal feature banks ===")
    for slot in slots:
        roi_id = str(slot["id"])
        group = str(slot["group"])
        crops = crops_by_roi[roi_id]
        sources = sources_by_roi[roi_id]
        if len(crops) < 8:
            raise SystemExit(f"{roi_id} has only {len(crops)} accepted GOOD crops")

        raw_bank = extract_features(
            model,
            crops,
            device=device,
            input_size=args.input_size,
            batch_size=args.batch_size,
        )
        keep_mask, raw_scores, filter_stats = robust_bank_mask(
            raw_bank,
            k_neighbors=args.k_neighbors,
            outlier_quantile=args.outlier_quantile,
            mad_scale=args.outlier_mad_scale,
        )
        bank = raw_bank[keep_mask]
        threshold, calibration = calibrate_threshold(
            bank,
            k_neighbors=args.k_neighbors,
            quantile=args.threshold_quantile,
            margin=args.threshold_margin,
        )

        group_dir = output / group
        banks_dir = group_dir / "banks"
        banks_dir.mkdir(parents=True, exist_ok=True)
        bank_path = banks_dir / f"{roi_id}.npy"
        np.save(bank_path, bank.astype(np.float32, copy=False))

        removed_sources = [sources[i] for i, keep in enumerate(keep_mask.tolist()) if not keep]
        for i, source in enumerate(sources):
            quality_rows.append(
                {
                    "roi_id": roi_id,
                    "group": group,
                    "source": source,
                    "raw_loo_score": float(raw_scores[i]),
                    "kept": bool(keep_mask[i]),
                    "filter_cutoff": float(filter_stats["cutoff"]),
                }
            )

        entry = {
            "id": roi_id,
            "roi": list(slot["roi"]),
            "bank_file": f"banks/{roi_id}.npy",
            "raw_bank_count": int(raw_bank.shape[0]),
            "bank_count": int(bank.shape[0]),
            "feature_dim": int(bank.shape[1]),
            "threshold": float(threshold),
            "bank_filter": filter_stats,
            "removed_good_sources": removed_sources,
            "calibration": calibration,
        }
        group_entries[group].append(entry)
        removed_text = ",".join(removed_sources) if removed_sources else "-"
        print(
            f"{roi_id}: raw={raw_bank.shape[0]} kept={bank.shape[0]}x{bank.shape[1]} "
            f"removed={len(removed_sources)} [{removed_text}] "
            f"p98={calibration['p98']:.6f} max={calibration['max']:.6f} "
            f"threshold={threshold:.6f}"
        )

    for group in ("S", "E"):
        group_dir = output / group
        normal_semantics = "screw_present" if group == "S" else "empty"
        manifest = {
            "schema_version": 2,
            "model_type": "resnet18_goodonly_knn",
            "group": group,
            "normal_semantics": normal_semantics,
            "architecture": "resnet18",
            "feature_layer": "global_avgpool_512",
            "weights_file": weights_path.name,
            "input_size": int(args.input_size),
            "color_order": "RGB",
            "preprocess": {
                "pad_to_square": True,
                "pad_value_rgb": [255, 255, 255],
                "resize": [args.input_size, args.input_size],
                "scale": "uint8 / 255.0",
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
                "layout": "NCHW",
                "feature_l2_normalize": True,
            },
            "scoring": {
                "distance": "cosine",
                "score": "mean distance to k nearest GOOD features",
                "k_neighbors": int(args.k_neighbors),
                "threshold_quantile": float(args.threshold_quantile),
                "threshold_margin": float(args.threshold_margin),
                "calibration": "robust-filtered leave-one-out GOOD only",
                "outlier_quantile": float(args.outlier_quantile),
                "outlier_mad_scale": float(args.outlier_mad_scale),
            },
            "training": {
                "neural_finetuning": False,
                "good_source_images": int(accepted),
                "ng_source_images": 0,
                "test_source_images": 0,
            },
            "rois": group_entries[group],
        }
        (group_dir / "model.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    report_path = output / "build_report.csv"
    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "status",
                "alignment_method",
                "feature_matches",
                "feature_inliers",
                "feature_inlier_ratio",
                "ecc_score",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    quality_path = output / "bank_quality.csv"
    with quality_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["roi_id", "group", "source", "raw_loo_score", "kept", "filter_cutoff"],
        )
        writer.writeheader()
        writer.writerows(quality_rows)

    summary = {
        "model_type": "resnet18_goodonly_knn",
        "accepted_good_images": accepted,
        "skipped_good_images": len(images) - accepted,
        "S_model": str(output / "S" / "model.json"),
        "E_model": str(output / "E" / "model.json"),
        "build_report": str(report_path),
        "bank_quality": str(quality_path),
        "reference_roi_preview": str(output / "reference_roi_preview.png"),
        "debug_good_crops": str(output / "debug_good_crops"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("")
    print("=== Build complete ===")
    print(f"Accepted GOOD: {accepted}/{len(images)}")
    print(f"S model: {output / 'S' / 'model.json'}")
    print(f"E model: {output / 'E' / 'model.json'}")
    print(f"Reference ROI preview: {output / 'reference_roi_preview.png'}")
    print(f"GOOD crop debug: {output / 'debug_good_crops'}")
    print(f"Bank quality: {quality_path}")
    print(f"Report: {report_path}")
    print("PatchCore files were not changed.")


if __name__ == "__main__":
    main()
