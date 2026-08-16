from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.anomaly.coreset import approximate_greedy_coreset
from product_align_inspector.anomaly.dinov2_adapter import DINOv2Adapter, DINOv2Config
from product_align_inspector.anomaly.roi_patchcore import (
    ROIModel,
    save_roi_model,
    score_patch_tokens,
    select_regions,
    write_model_manifest,
)
from product_align_inspector.io_utils import read_image
from product_align_inspector.roi import crop_roi, validate_roi

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-ROI GOOD-only DINOv2/PatchCore memory banks after product alignment. "
            "No defect images are used for training."
        )
    )
    parser.add_argument("--good-dir", required=True, help="Directory containing full-resolution GOOD product images")
    parser.add_argument("--reference", required=True, help="Canonical reference_aligned.png")
    parser.add_argument("--config", required=True, help="ROI config JSON")
    parser.add_argument("--output", default="artifacts/roi_dino_patchcore")
    parser.add_argument("--roi-id", action="append", default=[], help="ROI ID to model; repeat for multiple ROIs")
    parser.add_argument("--dino-repo", default="third_party/dinov2")
    parser.add_argument("--dino-weights", default="weights/dinov2_vits14_pretrain.pth")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--image-size", type=int, default=224, help="DINO input size; must be divisible by 14")
    parser.add_argument("--coreset", type=float, default=0.10, help="Memory coreset ratio in (0,1]")
    parser.add_argument("--coreset-projection-dim", type=int, default=64)
    parser.add_argument("--calibration-ratio", type=float, default=0.20, help="GOOD source-image fraction reserved for threshold calibration")
    parser.add_argument("--threshold-margin", type=float, default=1.10, help="threshold = max(calibration GOOD scores) * margin")
    parser.add_argument("--score-top-fraction", type=float, default=0.05, help="Mean of highest local patch distances used as ROI score")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=int, default=238, help="Alignment foreground fallback threshold")
    parser.add_argument("--min-inlier-ratio", type=float, default=0.25)
    args = parser.parse_args()

    good_dir = Path(args.good_dir)
    reference_path = Path(args.reference)
    config_path = Path(args.config)
    out = Path(args.output)

    if not good_dir.is_dir():
        raise SystemExit(f"GOOD directory not found: {good_dir}")
    if not 0.0 < args.coreset <= 1.0:
        raise SystemExit("--coreset must be in (0,1]")
    if not 0.0 < args.calibration_ratio < 0.5:
        raise SystemExit("--calibration-ratio must be in (0,0.5)")
    if args.threshold_margin < 1.0:
        raise SystemExit("--threshold-margin should be >= 1.0")
    if not 0.0 < args.score_top_fraction <= 1.0:
        raise SystemExit("--score-top-fraction must be in (0,1]")

    images = _collect_images(good_dir)
    if len(images) < 2:
        raise SystemExit("At least 2 distinct GOOD source images are required; 10+ is recommended for a first useful model.")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    regions = select_regions(config, args.roi_id or None)
    selected_ids = [r.id for r in regions]

    rng = random.Random(args.seed)
    shuffled = images[:]
    rng.shuffle(shuffled)
    calibration_count = max(1, int(round(len(shuffled) * args.calibration_ratio)))
    calibration_count = min(calibration_count, len(shuffled) - 1)
    calibration_set = set(shuffled[:calibration_count])
    bank_set = set(shuffled[calibration_count:])

    print("=== ROI DINOv2 / PatchCore Builder ===", flush=True)
    print(f"GOOD source images: {len(images)}", flush=True)
    print(f"Bank sources:        {len(bank_set)}", flush=True)
    print(f"Calibration sources: {len(calibration_set)}", flush=True)
    print(f"ROI IDs:              {selected_ids}", flush=True)
    print(f"Coreset ratio:        {args.coreset:.3f}", flush=True)
    print(f"Score top fraction:   {args.score_top_fraction:.3f}", flush=True)
    if len(images) < 10:
        print("WARNING: fewer than 10 GOOD source images; use this only as a smoke-test model.", flush=True)

    reference = read_image(reference_path)
    align_cfg = ProductLocatorConfig(foreground_threshold=args.threshold)

    dino_cfg = DINOv2Config(
        image_size=args.image_size,
        repo_dir=args.dino_repo,
        weights_path=args.dino_weights,
    )
    dino = DINOv2Adapter(device=args.device, config=dino_cfg, project_root=REPO_ROOT)
    dino.load()

    train_tokens: dict[str, list[np.ndarray]] = {r.id: [] for r in regions}
    calibration_tokens: dict[str, list[tuple[str, np.ndarray]]] = {r.id: [] for r in regions}
    rows: list[dict[str, object]] = []
    success_bank = 0
    success_calibration = 0

    start_all = time.perf_counter()
    for index, image_path in enumerate(images, 1):
        split = "calibration" if image_path in calibration_set else "bank"
        t0 = time.perf_counter()
        try:
            raw = read_image(image_path)
            alignment = align_to_reference(raw, reference, align_cfg)
            if alignment.method.startswith("sift") and alignment.feature_inlier_ratio < args.min_inlier_ratio:
                raise RuntimeError(
                    f"weak alignment inlier_ratio={alignment.feature_inlier_ratio:.3f} < {args.min_inlier_ratio:.3f}"
                )

            aligned = alignment.aligned
            h, w = aligned.shape[:2]
            crops: list[np.ndarray] = []
            for region in regions:
                if not validate_roi(region.roi, w, h):
                    raise RuntimeError(f"invalid ROI {region.id}: {region.roi} for aligned {w}x{h}")
                crops.append(crop_roi(aligned, region.roi))

            batch_tokens = dino.patch_tokens_batch(crops)
            if len(batch_tokens) != len(regions):
                raise RuntimeError("DINO batch output count does not match ROI count")

            source_key = str(image_path.relative_to(good_dir).as_posix())
            for region, tokens in zip(regions, batch_tokens):
                if split == "bank":
                    train_tokens[region.id].append(tokens)
                else:
                    calibration_tokens[region.id].append((source_key, tokens))

            if split == "bank":
                success_bank += 1
            else:
                success_calibration += 1

            elapsed = time.perf_counter() - t0
            rows.append(
                {
                    "source": source_key,
                    "split": split,
                    "status": "ok",
                    "alignment_method": alignment.method,
                    "feature_matches": alignment.feature_matches,
                    "feature_inliers": alignment.feature_inliers,
                    "feature_inlier_ratio": alignment.feature_inlier_ratio,
                    "ecc_score": "" if alignment.ecc_score is None else alignment.ecc_score,
                    "seconds": elapsed,
                    "error": "",
                }
            )
            print(
                f"[{index}/{len(images)}] {image_path.name} -> {split.upper()} OK "
                f"({alignment.method}, inliers={alignment.feature_inlier_ratio:.1%}, {elapsed:.2f}s)",
                flush=True,
            )
        except Exception as exc:
            rows.append(
                {
                    "source": str(image_path.relative_to(good_dir).as_posix()),
                    "split": split,
                    "status": "failed",
                    "alignment_method": "",
                    "feature_matches": "",
                    "feature_inliers": "",
                    "feature_inlier_ratio": "",
                    "ecc_score": "",
                    "seconds": time.perf_counter() - t0,
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(images)}] {image_path.name} -> FAILED: {exc}", flush=True)

    if success_bank == 0:
        raise RuntimeError("No bank GOOD image succeeded")
    if success_calibration == 0:
        raise RuntimeError("No calibration GOOD image succeeded")

    out.mkdir(parents=True, exist_ok=True)
    roi_summaries: list[dict[str, object]] = []

    print("", flush=True)
    print("=== Building per-ROI memory banks ===", flush=True)
    for roi_index, region in enumerate(regions, 1):
        if not train_tokens[region.id]:
            raise RuntimeError(f"No bank tokens for ROI {region.id}")

        all_features = np.concatenate(train_tokens[region.id], axis=0).astype(np.float32)
        coreset_device = dino.device
        t0 = time.perf_counter()
        memory = approximate_greedy_coreset(
            all_features,
            ratio=args.coreset,
            device=coreset_device,
            projection_dim=args.coreset_projection_dim,
            seed=args.seed + roi_index,
        )
        coreset_seconds = time.perf_counter() - t0

        calibration_scores: list[float] = []
        calibration_detail: list[dict[str, object]] = []
        for source_name, tokens in calibration_tokens[region.id]:
            score, _map, stats = score_patch_tokens(
                tokens,
                memory,
                patch_grid=dino.patch_grid,
                top_fraction=args.score_top_fraction,
            )
            calibration_scores.append(score)
            calibration_detail.append({"source": source_name, **stats})

        threshold = None
        if calibration_scores:
            threshold = max(1e-6, float(max(calibration_scores)) * float(args.threshold_margin))

        roi_model = ROIModel(
            roi_id=region.id,
            roi=region.roi,
            source_group=region.source_group,
            memory=memory,
            threshold=threshold,
            calibration_scores=calibration_scores,
            score_top_fraction=args.score_top_fraction,
            patch_grid=dino.patch_grid,
            feature_dim=dino_cfg.embedding_dim,
        )
        bank_path = save_roi_model(out, roi_model)

        summary = {
            "id": region.id,
            "roi": list(region.roi),
            "source_group": region.source_group,
            "raw_patch_features": int(len(all_features)),
            "memory_features": int(len(memory)),
            "memory_mb_float32": float(memory.nbytes / (1024 * 1024)),
            "threshold": threshold,
            "calibration_scores": calibration_scores,
            "calibration_detail": calibration_detail,
            "bank_file": str(bank_path.relative_to(out).as_posix()),
            "coreset_seconds": coreset_seconds,
        }
        roi_summaries.append(summary)
        threshold_text = "UNSET" if threshold is None else f"{threshold:.6f}"
        print(
            f"{region.id}: raw={len(all_features)} -> memory={len(memory)} "
            f"({memory.nbytes / (1024 * 1024):.2f} MB), threshold={threshold_text}, "
            f"coreset={coreset_seconds:.2f}s",
            flush=True,
        )

    manifest_payload = {
        "schema_version": 1,
        "model_type": "roi_dinov2_patchcore",
        "product": config.get("product", "unknown"),
        "reference_image": str(reference_path),
        "config_file": str(config_path),
        "training_data": {
            "good_dir": str(good_dir),
            "source_images": len(images),
            "successful_bank_sources": success_bank,
            "successful_calibration_sources": success_calibration,
            "seed": args.seed,
            "calibration_ratio": args.calibration_ratio,
        },
        "alignment": {
            "min_inlier_ratio": args.min_inlier_ratio,
            "foreground_threshold": args.threshold,
        },
        "dino": {
            "model_name": dino_cfg.model_name,
            "image_size": dino_cfg.image_size,
            "patch_size": 14,
            "patch_grid": dino.patch_grid,
            "embedding_dim": dino_cfg.embedding_dim,
            "repo_dir": str(dino.repo_dir),
            "weights_path": str(dino.weights_path),
            "preprocess": {
                "pad_to_square": True,
                "pad_value": dino_cfg.pad_value,
                "resize": [dino_cfg.image_size, dino_cfg.image_size],
                "interpolation": "OpenCV INTER_LINEAR",
                "color_order": "RGB",
                "scale": "uint8 / 255.0",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "layout": "NCHW",
            },
        },
        "patchcore": {
            "metric": "cosine_distance_on_l2_normalized_patch_tokens",
            "coreset_ratio": args.coreset,
            "coreset_method": "approximate_greedy",
            "score_top_fraction": args.score_top_fraction,
            "threshold_rule": "max_good_calibration_score * threshold_margin",
            "threshold_margin": args.threshold_margin,
        },
        "rois": roi_summaries,
        "build_seconds": time.perf_counter() - start_all,
    }
    model_json = write_model_manifest(out, manifest_payload)

    report_path = out / "build_report.csv"
    fieldnames = [
        "source",
        "split",
        "status",
        "alignment_method",
        "feature_matches",
        "feature_inliers",
        "feature_inlier_ratio",
        "ecc_score",
        "seconds",
        "error",
    ]
    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("", flush=True)
    print("Build complete.", flush=True)
    print(f"Model:  {model_json}", flush=True)
    print(f"Report: {report_path}", flush=True)
    print(f"Banks:  {out / 'banks'}", flush=True)


if __name__ == "__main__":
    main()
