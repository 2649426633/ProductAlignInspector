from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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


def collect_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def classify_group(roi_id: str) -> str:
    rid = roi_id.upper()
    if rid.startswith("SPRING"):
        return "SPRING"
    if rid.startswith("E"):
        return "EMPTY"
    if rid.startswith("S"):
        return "SCREW"
    return "OTHER"


def scenario_policy(name: str, groups: dict[str, set[str]]) -> tuple[str, set[str], bool]:
    """Return (mode, target_rois, require_all_targets).

    GOOD: exact all-normal.
    all_empty: all expected-screw S* ROIs should be anomalous; E*/SPRING* stay normal.
    missing_screws: location unknown, detect if any S* ROI is anomalous.
    excess_screws: location unknown, detect if any E* ROI is anomalous.
    missing_springs/spring_missing: location unknown, detect if any SPRING* ROI is anomalous.
    Unknown NG folders fall back to any-ROI image-level detection.
    """
    lower = name.strip().lower()
    if lower == "good":
        return "GOOD_EXACT", set(), False
    if lower in {"all_empty", "all_missing", "all_screws_empty"}:
        return "SCREW_ALL", set(groups["SCREW"]), True
    if lower in {"missing_screws", "missing_screw"}:
        return "SCREW_ANY", set(groups["SCREW"]), False
    if lower in {"excess_screws", "extra_screws", "excess_screw", "extra_screw"}:
        return "EMPTY_ANY", set(groups["EMPTY"]), False
    if lower in {"missing_springs", "missing_spring", "spring_missing"}:
        return "SPRING_ANY", set(groups["SPRING"]), False
    return "GENERIC_ANY", set().union(*groups.values()), False


def fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.6f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Group-aware full-product ROI DINOv2/PatchCore evaluation.")
    p.add_argument("--test-root", required=True)
    p.add_argument("--model-dir", default="artifacts/roi_dino_full")
    p.add_argument("--reference")
    p.add_argument("--dino-repo")
    p.add_argument("--dino-weights")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="artifacts/roi_dino_full_evaluation")
    p.add_argument("--threshold-scale", type=float, default=1.0)
    args = p.parse_args()

    if args.threshold_scale <= 0:
        raise SystemExit("--threshold-scale must be > 0")

    test_root = Path(args.test_root)
    model_dir = Path(args.model_dir)
    manifest = read_model_manifest(model_dir)
    models = [load_roi_model(model_dir, str(item["id"])) for item in manifest.get("rois", [])]
    if not models:
        raise RuntimeError("No ROI models in model.json")

    groups: dict[str, set[str]] = {"SCREW": set(), "EMPTY": set(), "SPRING": set(), "OTHER": set()}
    for model in models:
        groups[classify_group(model.roi_id)].add(model.roi_id)

    scenarios: list[tuple[str, Path]] = []
    for image in collect_images(test_root / "good"):
        scenarios.append(("GOOD", image))
    ng_root = test_root / "ng"
    if ng_root.is_dir():
        for scenario_dir in sorted(p for p in ng_root.iterdir() if p.is_dir()):
            for image in collect_images(scenario_dir):
                scenarios.append((scenario_dir.name, image))
    if not scenarios:
        raise SystemExit("No test images found under test/good or test/ng/<scenario>.")

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

    reference = read_image(reference_path)
    align_cfg = ProductLocatorConfig(foreground_threshold=int(align_meta.get("foreground_threshold", 238)))
    min_inlier_ratio = float(align_meta.get("min_inlier_ratio", 0.25))
    dino = DINOv2Adapter(device=args.device, config=dino_cfg, project_root=REPO_ROOT)
    dino.load()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    valid = 0
    failed = 0
    good_total = good_ok = 0
    ng_total = ng_detected = 0
    target_total = target_detected = 0
    exact_total = exact_ok = 0
    good_false_alarm_rois = 0
    good_roi_total = 0
    off_target_defects = 0
    total_dino = 0.0
    t_all = time.perf_counter()

    print("=== Full ROI DINOv2 / PatchCore Evaluation ===", flush=True)
    print(f"Images: {len(scenarios)}", flush=True)
    print(f"SCREW ROIs:  {sorted(groups['SCREW'])}", flush=True)
    print(f"EMPTY ROIs:  {sorted(groups['EMPTY'])}", flush=True)
    print(f"SPRING ROIs: {sorted(groups['SPRING'])}", flush=True)
    print("", flush=True)

    for idx, (scenario, image_path) in enumerate(scenarios, 1):
        mode, targets, require_all = scenario_policy(scenario, groups)
        try:
            raw = read_image(image_path)
            alignment = align_to_reference(raw, reference, align_cfg)
            if alignment.method.startswith("sift") and alignment.feature_inlier_ratio < min_inlier_ratio:
                raise RuntimeError(
                    f"weak alignment {alignment.feature_inlier_ratio:.3f} < {min_inlier_ratio:.3f}"
                )

            aligned = alignment.aligned
            h, w = aligned.shape[:2]
            crops = []
            for model in models:
                if not validate_roi(model.roi, w, h):
                    raise RuntimeError(f"invalid ROI {model.roi_id}: {model.roi}")
                crops.append(crop_roi(aligned, model.roi))

            t0 = time.perf_counter()
            token_batch = dino.patch_tokens_batch(crops)
            dino_s = time.perf_counter() - t0
            total_dino += dino_s

            predicted: set[str] = set()
            score_map: dict[str, tuple[float, float | None]] = {}
            for model, tokens in zip(models, token_batch):
                score, _heat, _stats = score_patch_tokens(
                    tokens,
                    model.memory,
                    patch_grid=model.patch_grid,
                    top_fraction=model.score_top_fraction,
                )
                threshold = None if model.threshold is None else float(model.threshold) * args.threshold_scale
                is_defect = threshold is not None and score > threshold
                if is_defect:
                    predicted.add(model.roi_id)
                score_map[model.roi_id] = (float(score), threshold)
                rows.append({
                    "scenario": scenario,
                    "mode": mode,
                    "source": str(image_path),
                    "roi_id": model.roi_id,
                    "roi_group": classify_group(model.roi_id),
                    "target_roi": model.roi_id in targets,
                    "prediction": "DEFECT" if is_defect else "NORMAL",
                    "score": float(score),
                    "threshold": threshold,
                    "score_over_threshold": None if threshold is None else float(score / threshold),
                    "alignment_method": alignment.method,
                    "feature_inlier_ratio": float(alignment.feature_inlier_ratio),
                    "ecc_score": "" if alignment.ecc_score is None else float(alignment.ecc_score),
                    "dino_seconds": dino_s,
                    "status": "ok",
                    "error": "",
                })

            valid += 1
            target_preds = predicted & targets
            off_target = predicted - targets

            if mode == "GOOD_EXACT":
                good_total += 1
                good_roi_total += len(models)
                good_false_alarm_rois += len(predicted)
                image_ok = not predicted
                good_ok += int(image_ok)
                exact_total += 1
                exact_ok += int(image_ok)
                result = "OK" if image_ok else "FALSE_ALARM"
            else:
                ng_total += 1
                if require_all:
                    image_detected = bool(targets) and targets.issubset(predicted)
                    exact_match = predicted == targets
                    exact_total += 1
                    exact_ok += int(exact_match)
                else:
                    image_detected = bool(target_preds) if targets else bool(predicted)
                    exact_match = None
                ng_detected += int(image_detected)
                target_total += 1
                target_detected += int(image_detected)
                off_target_defects += len(off_target)
                result = "DETECTED" if image_detected else "MISS"
                if require_all and image_detected and not exact_match:
                    result = "DETECTED+OFFTARGET"

            pred_text = "GOOD" if not predicted else "+".join(sorted(predicted))
            target_text = "-" if not targets else "+".join(sorted(targets))
            print(
                f"[{idx}/{len(scenarios)}] {scenario:<18} {image_path.name:<24} "
                f"mode={mode:<11} target={target_text:<18} pred={pred_text:<45} {result}",
                flush=True,
            )

        except Exception as exc:
            failed += 1
            rows.append({
                "scenario": scenario,
                "mode": mode,
                "source": str(image_path),
                "roi_id": "",
                "roi_group": "",
                "target_roi": "",
                "prediction": "",
                "score": "",
                "threshold": "",
                "score_over_threshold": "",
                "alignment_method": "",
                "feature_inlier_ratio": "",
                "ecc_score": "",
                "dino_seconds": "",
                "status": "failed",
                "error": str(exc),
            })
            print(f"[{idx}/{len(scenarios)}] {image_path.name} -> FAILED: {exc}", flush=True)

    csv_path = out / "scores_full.csv"
    fieldnames = [
        "scenario", "mode", "source", "roi_id", "roi_group", "target_roi", "prediction",
        "score", "threshold", "score_over_threshold", "alignment_method", "feature_inlier_ratio",
        "ecc_score", "dino_seconds", "status", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_s = time.perf_counter() - t_all
    summary = {
        "schema_version": 1,
        "model_dir": str(model_dir),
        "test_root": str(test_root),
        "groups": {k: sorted(v) for k, v in groups.items()},
        "images": {"valid": valid, "failed": failed},
        "good": {
            "total": good_total,
            "pass": good_ok,
            "false_alarm_images": good_total - good_ok,
            "roi_total": good_roi_total,
            "false_alarm_rois": good_false_alarm_rois,
        },
        "ng": {
            "total": ng_total,
            "detected": ng_detected,
            "missed": ng_total - ng_detected,
            "target_tests": target_total,
            "target_detected": target_detected,
            "off_target_defect_predictions": off_target_defects,
        },
        "exact": {
            "total": exact_total,
            "correct": exact_ok,
        },
        "timing_seconds": {
            "total": total_s,
            "dino_total": total_dino,
            "mean_dino_per_valid_image": None if not valid else total_dino / valid,
        },
    }
    summary_path = out / "summary_full.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("", flush=True)
    print("=== Full evaluation summary ===", flush=True)
    print(f"Valid / failed:          {valid} / {failed}", flush=True)
    print(f"GOOD pass:               {good_ok}/{good_total}" if good_total else "GOOD pass:               -", flush=True)
    print(f"GOOD false-alarm ROIs:   {good_false_alarm_rois}/{good_roi_total}" if good_roi_total else "GOOD false-alarm ROIs:   -", flush=True)
    print(f"NG detected:             {ng_detected}/{ng_total}" if ng_total else "NG detected:             -", flush=True)
    print(f"Exact result:            {exact_ok}/{exact_total}" if exact_total else "Exact result:            -", flush=True)
    print(f"Off-target predictions:  {off_target_defects}", flush=True)
    print(f"Mean DINO/image:         {total_dino / valid:.3f}s" if valid else "Mean DINO/image:         -", flush=True)
    print(f"CSV:                     {csv_path}", flush=True)
    print(f"JSON:                    {summary_path}", flush=True)


if __name__ == "__main__":
    main()
