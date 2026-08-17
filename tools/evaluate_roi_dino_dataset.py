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


def collect_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def parse_named_roi_scenario(folder_name: str, known_ids: set[str]) -> set[str] | None:
    """Return an exact ROI defect set for names such as missing_S01 or missing_S01+missing_S02.

    Returns None when the folder only provides an image-level NG label, such as
    missing_screws or excess_screws.
    """
    lower_name = folder_name.strip().lower()
    if lower_name == "all_empty":
        return set(known_ids)
    if lower_name in {"missing_screws", "missing_screw", "excess_screws", "excess_screw"}:
        return None

    name_map = {rid.lower(): rid for rid in known_ids}
    defects: set[str] = set()
    for token in folder_name.split("+"):
        token = token.strip()
        lower = token.lower()
        roi_part = None
        for prefix in ("missing_", "defect_", "ng_", "anomaly_", "bad_"):
            if lower.startswith(prefix):
                roi_part = token[len(prefix):]
                break
        if roi_part is None:
            roi_part = token
        exact = name_map.get(roi_part.lower())
        if exact is None:
            return None
        defects.add(exact)
    return defects


def fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.6f}"


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate the current dataset without renaming good/missing_screws/excess_screws/all_empty folders."
    )
    p.add_argument("--test-root", required=True)
    p.add_argument("--model-dir", default="artifacts/roi_dino_patchcore")
    p.add_argument("--reference")
    p.add_argument("--dino-repo")
    p.add_argument("--dino-weights")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="artifacts/roi_dino_evaluation_dataset")
    p.add_argument("--threshold-scale", type=float, default=1.0)
    args = p.parse_args()

    if args.threshold_scale <= 0:
        raise SystemExit("--threshold-scale must be > 0")

    test_root = Path(args.test_root)
    good_dir = test_root / "good"
    ng_root = test_root / "ng"

    model_dir = Path(args.model_dir)
    manifest = read_model_manifest(model_dir)
    models = [load_roi_model(model_dir, str(item["id"])) for item in manifest.get("rois", [])]
    if not models:
        raise RuntimeError("No ROI models in model.json")
    known_ids = {m.roi_id for m in models}

    # truth_mode:
    #   exact  -> exact per-ROI ground truth is known.
    #   any_ng -> folder says the image is NG, but does not say which current ROI is defective.
    scenarios: list[tuple[str, Path, str, set[str]]] = []
    for image in collect_images(good_dir):
        scenarios.append(("GOOD", image, "exact", set()))

    if ng_root.is_dir():
        for scenario_dir in sorted(p for p in ng_root.iterdir() if p.is_dir()):
            exact_rois = parse_named_roi_scenario(scenario_dir.name, known_ids)
            truth_mode = "exact" if exact_rois is not None else "any_ng"
            truth_rois = set() if exact_rois is None else exact_rois
            for image in collect_images(scenario_dir):
                scenarios.append((scenario_dir.name, image, truth_mode, truth_rois))

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
    align_cfg = ProductLocatorConfig(
        foreground_threshold=int(align_meta.get("foreground_threshold", 238))
    )
    min_inlier_ratio = float(align_meta.get("min_inlier_ratio", 0.25))
    dino = DINOv2Adapter(device=args.device, config=dino_cfg, project_root=REPO_ROOT)
    dino.load()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    per_roi = {m.roi_id: {"NORMAL": [], "DEFECT": []} for m in models}
    scenario_summary = defaultdict(lambda: {"count": 0, "correct": 0, "failed": 0, "truth_mode": ""})

    valid_images = 0
    failed_images = 0
    exact_total = 0
    exact_correct = 0
    generic_ng_total = 0
    generic_ng_detected = 0
    false_alarm_rois = 0
    missed_defect_rois = 0
    normal_roi_total = 0
    defect_roi_total = 0
    total_dino = 0.0
    total_start = time.perf_counter()

    print("=== ROI DINOv2 / PatchCore Dataset Evaluation ===", flush=True)
    print(f"Total images: {len(scenarios)}", flush=True)
    print(f"ROI IDs: {[m.roi_id for m in models]}", flush=True)
    print("Folder policy:", flush=True)
    print("  good          -> exact GOOD", flush=True)
    print("  all_empty     -> all current ROI IDs are DEFECT", flush=True)
    print("  missing_screws/excess_screws -> image-level NG only (ROI truth unknown)", flush=True)
    print("", flush=True)

    for idx, (scenario, image_path, truth_mode, truth_rois) in enumerate(scenarios, 1):
        scenario_summary[scenario]["truth_mode"] = truth_mode
        scenario_summary[scenario]["count"] += 1
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

            predicted_defects: set[str] = set()
            detail = []
            for model, tokens in zip(models, token_batch):
                score, _map, patch_stats = score_patch_tokens(
                    tokens,
                    model.memory,
                    patch_grid=model.patch_grid,
                    top_fraction=model.score_top_fraction,
                )
                threshold = None if model.threshold is None else float(model.threshold) * args.threshold_scale
                pred_defect = threshold is not None and score > threshold
                if pred_defect:
                    predicted_defects.add(model.roi_id)

                truth_state = "UNKNOWN"
                roi_correct: bool | str = ""
                if truth_mode == "exact":
                    truth_defect = model.roi_id in truth_rois
                    truth_state = "DEFECT" if truth_defect else "NORMAL"
                    per_roi[model.roi_id][truth_state].append(float(score))
                    roi_correct = pred_defect == truth_defect
                    if truth_defect:
                        defect_roi_total += 1
                        if not pred_defect:
                            missed_defect_rois += 1
                    else:
                        normal_roi_total += 1
                        if pred_defect:
                            false_alarm_rois += 1

                rows.append({
                    "scenario": scenario,
                    "source": str(image_path),
                    "truth_mode": truth_mode,
                    "roi_id": model.roi_id,
                    "truth": truth_state,
                    "prediction": "DEFECT" if pred_defect else "NORMAL",
                    "correct": roi_correct,
                    "score": float(score),
                    "threshold": threshold,
                    "score_over_threshold": None if threshold is None else float(score / threshold),
                    "max_patch": float(patch_stats["max"]),
                    "alignment_method": alignment.method,
                    "feature_inlier_ratio": float(alignment.feature_inlier_ratio),
                    "ecc_score": "" if alignment.ecc_score is None else float(alignment.ecc_score),
                    "dino_seconds": dino_s,
                    "status": "ok",
                    "error": "",
                })
                detail.append(f"{model.roi_id}={score:.4f}/t={fmt(threshold)} {'D' if pred_defect else 'N'}")

            valid_images += 1
            if truth_mode == "exact":
                exact_total += 1
                image_ok = predicted_defects == truth_rois
                exact_correct += int(image_ok)
                truth_text = "GOOD" if not truth_rois else "+".join(sorted(truth_rois))
            else:
                generic_ng_total += 1
                image_ok = bool(predicted_defects)
                generic_ng_detected += int(image_ok)
                truth_text = "NG(any ROI)"

            scenario_summary[scenario]["correct"] += int(image_ok)
            pred_text = "GOOD" if not predicted_defects else "+".join(sorted(predicted_defects))
            print(
                f"[{idx}/{len(scenarios)}] {scenario:<18} {image_path.name:<26} "
                f"truth={truth_text:<12} pred={pred_text:<10} {'OK' if image_ok else 'MISS'} "
                f"align={alignment.method} {alignment.feature_inlier_ratio*100:.1f}%",
                flush=True,
            )
            print("    " + "  ".join(detail), flush=True)

        except Exception as exc:
            failed_images += 1
            scenario_summary[scenario]["failed"] += 1
            rows.append({
                "scenario": scenario,
                "source": str(image_path),
                "truth_mode": truth_mode,
                "roi_id": "",
                "truth": "",
                "prediction": "",
                "correct": "",
                "score": "",
                "threshold": "",
                "score_over_threshold": "",
                "max_patch": "",
                "alignment_method": "",
                "feature_inlier_ratio": "",
                "ecc_score": "",
                "dino_seconds": "",
                "status": "failed",
                "error": str(exc),
            })
            print(f"[{idx}/{len(scenarios)}] {image_path.name} -> FAILED: {exc}", flush=True)

    csv_path = out / "scores_dataset.csv"
    fieldnames = [
        "scenario", "source", "truth_mode", "roi_id", "truth", "prediction", "correct",
        "score", "threshold", "score_over_threshold", "max_patch", "alignment_method",
        "feature_inlier_ratio", "ecc_score", "dino_seconds", "status", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("", flush=True)
    print("=== Exact-label per-ROI separation ===", flush=True)
    print(f"{'ROI':<8} {'NORMAL min':>11} {'NORMAL mean':>11} {'NORMAL max':>11} {'DEFECT min':>11} {'DEFECT mean':>11} {'DEFECT max':>11} {'Threshold':>11}", flush=True)
    print("-" * 101, flush=True)

    roi_summary = {}
    for model in models:
        normal = per_roi[model.roi_id]["NORMAL"]
        defect = per_roi[model.roi_id]["DEFECT"]
        ns, ds = stats(normal), stats(defect)
        threshold = None if model.threshold is None else float(model.threshold) * args.threshold_scale
        gap = None
        separated = None
        if normal and defect:
            gap = float(min(defect) - max(normal))
            separated = bool(gap > 0)
        roi_summary[model.roi_id] = {
            "threshold": threshold,
            "normal": ns,
            "defect": ds,
            "gap_defect_min_minus_normal_max": gap,
            "separated_without_overlap": separated,
        }
        print(
            f"{model.roi_id:<8} {fmt(ns['min']):>11} {fmt(ns['mean']):>11} {fmt(ns['max']):>11} "
            f"{fmt(ds['min']):>11} {fmt(ds['mean']):>11} {fmt(ds['max']):>11} {fmt(threshold):>11}",
            flush=True,
        )

    print("", flush=True)
    print("=== Scenario summary ===", flush=True)
    for scenario, s in sorted(scenario_summary.items()):
        usable = int(s["count"]) - int(s["failed"])
        print(
            f"{scenario:<18} mode={s['truth_mode']:<6} correct/detected={s['correct']}/{usable} failed={s['failed']}",
            flush=True,
        )

    total_s = time.perf_counter() - total_start
    summary = {
        "schema_version": 1,
        "model_dir": str(model_dir),
        "test_root": str(test_root),
        "valid_images": valid_images,
        "failed_images": failed_images,
        "exact_label_images": {
            "count": exact_total,
            "correct_exact_roi_set": exact_correct,
            "accuracy": None if not exact_total else exact_correct / exact_total,
        },
        "generic_ng_images": {
            "count": generic_ng_total,
            "detected_by_any_current_roi": generic_ng_detected,
            "detection_rate": None if not generic_ng_total else generic_ng_detected / generic_ng_total,
            "note": "ROI location ground truth is unknown for these folders; excluded from strict per-ROI metrics.",
        },
        "roi_level_exact_labels_only": {
            "normal_roi_total": normal_roi_total,
            "defect_roi_total": defect_roi_total,
            "false_alarm_rois": false_alarm_rois,
            "missed_defect_rois": missed_defect_rois,
            "normal_false_alarm_rate": None if not normal_roi_total else false_alarm_rois / normal_roi_total,
            "defect_miss_rate": None if not defect_roi_total else missed_defect_rois / defect_roi_total,
        },
        "scenarios": dict(scenario_summary),
        "timing_seconds": {
            "total": total_s,
            "dino_total": total_dino,
            "mean_dino_per_valid_image": None if not valid_images else total_dino / valid_images,
        },
        "rois": roi_summary,
    }
    summary_path = out / "summary_dataset.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("", flush=True)
    print("=== Dataset summary ===", flush=True)
    print(f"Valid images:          {valid_images}", flush=True)
    print(f"Failed images:         {failed_images}", flush=True)
    print(f"Exact ROI-set result:  {exact_correct}/{exact_total}" if exact_total else "Exact ROI-set result:  -", flush=True)
    print(f"Generic NG detected:   {generic_ng_detected}/{generic_ng_total}" if generic_ng_total else "Generic NG detected:   -", flush=True)
    print(f"False-alarm ROIs:      {false_alarm_rois}/{normal_roi_total}" if normal_roi_total else "False-alarm ROIs:      -", flush=True)
    print(f"Missed defect ROIs:    {missed_defect_rois}/{defect_roi_total}" if defect_roi_total else "Missed defect ROIs:    -", flush=True)
    print(f"Mean DINO/image:       {total_dino / valid_images:.3f}s" if valid_images else "Mean DINO/image:       -", flush=True)
    print(f"CSV:                   {csv_path}", flush=True)
    print(f"JSON:                  {summary_path}", flush=True)
    print("", flush=True)
    print("NOTE: excess_screws is only image-level NG in this report. If extra screws are outside the current S01/S02 ROIs,", flush=True)
    print("      a miss does not prove DINO failed; add empty-position ROIs (E01/E02/...) before strict excess-screw evaluation.", flush=True)


if __name__ == "__main__":
    main()
