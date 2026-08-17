from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.anomaly.dinov2_adapter import DINOv2Adapter, DINOv2Config
from product_align_inspector.anomaly.roi_patchcore import load_roi_model, read_model_manifest, score_patch_tokens
from product_align_inspector.decision_rules import load_decision_multipliers, roi_multiplier
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
    lower = name.strip().lower()
    if lower == "good":
        return "GOOD", set(), False
    if lower in {"all_empty", "all_missing", "all_screws_empty"}:
        return "SCREW_ALL", set(groups["SCREW"]), True
    if lower in {"missing_screws", "missing_screw"}:
        return "SCREW_ANY", set(groups["SCREW"]), False
    if lower in {"excess_screws", "extra_screws", "excess_screw", "extra_screw"}:
        return "EMPTY_ANY", set(groups["EMPTY"]), False
    return "PRODUCT_ANY", set(groups["SCREW"]) | set(groups["EMPTY"]), False


def _resolve_rules_path(text: str | None) -> Path | None:
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate product rules: S=must have screw, E=must be empty; spring ROIs are ignored."
    )
    p.add_argument("--test-root", required=True)
    p.add_argument("--model-dir", default="artifacts/roi_dino_full")
    p.add_argument("--reference")
    p.add_argument("--dino-repo")
    p.add_argument("--dino-weights")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="artifacts/roi_dino_full_evaluation")
    p.add_argument("--threshold-scale", type=float, default=1.0)
    p.add_argument(
        "--decision-rules",
        default="configs/brunei_decision_rules.json",
        help="JSON with per-ROI threshold multipliers",
    )
    args = p.parse_args()

    if args.threshold_scale <= 0:
        raise SystemExit("--threshold-scale must be > 0")

    rules_path = _resolve_rules_path(args.decision_rules)
    default_multiplier, roi_multipliers = load_decision_multipliers(rules_path)

    test_root = Path(args.test_root)
    model_dir = Path(args.model_dir)
    manifest = read_model_manifest(model_dir)
    all_models = [load_roi_model(model_dir, str(item["id"])) for item in manifest.get("rois", [])]
    if not all_models:
        raise RuntimeError("No ROI models in model.json")

    groups: dict[str, set[str]] = {"SCREW": set(), "EMPTY": set(), "SPRING": set(), "OTHER": set()}
    for model in all_models:
        groups[classify_group(model.roi_id)].add(model.roi_id)

    models = [m for m in all_models if classify_group(m.roi_id) in {"SCREW", "EMPTY"}]
    ignored_models = [m for m in all_models if m not in models]
    if not models:
        raise RuntimeError("No S/E ROI models found. Expected Sxx and/or Exx banks.")

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
    dino = DINOv2Adapter(device=args.device, config=dino_cfg, project_root=REPO_ROOT)
    dino.load()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    valid = 0
    failed = 0
    good_total = 0
    good_ok = 0
    good_false_alarm_rois = 0
    good_roi_total = 0
    ng_total = 0
    ng_detected = 0
    scenario_stats: dict[str, dict[str, int]] = {}
    total_dino = 0.0
    t_all = time.perf_counter()

    print("=== Product Rule Evaluation ===", flush=True)
    print("Rule: Sxx must HAVE screw; Exx must be EMPTY.", flush=True)
    print("SPRING ROIs are ignored for now.", flush=True)
    print(f"Images:       {len(scenarios)}", flush=True)
    print(f"SCREW ROIs:   {sorted(groups['SCREW'])}", flush=True)
    print(f"EMPTY ROIs:   {sorted(groups['EMPTY'])}", flush=True)
    print(f"Ignored ROIs: {[m.roi_id for m in ignored_models]}", flush=True)
    print(f"Decision ROI count: {len(models)}", flush=True)
    print(f"Rules:        {rules_path if rules_path and rules_path.is_file() else 'default x1.0'}", flush=True)
    print(f"Margins:      {roi_multipliers if roi_multipliers else '-'}", flush=True)
    print("", flush=True)

    for idx, (scenario, image_path) in enumerate(scenarios, 1):
        mode, targets, require_all = scenario_policy(scenario, groups)
        try:
            raw = read_image(image_path)
            alignment = align_to_reference(raw, reference, align_cfg)

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
            for model, tokens in zip(models, token_batch):
                score, _heat, _stats = score_patch_tokens(
                    tokens,
                    model.memory,
                    patch_grid=model.patch_grid,
                    top_fraction=model.score_top_fraction,
                )
                base_threshold = None if model.threshold is None else float(model.threshold)
                multiplier = roi_multiplier(model.roi_id, default_multiplier, roi_multipliers)
                threshold = None if base_threshold is None else base_threshold * args.threshold_scale * multiplier
                is_defect = threshold is not None and score > threshold
                if is_defect:
                    predicted.add(model.roi_id)

                roi_group = classify_group(model.roi_id)
                if roi_group == "SCREW":
                    expected = "SCREW_PRESENT"
                    defect_meaning = "MISSING_OR_WRONG_SCREW"
                else:
                    expected = "EMPTY"
                    defect_meaning = "UNEXPECTED_SCREW_OR_OBJECT"

                rows.append({
                    "scenario": scenario,
                    "mode": mode,
                    "source": str(image_path),
                    "roi_id": model.roi_id,
                    "roi_group": roi_group,
                    "expected": expected,
                    "prediction": "DEFECT" if is_defect else "NORMAL",
                    "defect_meaning": defect_meaning if is_defect else "",
                    "score": float(score),
                    "base_threshold": base_threshold,
                    "decision_multiplier": float(multiplier),
                    "decision_threshold": threshold,
                    "score_over_decision_threshold": None if threshold is None else float(score / threshold),
                    "alignment_method": alignment.method,
                    "feature_inlier_ratio": float(alignment.feature_inlier_ratio),
                    "ecc_score": "" if alignment.ecc_score is None else float(alignment.ecc_score),
                    "dino_seconds": dino_s,
                    "status": "ok",
                    "error": "",
                })

            valid += 1
            if mode == "GOOD":
                good_total += 1
                good_roi_total += len(models)
                good_false_alarm_rois += len(predicted)
                image_ok = not predicted
                good_ok += int(image_ok)
                result = "PASS" if image_ok else "FALSE_ALARM"
            else:
                ng_total += 1
                target_preds = predicted & targets
                if require_all:
                    detected = bool(targets) and targets.issubset(predicted)
                else:
                    detected = bool(target_preds) if targets else bool(predicted)
                ng_detected += int(detected)
                result = "DETECTED" if detected else "MISS"
                stat = scenario_stats.setdefault(scenario, {"total": 0, "detected": 0})
                stat["total"] += 1
                stat["detected"] += int(detected)

            screw_pred = sorted(predicted & groups["SCREW"])
            empty_pred = sorted(predicted & groups["EMPTY"])
            print(
                f"[{idx}/{len(scenarios)}] {scenario:<18} {image_path.name:<24} "
                f"S={'+'.join(screw_pred) or '-':<10} E={'+'.join(empty_pred) or '-':<32} {result}",
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
                "expected": "",
                "prediction": "",
                "defect_meaning": "",
                "score": "",
                "base_threshold": "",
                "decision_multiplier": "",
                "decision_threshold": "",
                "score_over_decision_threshold": "",
                "alignment_method": "",
                "feature_inlier_ratio": "",
                "ecc_score": "",
                "dino_seconds": "",
                "status": "failed",
                "error": str(exc),
            })
            print(f"[{idx}/{len(scenarios)}] {image_path.name} -> FAILED: {exc}", flush=True)

    csv_path = out / "scores_product_rules.csv"
    fieldnames = [
        "scenario", "mode", "source", "roi_id", "roi_group", "expected", "prediction",
        "defect_meaning", "score", "base_threshold", "decision_multiplier", "decision_threshold",
        "score_over_decision_threshold", "alignment_method", "feature_inlier_ratio", "ecc_score",
        "dino_seconds", "status", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_s = time.perf_counter() - t_all
    summary = {
        "schema_version": 3,
        "product_rules": {
            "S": "must_have_screw",
            "E": "must_be_empty",
            "SPRING": "ignored_for_now",
            "decision_rules_file": "" if rules_path is None else str(rules_path),
            "default_threshold_multiplier": default_multiplier,
            "roi_threshold_multipliers": roi_multipliers,
        },
        "model_dir": str(model_dir),
        "test_root": str(test_root),
        "decision_rois": [m.roi_id for m in models],
        "ignored_rois": [m.roi_id for m in ignored_models],
        "images": {"valid": valid, "failed": failed},
        "good": {
            "total": good_total,
            "pass": good_ok,
            "false_alarm_images": good_total - good_ok,
            "decision_roi_total": good_roi_total,
            "false_alarm_rois": good_false_alarm_rois,
        },
        "ng": {
            "total": ng_total,
            "detected": ng_detected,
            "missed": ng_total - ng_detected,
        },
        "scenario_detection": scenario_stats,
        "timing_seconds": {
            "total": total_s,
            "dino_total": total_dino,
            "mean_dino_per_valid_image": None if not valid else total_dino / valid,
        },
    }
    summary_path = out / "summary_product_rules.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("", flush=True)
    print("=== Product-rule evaluation summary ===", flush=True)
    print(f"Valid / failed:        {valid} / {failed}", flush=True)
    print(f"GOOD pass:             {good_ok}/{good_total}" if good_total else "GOOD pass:             -", flush=True)
    print(
        f"GOOD false-alarm ROI:  {good_false_alarm_rois}/{good_roi_total} (S/E only)"
        if good_roi_total else "GOOD false-alarm ROI:  -",
        flush=True,
    )
    for name in sorted(scenario_stats):
        st = scenario_stats[name]
        print(f"{name:<22} {st['detected']}/{st['total']} detected", flush=True)
    print(f"NG detected overall:   {ng_detected}/{ng_total}" if ng_total else "NG detected overall:   -", flush=True)
    print(f"Mean DINO/image:       {total_dino / valid:.3f}s" if valid else "Mean DINO/image:       -", flush=True)
    print(f"CSV:                   {csv_path}", flush=True)
    print(f"JSON:                  {summary_path}", flush=True)


if __name__ == "__main__":
    main()
