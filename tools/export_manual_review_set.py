from __future__ import annotations

import argparse
import csv
import html
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
from product_align_inspector.anomaly.roi_patchcore import load_roi_model, read_model_manifest, score_patch_tokens
from product_align_inspector.decision_rules import load_decision_multipliers, roi_multiplier
from product_align_inspector.io_utils import read_image, write_image
from product_align_inspector.roi import crop_roi, validate_roi

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def group_of(roi_id: str) -> str:
    rid = roi_id.upper()
    if rid.startswith("SPRING"):
        return "SPRING"
    if rid.startswith("E"):
        return "EMPTY"
    if rid.startswith("S"):
        return "SCREW"
    return "OTHER"


def collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def scenario_of(relative: Path) -> str:
    parts = relative.parts
    if not parts:
        return "unknown"
    if parts[0].lower() == "good":
        return "good"
    if len(parts) >= 2 and parts[0].lower() == "ng":
        return parts[1]
    return parts[0]


def resolve_from_repo(text: str | None) -> Path | None:
    if not text:
        return None
    path = Path(text)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def draw_roi(canvas: np.ndarray, roi: tuple[int, int, int, int], label: str, status: str) -> None:
    x, y, w, h = roi
    color = (0, 190, 0) if status == "PASS" else (0, 0, 255)
    thickness = max(3, int(round(min(canvas.shape[:2]) / 700)))
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)

    font_scale = max(0.55, min(1.0, min(canvas.shape[:2]) / 1900.0))
    text_thickness = max(1, thickness - 1)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
    tx = max(0, min(x, canvas.shape[1] - tw - 10))
    ty = y - 8
    if ty - th - baseline - 4 < 0:
        ty = min(canvas.shape[0] - 5, y + th + baseline + 12)
    cv2.rectangle(
        canvas,
        (tx, max(0, ty - th - baseline - 4)),
        (min(canvas.shape[1] - 1, tx + tw + 8), min(canvas.shape[0] - 1, ty + 3)),
        color,
        -1,
    )
    cv2.putText(
        canvas,
        label,
        (tx + 4, ty - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def draw_banner(canvas: np.ndarray, filename: str, final_status: str, ng_rois: list[str]) -> None:
    if final_status == "PASS":
        color = (0, 150, 0)
    elif final_status == "NG":
        color = (0, 0, 220)
    else:
        color = (0, 180, 255)

    banner_h = max(72, int(round(canvas.shape[0] * 0.075)))
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], banner_h), color, -1)
    text = f"{filename}   FINAL: {final_status}"
    if ng_rois:
        text += "   NG: " + ",".join(ng_rois)
    cv2.putText(
        canvas,
        text,
        (20, int(banner_h * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.8, banner_h / 72.0),
        (255, 255, 255),
        max(2, int(round(banner_h / 34))),
        cv2.LINE_AA,
    )


def write_gallery(output: Path, rows: list[dict[str, object]]) -> Path:
    cards: list[str] = []
    for row in rows:
        overlay_rel = str(row.get("overlay", ""))
        if not overlay_rel:
            continue
        status = html.escape(str(row.get("final_status", "")))
        relative = html.escape(str(row.get("relative_path", "")))
        ng_rois = html.escape(str(row.get("ng_rois", "")))
        src = html.escape(overlay_rel.replace("\\", "/"))
        cards.append(
            f'<article class="card"><a href="{src}" target="_blank">'
            f'<img loading="lazy" src="{src}" alt="{relative}"></a>'
            f'<div class="meta"><b>{relative}</b><br>Status: {status}<br>NG: {ng_rois or "-"}</div></article>'
        )

    page = """<!doctype html>
<html><head><meta charset="utf-8"><title>Manual Review</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;margin:18px;background:#f4f4f4;color:#222}
h1{margin:0 0 16px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px #0001}
.card img{width:100%;display:block;background:#eee}.meta{padding:10px;line-height:1.45;font-size:14px}
</style></head><body><h1>Manual Review Gallery</h1><div class="grid">""" + "\n".join(cards) + "</div></body></html>"
    path = output / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Generate marked review images for every test image.")
    p.add_argument("--test-root", required=True)
    p.add_argument("--model-dir", default="artifacts/roi_dino_full")
    p.add_argument("--reference")
    p.add_argument("--dino-repo")
    p.add_argument("--dino-weights")
    p.add_argument("--device", default="auto")
    p.add_argument("--decision-rules", default="configs/brunei_decision_rules.json")
    p.add_argument("--threshold-scale", type=float, default=1.0)
    p.add_argument("--output", default="artifacts/manual_review")
    args = p.parse_args()

    if args.threshold_scale <= 0:
        raise SystemExit("--threshold-scale must be > 0")

    test_root = Path(args.test_root).resolve()
    model_dir = Path(args.model_dir).resolve()
    output = Path(args.output).resolve()
    overlays_root = output / "overlays"
    output.mkdir(parents=True, exist_ok=True)
    overlays_root.mkdir(parents=True, exist_ok=True)

    images = collect_images(test_root)
    if not images:
        raise SystemExit(f"No test images found: {test_root}")

    manifest = read_model_manifest(model_dir)
    reference_path = Path(args.reference or manifest["reference_image"]).resolve()
    reference = read_image(reference_path)
    align_meta = manifest.get("alignment", {})
    align_cfg = ProductLocatorConfig(foreground_threshold=int(align_meta.get("foreground_threshold", 238)))

    rules_path = resolve_from_repo(args.decision_rules)
    default_multiplier, multipliers = load_decision_multipliers(rules_path)

    all_models = [load_roi_model(model_dir, str(item["id"])) for item in manifest.get("rois", [])]
    models = [m for m in all_models if group_of(m.roi_id) in {"SCREW", "EMPTY"}]
    if not models:
        raise RuntimeError("No S/E ROI models found")

    dino_meta = manifest["dino"]
    dino_cfg = DINOv2Config(
        model_name=str(dino_meta.get("model_name", "dinov2_vits14")),
        image_size=int(dino_meta.get("image_size", 224)),
        embedding_dim=int(dino_meta.get("embedding_dim", 384)),
        repo_dir=str(args.dino_repo or dino_meta.get("repo_dir", "third_party/dinov2")),
        weights_path=str(args.dino_weights or dino_meta.get("weights_path", "weights/dinov2_vits14_pretrain.pth")),
        pad_value=int(dino_meta.get("preprocess", {}).get("pad_value", 255)),
    )
    dino = DINOv2Adapter(device=args.device, config=dino_cfg, project_root=REPO_ROOT)
    dino.load()

    print("=== Manual Review Export ===", flush=True)
    print(f"Images:        {len(images)}", flush=True)
    print(f"Decision ROIs: {[m.roi_id for m in models]}", flush=True)
    print(f"Rules:         {rules_path}", flush=True)
    print(f"Output:        {output}", flush=True)
    print("Green=PASS, Red=NG. Label ratio is score/decision-threshold.", flush=True)
    print("", flush=True)

    rows: list[dict[str, object]] = []
    roi_rows: list[dict[str, object]] = []
    t_all = time.perf_counter()

    for index, image_path in enumerate(images, 1):
        relative = image_path.relative_to(test_root)
        scenario = scenario_of(relative)
        overlay_path = overlays_root / relative.with_suffix(".png")
        overlay_path.parent.mkdir(parents=True, exist_ok=True)

        row: dict[str, object] = {
            "relative_path": str(relative).replace("\\", "/"),
            "scenario": scenario,
            "final_status": "RETRY",
            "ng_rois": "",
            "ng_roi_count": 0,
            "alignment_method": "",
            "ecc_score": "",
            "overlay": "",
            "error": "",
        }

        try:
            raw = read_image(image_path)
            alignment = align_to_reference(raw, reference, align_cfg)
            aligned = alignment.aligned
            h, w = aligned.shape[:2]

            crops: list[np.ndarray] = []
            for model in models:
                if not validate_roi(model.roi, w, h):
                    raise RuntimeError(f"Invalid ROI {model.roi_id}: {model.roi}")
                crops.append(crop_roi(aligned, model.roi))

            token_batch = dino.patch_tokens_batch(crops)
            preview = aligned.copy()
            ng_rois: list[str] = []

            for model, tokens in zip(models, token_batch):
                score, _heat, _stats = score_patch_tokens(
                    tokens,
                    model.memory,
                    patch_grid=model.patch_grid,
                    top_fraction=model.score_top_fraction,
                )
                base_threshold = None if model.threshold is None else float(model.threshold)
                multiplier = roi_multiplier(model.roi_id, default_multiplier, multipliers)
                decision_threshold = None
                if base_threshold is not None:
                    decision_threshold = base_threshold * float(args.threshold_scale) * float(multiplier)

                is_ng = decision_threshold is not None and score > decision_threshold
                status = "NG" if is_ng else "PASS"
                if is_ng:
                    ng_rois.append(model.roi_id)

                ratio = None if not decision_threshold else float(score / decision_threshold)
                ratio_text = "-" if ratio is None else f"{ratio:.2f}x"
                draw_roi(preview, model.roi, f"{model.roi_id} {status} {ratio_text}", status)

                roi_rows.append({
                    "relative_path": str(relative).replace("\\", "/"),
                    "scenario": scenario,
                    "roi_id": model.roi_id,
                    "group": group_of(model.roi_id),
                    "status": status,
                    "score": float(score),
                    "base_threshold": base_threshold,
                    "multiplier": float(multiplier),
                    "decision_threshold": decision_threshold,
                    "score_over_threshold": ratio,
                })

            final_status = "NG" if ng_rois else "PASS"
            draw_banner(preview, image_path.name, final_status, ng_rois)
            write_image(overlay_path, preview)

            row.update({
                "final_status": final_status,
                "ng_rois": ";".join(ng_rois),
                "ng_roi_count": len(ng_rois),
                "alignment_method": alignment.method,
                "ecc_score": "" if alignment.ecc_score is None else float(alignment.ecc_score),
                "overlay": str(overlay_path.relative_to(output)).replace("\\", "/"),
            })
            print(
                f"[{index:>3}/{len(images)}] {str(relative):<50} {final_status:<5} "
                f"NG={','.join(ng_rois) or '-'}",
                flush=True,
            )
        except Exception as exc:
            row["error"] = str(exc)
            print(f"[{index:>3}/{len(images)}] {relative} -> RETRY: {exc}", flush=True)

        rows.append(row)

    summary_csv = output / "review_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    roi_csv = output / "review_rois.csv"
    if roi_rows:
        with roi_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(roi_rows[0].keys()))
            writer.writeheader()
            writer.writerows(roi_rows)

    summary = {
        "schema_version": 1,
        "images": len(rows),
        "pass": sum(1 for row in rows if row["final_status"] == "PASS"),
        "ng": sum(1 for row in rows if row["final_status"] == "NG"),
        "retry": sum(1 for row in rows if row["final_status"] == "RETRY"),
        "decision_rois": [m.roi_id for m in models],
        "rules": None if rules_path is None else str(rules_path),
        "output": str(output),
        "elapsed_seconds": time.perf_counter() - t_all,
    }
    summary_json = output / "review_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    gallery = write_gallery(output, rows)

    print("", flush=True)
    print("=== Manual Review Ready ===", flush=True)
    print(f"Marked images: {overlays_root}", flush=True)
    print(f"Gallery:       {gallery}", flush=True)
    print(f"Image CSV:     {summary_csv}", flush=True)
    print(f"ROI CSV:       {roi_csv}", flush=True)
    print(f"Summary JSON:  {summary_json}", flush=True)


if __name__ == "__main__":
    main()
