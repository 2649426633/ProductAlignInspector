from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.io_utils import read_image, write_image
from product_align_inspector.roi import crop_roi, validate_roi

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def scenario_of(relative: Path) -> str:
    parts = relative.parts
    if parts and parts[0].lower() == "good":
        return "good"
    if len(parts) >= 2 and parts[0].lower() == "ng":
        return parts[1]
    return parts[0] if parts else "unknown"


def load_checkpoint(path: Path, device: torch.device):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()
    return model, ckpt


def make_transform(input_size: int):
    def pad_square(image: Image.Image) -> Image.Image:
        w, h = image.size
        side = max(w, h)
        out = Image.new("RGB", (side, side), (255, 255, 255))
        out.paste(image, ((side - w) // 2, (side - h) // 2))
        return out
    return transforms.Compose([
        transforms.Lambda(pad_square),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def predict_screw_probability(model, crops: list[np.ndarray], tf, device: torch.device) -> np.ndarray:
    tensors = []
    for crop in crops:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensors.append(tf(Image.fromarray(rgb)))
    if not tensors:
        return np.empty(0, dtype=np.float32)
    batch = torch.stack(tensors, dim=0).to(device)
    with torch.inference_mode():
        p = torch.softmax(model(batch), dim=1)[:, 1]
    return p.cpu().numpy().astype(np.float32)


def jittered_rois(roi, width: int, height: int, amount: int):
    x, y, w, h = [int(v) for v in roi]
    offsets = [(0, 0)]
    if amount > 0:
        offsets += [(-amount, 0), (amount, 0), (0, -amount), (0, amount), (-amount, -amount), (-amount, amount), (amount, -amount), (amount, amount)]
    out = []
    for dx, dy in offsets:
        nx = max(0, min(width - w, x + dx))
        ny = max(0, min(height - h, y + dy))
        out.append(([nx, ny, w, h], dx, dy))
    return out


def draw_box(canvas: np.ndarray, roi, label: str, status: str):
    x, y, w, h = [int(v) for v in roi]
    color = (0, 190, 0) if status == "PASS" else (0, 0, 255)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
    cv2.putText(canvas, label, (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the independent semantic S/E ResNet18 models.")
    p.add_argument("--test-root", required=True)
    p.add_argument("--model-root", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--scenario", action="append")
    p.add_argument("--local-jitter", type=int, default=4)
    p.add_argument("--foreground-threshold", type=int, default=238)
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    test_root = Path(args.test_root).resolve()
    model_root = Path(args.model_root).resolve()
    reference_path = Path(args.reference).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    aligned_dir = output / "overlays_aligned"
    output.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)

    s_model, s_ckpt = load_checkpoint(model_root / "S" / "best.pt", device)
    e_model, e_ckpt = load_checkpoint(model_root / "E" / "best.pt", device)
    s_threshold = float(s_ckpt.get("threshold_screw", 0.5))
    e_threshold = float(e_ckpt.get("threshold_screw", 0.5))
    s_tf = make_transform(int(s_ckpt.get("input_size", 224)))
    e_tf = make_transform(int(e_ckpt.get("input_size", 224)))

    reference = read_image(reference_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    slots = [x for x in config.get("screw_slots", []) if bool(x.get("enabled", True))]
    s_slots = [x for x in slots if str(x.get("id", "")).upper().startswith("S")]
    e_slots = [x for x in slots if str(x.get("id", "")).upper().startswith("E")]

    filters = {str(v).strip().lower() for v in (args.scenario or []) if str(v).strip()}
    selected = []
    for path in collect_images(test_root):
        relative = path.relative_to(test_root)
        scenario = scenario_of(relative)
        if filters and scenario.lower() not in filters:
            continue
        selected.append((path, relative, scenario))

    print("=== Semantic ResNet18 S/E Evaluation ===")
    print(f"Device: {device}")
    print(f"Images: {len(selected)}")
    print(f"S threshold(screw): {s_threshold:.3f}")
    print(f"E threshold(screw): {e_threshold:.3f}")
    print(f"S ROIs: {[x['id'] for x in s_slots]}")
    print(f"E ROIs: {[x['id'] for x in e_slots]}")
    print("Interpretation: S PASS if screw probability is high; E PASS if screw probability is low.")

    align_cfg = ProductLocatorConfig(foreground_threshold=args.foreground_threshold)
    image_rows = []
    roi_rows = []

    for index, (path, relative, scenario) in enumerate(selected, 1):
        try:
            raw = read_image(path)
            alignment = align_to_reference(raw, reference, align_cfg)
            aligned = alignment.aligned
            h, w = aligned.shape[:2]
            canvas = aligned.copy()
            ng_rois = []

            for item in s_slots:
                roi = item["roi"]
                if not validate_roi(roi, w, h):
                    raise RuntimeError(f"invalid ROI {item['id']}: {roi}")
                candidates = jittered_rois(roi, w, h, args.local_jitter)
                crops = [crop_roi(aligned, c[0]) for c in candidates]
                probs = predict_screw_probability(s_model, crops, s_tf, device)
                best_i = int(np.argmax(probs))
                p_screw = float(probs[best_i])
                status = "PASS" if p_screw >= s_threshold else "NG"
                if status == "NG":
                    ng_rois.append(str(item["id"]))
                used_roi, dx, dy = candidates[best_i]
                draw_box(canvas, used_roi, f"{item['id']} {status} P(screw)={p_screw:.3f}", status)
                roi_rows.append({"relative_path": relative.as_posix(), "scenario": scenario, "roi_id": item["id"], "group": "S", "p_screw": p_screw, "threshold": s_threshold, "status": status, "dx": dx, "dy": dy})

            for item in e_slots:
                roi = item["roi"]
                if not validate_roi(roi, w, h):
                    raise RuntimeError(f"invalid ROI {item['id']}: {roi}")
                candidates = jittered_rois(roi, w, h, args.local_jitter)
                crops = [crop_roi(aligned, c[0]) for c in candidates]
                probs = predict_screw_probability(e_model, crops, e_tf, device)
                best_i = int(np.argmin(probs))
                p_screw = float(probs[best_i])
                status = "PASS" if p_screw < e_threshold else "NG"
                if status == "NG":
                    ng_rois.append(str(item["id"]))
                used_roi, dx, dy = candidates[best_i]
                draw_box(canvas, used_roi, f"{item['id']} {status} P(screw)={p_screw:.3f}", status)
                roi_rows.append({"relative_path": relative.as_posix(), "scenario": scenario, "roi_id": item["id"], "group": "E", "p_screw": p_screw, "threshold": e_threshold, "status": status, "dx": dx, "dy": dy})

            final_status = "NG" if ng_rois else "PASS"
            overlay_rel = Path("overlays_aligned") / relative.with_suffix(".jpg")
            overlay_path = output / overlay_rel
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            write_image(overlay_path, canvas)
            image_rows.append({"relative_path": relative.as_posix(), "scenario": scenario, "final_status": final_status, "ng_rois": ",".join(ng_rois), "alignment_method": alignment.method, "feature_inlier_ratio": alignment.feature_inlier_ratio, "ecc_score": "" if alignment.ecc_score is None else alignment.ecc_score, "overlay": overlay_rel.as_posix(), "error": ""})
            print(f"[{index}/{len(selected)}] {relative.as_posix()} -> {final_status} | NG={ng_rois or '-'} | align={alignment.method} ecc={alignment.ecc_score}")
        except Exception as exc:
            image_rows.append({"relative_path": relative.as_posix(), "scenario": scenario, "final_status": "RETRY", "ng_rois": "", "alignment_method": "", "feature_inlier_ratio": "", "ecc_score": "", "overlay": "", "error": str(exc)})
            print(f"[{index}/{len(selected)}] {relative.as_posix()} -> RETRY: {exc}")

    image_csv = output / "image_summary.csv"
    with image_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "scenario", "final_status", "ng_rois", "alignment_method", "feature_inlier_ratio", "ecc_score", "overlay", "error"])
        writer.writeheader(); writer.writerows(image_rows)
    roi_csv = output / "roi_scores.csv"
    with roi_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "scenario", "roi_id", "group", "p_screw", "threshold", "status", "dx", "dy"])
        writer.writeheader(); writer.writerows(roi_rows)

    counts = {k: sum(1 for r in image_rows if r["final_status"] == k) for k in ("PASS", "NG", "RETRY")}
    (output / "summary.json").write_text(json.dumps({"images": len(image_rows), "counts": counts, "S_threshold_screw": s_threshold, "E_threshold_screw": e_threshold}, indent=2), encoding="utf-8")
    print("\n=== Evaluation complete ===")
    print(f"PASS/NG/RETRY: {counts['PASS']}/{counts['NG']}/{counts['RETRY']}")
    print(f"Aligned overlays: {aligned_dir}")
    print(f"ROI scores: {roi_csv}")


if __name__ == "__main__":
    main()
