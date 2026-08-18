from __future__ import annotations

import argparse
import csv
import html
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
    build_feature_extractor,
    extract_features,
    score_features,
)
from product_align_inspector.roi import crop_roi, validate_roi

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


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


def _affine_h(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (2, 3):
        raise ValueError(f"Expected 2x3 affine matrix, got {m.shape}")
    out = np.eye(3, dtype=np.float64)
    out[:2, :] = m
    return out


def aligned_to_input_affine(alignment) -> np.ndarray:
    """Map final aligned/reference coordinates back to the raw input image."""
    if alignment.feature_matrix is None:
        raise RuntimeError(
            f"Cannot map ROI to original image for alignment method {alignment.method}: feature_matrix unavailable"
        )
    input_to_feature = _affine_h(alignment.feature_matrix)
    input_to_final = input_to_feature
    if alignment.ecc_matrix is not None:
        # findTransformECC returns final(template)->feature(moving). The actual
        # refined image uses WARP_INVERSE_MAP, therefore feature->final = inv(W).
        final_to_feature = _affine_h(alignment.ecc_matrix)
        feature_to_final = np.linalg.inv(final_to_feature)
        input_to_final = feature_to_final @ input_to_feature
    return np.linalg.inv(input_to_final)[:2, :].astype(np.float32)


def roi_polygon_in_input(roi: list[int] | tuple[int, int, int, int], final_to_input: np.ndarray) -> np.ndarray:
    x, y, w, h = [int(v) for v in roi]
    corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
    return np.round(cv2.transform(corners, final_to_input).reshape(-1, 2)).astype(np.int32)


def roi_polygon_aligned(roi: list[int] | tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = [int(v) for v in roi]
    return np.asarray([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)


def draw_polygon(canvas: np.ndarray, polygon: np.ndarray, label: str, status: str) -> None:
    color = (0, 190, 0) if status == "PASS" else (0, 0, 255)
    thickness = max(3, int(round(min(canvas.shape[:2]) / 700)))
    poly = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [poly], True, color, thickness, cv2.LINE_AA)

    pts = poly.reshape(-1, 2)
    anchor = pts[np.argmin(pts[:, 1])]
    font_scale = max(0.52, min(0.9, min(canvas.shape[:2]) / 2000.0))
    text_thickness = max(1, thickness - 1)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
    tx = int(np.clip(anchor[0], 0, max(0, canvas.shape[1] - tw - 10)))
    ty = int(anchor[1]) - 8
    if ty - th - baseline - 4 < 0:
        ty = min(canvas.shape[0] - 5, int(anchor[1]) + th + baseline + 12)
    ty = int(np.clip(ty, th + baseline + 4, canvas.shape[0] - 5))
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


def draw_banner(canvas: np.ndarray, relative: str, scenario: str, final_status: str, ng_rois: list[str], prefix: str) -> None:
    if final_status == "PASS":
        color = (0, 150, 0)
    elif final_status == "NG":
        color = (0, 0, 220)
    else:
        color = (0, 180, 255)
    banner_h = max(82, int(round(canvas.shape[0] * 0.082)))
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], banner_h), color, -1)
    text = f"{prefix} | {scenario} | {relative} | FINAL: {final_status}"
    if ng_rois:
        text += " | NG: " + ",".join(ng_rois)
    cv2.putText(
        canvas,
        text,
        (20, int(banner_h * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.65, banner_h / 90.0),
        (255, 255, 255),
        max(2, int(round(banner_h / 40))),
        cv2.LINE_AA,
    )


def load_group(model_root: Path, group: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    group_dir = model_root / group
    model_path = group_dir / "model.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model manifest not found: {model_path}")
    manifest = json.loads(model_path.read_text(encoding="utf-8"))
    if manifest.get("model_type") != "resnet18_goodonly_knn":
        raise ValueError(f"Unsupported model_type in {model_path}: {manifest.get('model_type')}")
    entries: list[dict[str, object]] = []
    for item in manifest.get("rois", []):
        entry = dict(item)
        bank_path = group_dir / str(entry["bank_file"])
        bank = np.load(bank_path).astype(np.float32, copy=False)
        entry["bank"] = bank
        entry["group"] = group
        entry["normal_semantics"] = str(manifest.get("normal_semantics", "normal"))
        entries.append(entry)
    return manifest, entries


def jitter_offsets(radius: int) -> list[tuple[int, int]]:
    r = max(0, int(radius))
    if r == 0:
        return [(0, 0)]
    return [
        (0, 0),
        (-r, 0),
        (r, 0),
        (0, -r),
        (0, r),
        (-r, -r),
        (r, -r),
        (-r, r),
        (r, r),
    ]


def shifted_roi(roi: list[int], dx: int, dy: int, image_w: int, image_h: int) -> list[int] | None:
    x, y, w, h = roi
    candidate = [x + dx, y + dy, w, h]
    return candidate if validate_roi(candidate, image_w, image_h) else None


def write_gallery(output: Path, rows: list[dict[str, object]], scenarios: list[str]) -> Path:
    cards = []
    for row in rows:
        original_overlay = str(row.get("overlay_original", ""))
        aligned_overlay = str(row.get("overlay_aligned", ""))
        if not original_overlay and not aligned_overlay:
            continue
        original_src = html.escape(original_overlay.replace("\\", "/"))
        aligned_src = html.escape(aligned_overlay.replace("\\", "/"))
        relative = html.escape(str(row.get("relative_path", "")))
        scenario = html.escape(str(row.get("scenario", "")))
        status = html.escape(str(row.get("final_status", "")))
        ng = html.escape(str(row.get("ng_rois", "")))
        images_html = ""
        if original_src:
            images_html += f'<div><b>ORIGINAL mapped ROIs</b><a href="{original_src}" target="_blank"><img loading="lazy" src="{original_src}"></a></div>'
        if aligned_src:
            images_html += f'<div><b>ALIGNED canonical ROIs</b><a href="{aligned_src}" target="_blank"><img loading="lazy" src="{aligned_src}"></a></div>'
        cards.append(
            f'<article class="card"><div class="pair">{images_html}</div>'
            f'<div class="meta"><b>{relative}</b><br>Scenario: {scenario}<br>Status: {status}<br>NG ROIs: {ng or "-"}</div></article>'
        )
    scenario_text = ", ".join(html.escape(v) for v in scenarios) if scenarios else "all"
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>ResNet18 GOOD-only Review</title>
<style>body{{font-family:Segoe UI,Arial;margin:18px;background:#f4f4f4}}.grid{{display:grid;grid-template-columns:1fr;gap:16px}}.card{{background:#fff;border:1px solid #ddd;border-radius:8px;overflow:hidden}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:8px}}.pair img{{width:100%;display:block;margin-top:4px}}.meta{{padding:10px;line-height:1.45}}@media(max-width:1100px){{.pair{{grid-template-columns:1fr}}}}</style></head>
<body><h1>ResNet18 GOOD-only S/E Review</h1><p>Scenario filter: {scenario_text}. Left is mapped to the ORIGINAL image; right is the exact ALIGNED image used for inference. If the right-hand boxes are wrong, the ROI config/alignment is wrong. If only the left-hand boxes are wrong, the inverse drawing transform is wrong.</p><div class='grid'>{''.join(cards)}</div></body></html>"""
    path = output / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate the separate S/E GOOD-only ResNet18 normal models and draw every ROI on original/aligned images."
    )
    p.add_argument("--test-root", required=True)
    p.add_argument("--model-root", default="artifacts/resnet18_goodonly")
    p.add_argument("--reference", required=True)
    p.add_argument("--weights", default=str(DEFAULT_RESNET18_WEIGHTS))
    p.add_argument("--output", default="artifacts/resnet18_goodonly_review")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--foreground-threshold", type=int, default=238)
    p.add_argument("--threshold-scale", type=float, default=1.0)
    p.add_argument(
        "--local-jitter",
        type=int,
        default=4,
        help="Test +/- this many aligned pixels around each fixed ROI and use the best normal match. 0 disables.",
    )
    p.add_argument("--scenario", action="append", help="Optional scenario filter; may be repeated")
    args = p.parse_args()

    if args.threshold_scale <= 0:
        raise SystemExit("--threshold-scale must be > 0")
    if args.local_jitter < 0:
        raise SystemExit("--local-jitter must be >= 0")

    test_root = Path(args.test_root).resolve()
    model_root = Path(args.model_root).resolve()
    reference_path = Path(args.reference).resolve()
    weights_path = Path(args.weights).resolve()
    output = Path(args.output).resolve()
    overlays_original_root = output / "overlays_original"
    overlays_aligned_root = output / "overlays_aligned"
    output.mkdir(parents=True, exist_ok=True)
    overlays_original_root.mkdir(parents=True, exist_ok=True)
    overlays_aligned_root.mkdir(parents=True, exist_ok=True)

    reference = read_image(reference_path)
    s_manifest, s_entries = load_group(model_root, "S")
    e_manifest, e_entries = load_group(model_root, "E")
    entries = s_entries + e_entries
    if not entries:
        raise SystemExit("No S/E ROI model entries found")

    input_sizes = {int(s_manifest.get("input_size", 224)), int(e_manifest.get("input_size", 224))}
    if len(input_sizes) != 1:
        raise SystemExit(f"S/E input sizes differ: {sorted(input_sizes)}")
    input_size = next(iter(input_sizes))
    k_values = {
        int(s_manifest.get("scoring", {}).get("k_neighbors", 3)),
        int(e_manifest.get("scoring", {}).get("k_neighbors", 3)),
    }
    if len(k_values) != 1:
        raise SystemExit(f"S/E k-neighbors differ: {sorted(k_values)}")
    k_neighbors = next(iter(k_values))

    model, device = build_feature_extractor(weights_path, args.device)
    align_cfg = ProductLocatorConfig(foreground_threshold=args.foreground_threshold)

    all_images = collect_images(test_root)
    filters = {str(v).strip().lower() for v in (args.scenario or []) if str(v).strip()}
    selected: list[tuple[Path, Path, str]] = []
    for image_path in all_images:
        relative = image_path.relative_to(test_root)
        scenario = scenario_of(relative)
        if filters and scenario.lower() not in filters:
            continue
        selected.append((image_path, relative, scenario))
    if not selected:
        raise SystemExit("No test images selected")

    offsets = jitter_offsets(args.local_jitter)
    print("=== ResNet18 GOOD-only S/E Evaluation ===")
    print(f"Device: {device}")
    print(f"Images: {len(selected)}")
    print(f"S ROIs: {[e['id'] for e in s_entries]}")
    print(f"E ROIs: {[e['id'] for e in e_entries]}")
    print(f"Local ROI jitter: +/-{args.local_jitter}px ({len(offsets)} candidates/ROI)")
    print("All S and E ROIs are evaluated and drawn.")

    image_rows: list[dict[str, object]] = []
    roi_rows: list[dict[str, object]] = []

    for index, (image_path, relative, scenario) in enumerate(selected, 1):
        try:
            raw = read_image(image_path)
            alignment = align_to_reference(raw, reference, align_cfg)
            final_to_input = aligned_to_input_affine(alignment)
            aligned = alignment.aligned
            h, w = aligned.shape[:2]

            all_crops: list[np.ndarray] = []
            candidate_ranges: list[tuple[int, int, list[tuple[int, int]]]] = []
            for entry in entries:
                roi = [int(v) for v in entry["roi"]]
                if not validate_roi(roi, w, h):
                    raise RuntimeError(f"invalid ROI {entry['id']}: {roi}")
                start = len(all_crops)
                valid_offsets: list[tuple[int, int]] = []
                for dx, dy in offsets:
                    candidate_roi = shifted_roi(roi, dx, dy, w, h)
                    if candidate_roi is None:
                        continue
                    all_crops.append(crop_roi(aligned, candidate_roi))
                    valid_offsets.append((dx, dy))
                end = len(all_crops)
                if end == start:
                    raise RuntimeError(f"no valid jitter candidates for ROI {entry['id']}")
                candidate_ranges.append((start, end, valid_offsets))

            all_features = extract_features(
                model,
                all_crops,
                device=device,
                input_size=input_size,
                batch_size=args.batch_size,
            )

            canvas_original = raw.copy()
            canvas_aligned = aligned.copy()
            ng_rois: list[str] = []
            for entry, (start, end, valid_offsets) in zip(entries, candidate_ranges):
                bank = entry["bank"]
                candidate_scores = score_features(all_features[start:end], bank, k_neighbors=k_neighbors)
                best_index = int(np.argmin(candidate_scores))
                score = float(candidate_scores[best_index])
                best_dx, best_dy = valid_offsets[best_index]
                base_threshold = float(entry["threshold"])
                threshold = base_threshold * float(args.threshold_scale)
                status = "PASS" if score <= threshold else "NG"
                roi_id = str(entry["id"])
                group = str(entry["group"])
                if status == "NG":
                    ng_rois.append(roi_id)

                base_roi = [int(v) for v in entry["roi"]]
                label = f"{roi_id} {status} {score:.3f}/{threshold:.3f} d=({best_dx},{best_dy})"
                draw_polygon(canvas_original, roi_polygon_in_input(base_roi, final_to_input), label, status)
                draw_polygon(canvas_aligned, roi_polygon_aligned(base_roi), label, status)

                roi_rows.append(
                    {
                        "relative_path": relative.as_posix(),
                        "scenario": scenario,
                        "roi_id": roi_id,
                        "group": group,
                        "normal_semantics": entry["normal_semantics"],
                        "score": score,
                        "threshold": threshold,
                        "status": status,
                        "best_offset_x": best_dx,
                        "best_offset_y": best_dy,
                    }
                )

            final_status = "NG" if ng_rois else "PASS"
            draw_banner(canvas_original, relative.as_posix(), scenario, final_status, ng_rois, "ORIGINAL mapped")
            draw_banner(canvas_aligned, relative.as_posix(), scenario, final_status, ng_rois, "ALIGNED inference")

            original_rel = Path("overlays_original") / relative.with_suffix(".jpg")
            aligned_rel = Path("overlays_aligned") / relative.with_suffix(".jpg")
            original_path = output / original_rel
            aligned_path = output / aligned_rel
            original_path.parent.mkdir(parents=True, exist_ok=True)
            aligned_path.parent.mkdir(parents=True, exist_ok=True)
            write_image(original_path, canvas_original)
            write_image(aligned_path, canvas_aligned)

            image_rows.append(
                {
                    "relative_path": relative.as_posix(),
                    "scenario": scenario,
                    "final_status": final_status,
                    "ng_rois": ",".join(ng_rois),
                    "alignment_method": alignment.method,
                    "feature_inlier_ratio": alignment.feature_inlier_ratio,
                    "ecc_score": "" if alignment.ecc_score is None else alignment.ecc_score,
                    "overlay_original": original_rel.as_posix(),
                    "overlay_aligned": aligned_rel.as_posix(),
                    "error": "",
                }
            )
            ecc_text = "-" if alignment.ecc_score is None else f"{alignment.ecc_score:.4f}"
            print(
                f"[{index}/{len(selected)}] {relative.as_posix()} -> {final_status} | "
                f"NG={ng_rois or '-'} | align={alignment.method} "
                f"inlier={alignment.feature_inlier_ratio:.1%} ecc={ecc_text}"
            )
        except Exception as exc:
            image_rows.append(
                {
                    "relative_path": relative.as_posix(),
                    "scenario": scenario,
                    "final_status": "RETRY",
                    "ng_rois": "",
                    "alignment_method": "",
                    "feature_inlier_ratio": "",
                    "ecc_score": "",
                    "overlay_original": "",
                    "overlay_aligned": "",
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(selected)}] {relative.as_posix()} -> RETRY: {exc}")

    image_csv = output / "image_summary.csv"
    with image_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "relative_path",
                "scenario",
                "final_status",
                "ng_rois",
                "alignment_method",
                "feature_inlier_ratio",
                "ecc_score",
                "overlay_original",
                "overlay_aligned",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(image_rows)

    roi_csv = output / "roi_scores.csv"
    with roi_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "relative_path",
                "scenario",
                "roi_id",
                "group",
                "normal_semantics",
                "score",
                "threshold",
                "status",
                "best_offset_x",
                "best_offset_y",
            ],
        )
        writer.writeheader()
        writer.writerows(roi_rows)

    counts = {
        "PASS": sum(1 for r in image_rows if r["final_status"] == "PASS"),
        "NG": sum(1 for r in image_rows if r["final_status"] == "NG"),
        "RETRY": sum(1 for r in image_rows if r["final_status"] == "RETRY"),
    }
    summary = {
        "images": len(image_rows),
        "counts": counts,
        "threshold_scale": float(args.threshold_scale),
        "local_jitter": int(args.local_jitter),
        "note": "Folder scenario is not exact per-ROI ground truth; compare aligned and original overlays to diagnose ROI geometry separately from model scoring.",
        "image_summary": str(image_csv),
        "roi_scores": str(roi_csv),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    gallery = write_gallery(output, image_rows, sorted({str(r["scenario"]) for r in image_rows}))

    print("")
    print("=== Evaluation complete ===")
    print(f"PASS/NG/RETRY: {counts['PASS']}/{counts['NG']}/{counts['RETRY']}")
    print(f"Gallery: {gallery}")
    print(f"Aligned overlays: {overlays_aligned_root}")
    print(f"Original overlays: {overlays_original_root}")
    print(f"ROI scores: {roi_csv}")


if __name__ == "__main__":
    main()
