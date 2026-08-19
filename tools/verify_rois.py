from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.io_utils import read_image, write_image, write_json
from product_align_inspector.roi import crop_roi, validate_roi


def _draw_label(image, roi, label: str, color: tuple[int, int, int]) -> None:
    x, y, w, h = map(int, roi)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, max(2, round(min(image.shape[:2]) / 600)))
    font_scale = max(0.5, min(image.shape[:2]) / 1400.0)
    thickness = max(1, round(min(image.shape[:2]) / 900))
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    ty = max(th + 6, y - 6)
    cv2.rectangle(image, (x, ty - th - 6), (x + tw + 8, ty + 3), color, -1)
    cv2.putText(image, label, (x + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _verify_config_matches_reference(reference, config: dict) -> None:
    h, w = reference.shape[:2]
    cfg_w = config.get("reference_width")
    cfg_h = config.get("reference_height")
    if cfg_w is not None and cfg_h is not None and (int(cfg_w), int(cfg_h)) != (w, h):
        raise SystemExit(
            f"CONFIG/REFERENCE SIZE MISMATCH: config={cfg_w}x{cfg_h}, reference={w}x{h}. "
            "Do not scale/remap ROIs automatically; use the ROI JSON annotated on this exact reference."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Align one image, draw fixed ROIs, and export crops. No detection is performed.")
    parser.add_argument("--input", required=True, help="Raw input image")
    parser.add_argument("--reference", required=True, help="Canonical clean reference_aligned.png")
    parser.add_argument("--config", required=True, help="ROI JSON annotated on the same canonical reference")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--threshold", type=int, default=238)
    parser.add_argument("--allow-foreground-fallback", action="store_true")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    image = read_image(args.input)
    reference = read_image(args.reference)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _verify_config_matches_reference(reference, config)

    result = align_to_reference(image, reference, ProductLocatorConfig(foreground_threshold=args.threshold))
    if result.feature_matrix is None and not args.allow_foreground_fallback:
        raise SystemExit(
            f"Alignment is fallback-only ({result.method}); treat this image as RETRY, not as a product NG."
        )

    aligned = result.aligned
    preview = aligned.copy()
    h, w = aligned.shape[:2]
    rows: list[dict[str, object]] = []

    for slot in config.get("screw_slots", []):
        if not bool(slot.get("enabled", True)):
            continue
        slot_id = str(slot.get("id", "S?"))
        expected = str(slot.get("expected", "unknown"))
        roi = slot.get("roi")
        if roi is None or not validate_roi(roi, w, h):
            rows.append({"id": slot_id, "type": "screw_slot", "expected": expected, "status": "invalid_roi", "roi": roi})
            continue
        crop = crop_roi(aligned, roi)
        write_image(out / "crops" / "screw_slots" / f"{slot_id}_{expected}.png", crop)
        color = (0, 180, 0) if expected == "screw" else (200, 120, 0)
        _draw_label(preview, roi, f"{slot_id}:{expected}", color)
        rows.append({"id": slot_id, "type": "screw_slot", "expected": expected, "status": "ok", "roi": list(map(int, roi))})

    for region in config.get("spring_regions", []):
        if not bool(region.get("enabled", True)):
            continue
        region_id = str(region.get("id", "SPRING?"))
        expected_count = int(region.get("expected_count", 0))
        roi = region.get("roi")
        if roi is None or not validate_roi(roi, w, h):
            rows.append({"id": region_id, "type": "spring_region", "expected_count": expected_count, "status": "invalid_roi", "roi": roi})
            continue
        crop = crop_roi(aligned, roi)
        write_image(out / "crops" / "spring_regions" / f"{region_id}_count{expected_count}.png", crop)
        _draw_label(preview, roi, f"{region_id}:count={expected_count}", (180, 0, 180))
        rows.append({"id": region_id, "type": "spring_region", "expected_count": expected_count, "status": "ok", "roi": list(map(int, roi))})

    write_image(out / "aligned.png", aligned)
    write_image(out / "roi_preview.png", preview)
    write_json(out / "verification.json", {"input": str(Path(args.input)), "reference": str(Path(args.reference)), "config": str(config_path), "alignment": result.to_dict(), "rois": rows})

    print(f"Method: {result.method}")
    print(f"Feature matches: {result.feature_matches}, inliers: {result.feature_inliers} ({result.feature_inlier_ratio:.1%})")
    if result.ecc_score is not None:
        print(f"ECC score: {result.ecc_score:.6f}")
    print(f"ROI preview: {out / 'roi_preview.png'}")
    print(f"ROI crops: {out / 'crops'}")


if __name__ == "__main__":
    main()
