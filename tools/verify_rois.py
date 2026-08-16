from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

# Allow direct execution: python tools\verify_rois.py ...
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Align one image, draw configured ROIs and export each ROI crop.")
    parser.add_argument("--input", required=True, help="Raw input image")
    parser.add_argument("--reference", required=True, help="Canonical reference_aligned.png")
    parser.add_argument("--config", required=True, help="Product ROI JSON created by annotate_rois.py")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--threshold", type=int, default=238)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    image = read_image(args.input)
    reference = read_image(args.reference)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    result = align_to_reference(image, reference, ProductLocatorConfig(foreground_threshold=args.threshold))
    aligned = result.aligned
    preview = aligned.copy()
    h, w = aligned.shape[:2]

    rows: list[dict[str, object]] = []

    for slot in config.get("screw_slots", []):
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
    write_json(
        out / "verification.json",
        {
            "input": str(Path(args.input)),
            "reference": str(Path(args.reference)),
            "config": str(config_path),
            "alignment": result.to_dict(),
            "rois": rows,
        },
    )

    print(f"Method: {result.method}")
    print(f"Feature matches: {result.feature_matches}, inliers: {result.feature_inliers} ({result.feature_inlier_ratio:.1%})")
    if result.ecc_score is not None:
        print(f"ECC score: {result.ecc_score:.6f}")
    print(f"ROI preview: {out / 'roi_preview.png'}")
    print(f"ROI crops: {out / 'crops'}")


if __name__ == "__main__":
    main()
