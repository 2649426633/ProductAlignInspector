from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.io_utils import read_image, write_image

COLOR_S = np.array([60, 210, 60], dtype=np.uint8)
COLOR_E = np.array([0, 170, 255], dtype=np.uint8)


def _detect_rectangles(image: np.ndarray, color: np.ndarray, expected_count: int) -> list[list[int]]:
    mask = cv2.inRange(image, color, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ih, iw = image.shape[:2]
    candidates: list[list[int]] = []

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < 40 or ch < 40 or cw > iw * 0.25 or ch > ih * 0.30:
            continue

        pad = 3
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(iw, x + cw + pad)
        y1 = min(ih, y + ch + pad)
        sub = (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
        row_counts = sub.sum(axis=1)
        col_counts = sub.sum(axis=0)

        rows = np.where(row_counts >= max(25, int(cw * 0.55)))[0]
        cols = np.where(col_counts >= max(25, int(ch * 0.55)))[0]
        if len(rows) < 2 or len(cols) < 2:
            continue

        top = int(rows.min() + y0)
        bottom = int(rows.max() + y0)
        left = int(cols.min() + x0)
        right = int(cols.max() + x0)
        width = right - left
        height = bottom - top
        if width < 30 or height < 30:
            continue
        candidates.append([left, top, width, height])

    kept: list[list[int]] = []
    for rect in sorted(candidates, key=lambda r: r[2] * r[3], reverse=True):
        cx = rect[0] + rect[2] / 2.0
        cy = rect[1] + rect[3] / 2.0
        duplicate = False
        for other in kept:
            ocx = other[0] + other[2] / 2.0
            ocy = other[1] + other[3] / 2.0
            if abs(cx - ocx) < 8 and abs(cy - ocy) < 8:
                duplicate = True
                break
        if not duplicate:
            kept.append(rect)

    if len(kept) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} rectangles for color {color.tolist()}, recovered {len(kept)}."
        )
    return kept


def _assign_ids(s_rects: list[list[int]], e_rects: list[list[int]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    s_sorted = sorted(s_rects, key=lambda r: r[0] + r[2] / 2.0)
    screw_slots = [
        {"id": f"S{i:02d}", "roi": rect, "enabled": True, "expected": "screw"}
        for i, rect in enumerate(s_sorted, 1)
    ]

    e_by_y = sorted(e_rects, key=lambda r: r[1] + r[3] / 2.0)
    if len(e_by_y) != 9:
        raise RuntimeError("This Brunei preview workflow expects exactly 9 E rectangles.")
    rows = [e_by_y[0:3], e_by_y[3:6], e_by_y[6:9]]
    ordered_e: list[list[int]] = []
    for row in rows:
        ordered_e.extend(sorted(row, key=lambda r: r[0] + r[2] / 2.0))
    empty_slots = [
        {"id": f"E{i:02d}", "roi": rect, "enabled": True, "expected": "empty"}
        for i, rect in enumerate(ordered_e, 1)
    ]
    return screw_slots, empty_slots


def _annotation_mask(image: np.ndarray) -> np.ndarray:
    exact_s = cv2.inRange(image, COLOR_S, COLOR_S)
    exact_e = cv2.inRange(image, COLOR_E, COLOR_E)
    mask = cv2.bitwise_or(exact_s, exact_e)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.dilate(mask, kernel, iterations=1)


def _draw_preview(reference: np.ndarray, slots: list[dict[str, object]]) -> np.ndarray:
    canvas = reference.copy()
    for item in slots:
        x, y, w, h = [int(v) for v in item["roi"]]
        expected = str(item["expected"])
        color = (60, 210, 60) if expected == "screw" else (0, 170, 255)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            canvas,
            f"{item['id']}:{expected}",
            (x, max(20, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Use the known-correct brunei_preview.png itself as the canonical S/E ROI template. "
            "No flipping, SIFT remapping, or existing brunei.json coordinates are used."
        )
    )
    p.add_argument("--preview", required=True, help="Known-correct brunei_preview.png")
    p.add_argument("--reference-output", default="artifacts/reference/brunei_preview_reference.png")
    p.add_argument("--config-output", default="configs/brunei_preview_template.json")
    p.add_argument("--verify-output", default="configs/brunei_preview_template_verify.png")
    args = p.parse_args()

    preview_path = Path(args.preview).resolve()
    reference_output = Path(args.reference_output).resolve()
    config_output = Path(args.config_output).resolve()
    verify_output = Path(args.verify_output).resolve()

    preview = read_image(preview_path)
    h, w = preview.shape[:2]

    s_rects = _detect_rectangles(preview, COLOR_S, 2)
    e_rects = _detect_rectangles(preview, COLOR_E, 9)
    screw_slots, empty_slots = _assign_ids(s_rects, e_rects)
    slots = screw_slots + empty_slots

    mask = _annotation_mask(preview)
    cleaned = cv2.inpaint(preview, mask, 5, cv2.INPAINT_TELEA)

    config = {
        "schema_version": 2,
        "product": "brunei",
        "reference_image": str(reference_output),
        "reference_width": int(w),
        "reference_height": int(h),
        "coordinate_system": {
            "source": str(preview_path),
            "image_width": int(w),
            "image_height": int(h),
            "note": "S/E ROI coordinates recovered directly from the known-correct brunei_preview.png. No remapping or flipping was applied.",
        },
        "screw_slots": slots,
        "spring_regions": [],
    }

    reference_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.parent.mkdir(parents=True, exist_ok=True)
    verify_output.parent.mkdir(parents=True, exist_ok=True)
    write_image(reference_output, cleaned)
    write_image(verify_output, _draw_preview(cleaned, slots))
    config_output.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Brunei preview template prepared ===")
    print(f"Preview size: {w}x{h}")
    print("No flip/remap used.")
    for item in slots:
        print(f"{item['id']}: {item['roi']} expected={item['expected']}")
    print(f"Reference: {reference_output}")
    print(f"Config: {config_output}")
    print(f"Verify: {verify_output}")
    print("Spring regions intentionally omitted because this preview contains no spring ROI overlays.")


if __name__ == "__main__":
    main()
