from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.io_utils import read_image
from product_align_inspector.roi import crop_roi, enabled_slots, roi_for_image


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fit_tile(image: np.ndarray, width: int = 260, height: int = 230) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    scale = min(width / max(1, w), height / max(1, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 238, dtype=np.uint8)
    x = (width - nw) // 2
    y = (height - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def _make_preview(
    canonical: np.ndarray,
    unknown_slots: list[str],
    slot_by_id: dict[str, dict],
    config: dict,
    title: str,
) -> np.ndarray:
    tile_w, tile_h = 260, 270
    cols = 3 if len(unknown_slots) > 2 else max(1, len(unknown_slots))
    rows = int(np.ceil(len(unknown_slots) / cols))
    header_h = 70
    canvas = np.full((header_h + rows * tile_h, cols * tile_w, 3), 250, dtype=np.uint8)
    cv2.putText(canvas, title[:110], (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Fill defect_slots in review CSV: missing->EMPTY S slots; excess->SCREW E slots",
        (14, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )

    h, w = canonical.shape[:2]
    for i, slot_id in enumerate(unknown_slots):
        slot = slot_by_id[slot_id]
        roi = roi_for_image(slot["roi"], config, w, h)
        crop = crop_roi(canonical, roi)
        tile = _fit_tile(crop, tile_w, tile_h - 40)
        y0 = header_h + (i // cols) * tile_h
        x0 = (i % cols) * tile_w
        canvas[y0:y0 + tile.shape[0], x0:x0 + tile.shape[1]] = tile
        cv2.rectangle(canvas, (x0, y0), (x0 + tile_w - 1, y0 + tile_h - 1), (170, 170, 170), 1)
        cv2.putText(
            canvas,
            slot_id,
            (x0 + 12, y0 + tile_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (10, 10, 10),
            2,
            cv2.LINE_AA,
        )
    return canvas


def export_review(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest).resolve()
    reference = read_image(Path(args.reference).resolve())
    config = json.loads(Path(args.config).resolve().read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    review_csv = Path(args.review_csv).resolve()

    fields, rows = _read_csv(manifest)
    slots = enabled_slots(config)
    slot_by_id = {str(slot["id"]): slot for slot in slots}
    slot_ids = list(slot_by_id)
    missing_columns = [slot_id for slot_id in slot_ids if slot_id not in fields]
    if missing_columns:
        raise RuntimeError(f"Manifest is missing slot columns: {missing_columns}")

    align_cfg = ProductLocatorConfig(
        ecc_accept_score=float(args.ecc_accept),
        feature_min_inlier_ratio=float(args.min_inlier_ratio),
        canonical_scale=1.0,
    )

    review_rows: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    review_index = 0
    for row_index, row in enumerate(rows, 2):
        unknown_slots = [slot_id for slot_id in slot_ids if str(row.get(slot_id, "")).strip() == "?"]
        if not unknown_slots:
            continue

        image_path = Path(str(row.get("image", "")).strip()).resolve()
        source = str(row.get("source", "")).strip()
        try:
            raw = read_image(image_path)
            alignment = align_to_reference(raw, reference, align_cfg)
            canonical = alignment.aligned
        except Exception as exc:
            skipped.append({"image": str(image_path), "error": str(exc)})
            continue

        review_index += 1
        preview_name = f"{review_index:03d}_{source}_{image_path.stem}.jpg"
        preview_path = output_dir / preview_name
        preview = _make_preview(
            canonical,
            unknown_slots,
            slot_by_id,
            config,
            f"{image_path.name} | source={source}",
        )
        if not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError(f"Failed to save preview: {preview_path}")

        review_rows.append(
            {
                "image": str(image_path),
                "source": source,
                "manifest_row": str(row_index),
                "preview": str(preview_path),
                "unknown_slots": ",".join(unknown_slots),
                "defect_slots": "",
                "notes": "",
            }
        )

    _write_csv(
        review_csv,
        ["image", "source", "manifest_row", "preview", "unknown_slots", "defect_slots", "notes"],
        review_rows,
    )
    if skipped:
        (output_dir / "skipped.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Semantic S/E label review exported ===")
    print(f"Manifest:          {manifest}")
    print(f"Review images:     {len(review_rows)}")
    print(f"Review CSV:        {review_csv}")
    print(f"Preview directory: {output_dir}")
    print(f"Skipped:           {len(skipped)}")
    print()
    print("Fill only the defect_slots column:")
    print("  missing_screws -> enter the S slot(s) that are EMPTY, e.g. S01 or S01,S02")
    print("  excess_screws  -> enter the E slot(s) that contain a SCREW, e.g. E05 or E03,E08")
    print("  leave blank only when none of the unknown slots is defective")
    print("Do not edit unknown_slots.")


def _parse_slots(text: str) -> set[str]:
    text = str(text or "").strip()
    if not text:
        return set()
    return {item.strip() for item in text.replace(";", ",").split(",") if item.strip()}


def apply_review(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest).resolve()
    review_csv = Path(args.review_csv).resolve()
    output = Path(args.output).resolve()

    fields, rows = _read_csv(manifest)
    _, reviews = _read_csv(review_csv)
    by_image = {str(row.get("image", "")).strip(): row for row in reviews}

    changed = 0
    unresolved: list[dict[str, str]] = []
    for row in rows:
        image = str(Path(str(row.get("image", "")).strip()).resolve())
        unknown = [field for field in fields if str(row.get(field, "")).strip() == "?"]
        if not unknown:
            continue
        review = by_image.get(image)
        if review is None:
            unresolved.append({"image": image, "reason": "missing review row"})
            continue

        source = str(row.get("source", "")).strip()
        expected_unknown = _parse_slots(review.get("unknown_slots", ""))
        if set(unknown) != expected_unknown:
            unresolved.append({"image": image, "reason": "unknown_slots no longer matches manifest"})
            continue

        defect_slots = _parse_slots(review.get("defect_slots", ""))
        invalid = sorted(defect_slots - set(unknown))
        if invalid:
            unresolved.append({"image": image, "reason": f"invalid defect_slots: {invalid}"})
            continue

        if source == "missing_screws":
            for slot_id in unknown:
                row[slot_id] = "empty" if slot_id in defect_slots else "screw"
        elif source == "excess_screws":
            for slot_id in unknown:
                row[slot_id] = "screw" if slot_id in defect_slots else "empty"
        else:
            unresolved.append({"image": image, "reason": f"unsupported source for compact review: {source}"})
            continue
        changed += 1

    _write_csv(output, fields, rows)
    remaining = sum(1 for row in rows for field in fields if str(row.get(field, "")).strip() == "?")

    print("=== Semantic S/E labels applied ===")
    print(f"Input manifest:    {manifest}")
    print(f"Review CSV:        {review_csv}")
    print(f"Output manifest:   {output}")
    print(f"Rows updated:      {changed}")
    print(f"Remaining ?:       {remaining}")
    print(f"Unresolved rows:   {len(unresolved)}")
    if unresolved:
        unresolved_path = output.with_suffix(".unresolved.json")
        unresolved_path.write_text(json.dumps(unresolved, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Unresolved log:    {unresolved_path}")
    if remaining:
        raise SystemExit("Manifest still contains unknown labels. Fix the review CSV before final training.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export compact S/E label-review previews and apply reviewed labels back to the semantic manifest."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="Create ROI preview sheets and a compact review CSV for '?' labels.")
    export.add_argument("--manifest", required=True)
    export.add_argument("--reference", required=True)
    export.add_argument("--config", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--review-csv", required=True)
    export.add_argument("--ecc-accept", type=float, default=0.70)
    export.add_argument("--min-inlier-ratio", type=float, default=0.10)
    export.set_defaults(func=export_review)

    apply_cmd = sub.add_parser("apply", help="Apply compact defect_slots decisions back to the wide manifest.")
    apply_cmd.add_argument("--manifest", required=True)
    apply_cmd.add_argument("--review-csv", required=True)
    apply_cmd.add_argument("--output", required=True)
    apply_cmd.set_defaults(func=apply_review)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
