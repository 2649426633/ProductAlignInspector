from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from product_align_inspector.alignment import ProductLocatorConfig
from product_align_inspector.inspection_pipeline import InspectionPipeline
from product_align_inspector.io_utils import read_image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_images(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_config(config: dict, reference_width: int, reference_height: int) -> str:
    """Validate ROI bounds; return a short coordinate-system note."""
    rw = config.get("reference_width")
    rh = config.get("reference_height")
    if rw is not None and rh is not None:
        if (int(rw), int(rh)) != (reference_width, reference_height):
            raise SystemExit(
                f"CONFIG/REFERENCE SIZE MISMATCH: config={rw}x{rh}, "
                f"reference={reference_width}x{reference_height}"
            )
        note = f"declared {int(rw)}x{int(rh)}"
    else:
        note = "size not declared in JSON; ROI bounds checked only"

    for slot in config.get("screw_slots", []):
        if not bool(slot.get("enabled", True)):
            continue
        roi = slot.get("roi")
        if roi is None:
            continue
        if not isinstance(roi, (list, tuple)) or len(roi) != 4:
            raise SystemExit(f"Invalid ROI for {slot.get('id', '?')}: {roi}")
        x, y, w, h = map(int, roi)
        if w <= 0 or h <= 0:
            raise SystemExit(f"Invalid ROI size for {slot.get('id', '?')}: {roi}")
        if x < 0 or y < 0 or x + w > reference_width or y + h > reference_height:
            raise SystemExit(
                f"ROI OUT OF CANONICAL BOUNDS for {slot.get('id', '?')}: {roi}; "
                f"reference={reference_width}x{reference_height}"
            )
    return note


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RAW -> canonical preview reference -> ROI coordinate chain -> inverse mapping to RAW. "
            "Per-frame geometry is rigid only: rotation + X/Y translation, scale=1."
        )
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument(
        "--reference",
        required=True,
        help="Canonical detection reference, e.g. artifacts/reference/brunei_preview_reference.png",
    )
    parser.add_argument(
        "--config",
        help="Optional canonical ROI JSON, e.g. configs/brunei_preview_template.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--ecc-accept", type=float, default=0.70)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.10)
    parser.add_argument("--no-save-canonical", action="store_true")
    parser.add_argument("--no-save-overlay", action="store_true")
    parser.add_argument("--no-save-restored", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    reference_path = Path(args.reference).resolve()
    reference = read_image(reference_path)
    ref_h, ref_w = reference.shape[:2]

    config = None
    config_note = "OFF"
    if args.config:
        config_path = Path(args.config).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config_note = validate_config(config, ref_w, ref_h)

    align_cfg = ProductLocatorConfig(
        ecc_accept_score=float(args.ecc_accept),
        feature_min_inlier_ratio=float(args.min_inlier_ratio),
        canonical_scale=1.0,
    )

    pipeline = InspectionPipeline(
        reference=reference,
        output_dir=output,
        align_cfg=align_cfg,
        detectors=[],
        config=config,
        save_aligned=not args.no_save_canonical,
        save_overlay=not args.no_save_overlay,
        save_restored=not args.no_save_restored,
    )

    files = collect_images(input_root)
    if not files:
        raise SystemExit(f"No images found under: {input_root}")

    jsonl_path = output / "inspection_log.jsonl"
    csv_path = output / "inspection_summary.csv"
    summary_path = output / "summary.json"

    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()

    print("=== ProductAlignInspector Main Framework ===")
    print(f"Input root:       {input_root}")
    print(f"Canonical ref:    {reference_path}")
    print(f"Canonical size:   {ref_w}x{ref_h}")
    print(f"ROI config:       {Path(args.config).resolve() if args.config else 'OFF'}")
    if args.config:
        print(f"ROI config check: {config_note}")
    print(f"Images:           {len(files)}")
    print("Geometry:         rotation + X/Y translation ONLY")
    print("Scale:            FIXED 1.00000")
    print("Flow: RAW -> canonical -> ROI/detector hook -> inverse-map to RAW")
    print("Alignment failure: SKIP_ALIGNMENT + log; detectors are not executed")
    print()

    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for index, path in enumerate(files, 1):
            rel = path.relative_to(input_root)
            record = pipeline.process(path, relative_path=rel)
            data = record.to_dict()
            jsonl.write(json.dumps(data, ensure_ascii=False) + "\n")
            jsonl.flush()

            counts[record.final_status] += 1
            rows.append(
                {
                    "timestamp": record.timestamp,
                    "path": record.relative_path,
                    "final_status": record.final_status,
                    "detection_run": record.detection_run,
                    "alignment_status": record.alignment_status,
                    "alignment_method": record.alignment_method,
                    "matches": record.feature_matches,
                    "inliers": record.feature_inliers,
                    "inlier_ratio": record.feature_inlier_ratio,
                    "canonical_scale": record.canonical_scale,
                    "rotation_deg": record.rotation_deg,
                    "tx": record.tx,
                    "ty": record.ty,
                    "ecc": record.alignment_ecc,
                    "alignment_time_sec": record.alignment_time_sec,
                    "total_time_sec": record.total_time_sec,
                    "canonical_path": record.canonical_path,
                    "overlay_path": record.overlay_path,
                    "restored_path": record.restored_path,
                    "error": record.error,
                }
            )

            if record.final_status == "SKIP_ALIGNMENT":
                print(
                    f"[{index}/{len(files)}] {rel.as_posix()} -> SKIP_ALIGNMENT | "
                    f"{record.error}"
                )
            elif record.alignment_status == "OK":
                ecc = (
                    "-"
                    if record.alignment_ecc is None
                    else f"{record.alignment_ecc:.4f}"
                )
                rot = (
                    "-"
                    if record.rotation_deg is None
                    else f"{record.rotation_deg:.3f}"
                )
                print(
                    f"[{index}/{len(files)}] {rel.as_posix()} -> {record.final_status} | "
                    f"scale=1.00000 rot={rot} ecc={ecc} | {record.total_time_sec:.3f}s"
                )
            else:
                print(
                    f"[{index}/{len(files)}] {rel.as_posix()} -> "
                    f"{record.final_status} | {record.error}"
                )

    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row["alignment_status"] == "OK"]
    summary = {
        "images": len(files),
        "counts": dict(counts),
        "alignment_ok": len(ok_rows),
        "alignment_skipped": counts.get("SKIP_ALIGNMENT", 0),
        "canonical_reference": str(reference_path),
        "canonical_size": [ref_w, ref_h],
        "roi_config": str(Path(args.config).resolve()) if args.config else None,
        "geometry": {
            "rotation": True,
            "translation_x": True,
            "translation_y": True,
            "scale": 1.0,
            "shear": False,
            "perspective": False,
        },
        "detectors_plugged_in": False,
        "next_detector_slots": [
            "S_presence",
            "E_empty",
            "spring",
            "surface_anomaly",
        ],
        "logs": {"jsonl": str(jsonl_path), "csv": str(csv_path)},
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== Summary ===")
    print(f"Images:            {len(files)}")
    print(f"Alignment OK:      {summary['alignment_ok']}")
    print(f"Alignment skipped: {summary['alignment_skipped']}")
    print("Scale:             1.00000 (fixed)")
    print(f"Canonical:         {output / 'canonical'}")
    print(f"Overlays:          {output / 'overlays'}")
    print(f"Restored:          {output / 'restored'}")
    print(f"CSV log:           {csv_path}")
    print(f"Summary:           {summary_path}")


if __name__ == "__main__":
    main()
