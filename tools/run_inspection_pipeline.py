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
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RAW -> canonical preview reference -> ROI coordinate chain -> inverse mapping to RAW. "
            "Alignment failures are skipped and logged."
        )
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument(
        "--reference",
        required=True,
        help="Canonical detection reference, e.g. artifacts/reference/brunei_preview_reference.png",
    )
    parser.add_argument("--config", help="Optional canonical ROI JSON, e.g. configs/brunei_preview_template.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ecc-accept", type=float, default=0.70)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.15)
    parser.add_argument("--canonical-scale", type=float, help="Optional locked RAW->canonical uniform scale")
    parser.add_argument("--scale-tolerance", type=float, default=0.04)
    parser.add_argument("--no-save-canonical", action="store_true")
    parser.add_argument("--no-save-overlay", action="store_true")
    parser.add_argument("--no-save-restored", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    reference_path = Path(args.reference).resolve()
    reference = read_image(reference_path)
    config = None
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        rw = config.get("reference_width")
        rh = config.get("reference_height")
        if rw is not None and rh is not None:
            if (int(rw), int(rh)) != (reference.shape[1], reference.shape[0]):
                raise SystemExit(
                    f"CONFIG/REFERENCE SIZE MISMATCH: config={rw}x{rh}, "
                    f"reference={reference.shape[1]}x{reference.shape[0]}"
                )

    align_cfg = ProductLocatorConfig(
        ecc_accept_score=float(args.ecc_accept),
        feature_min_inlier_ratio=float(args.min_inlier_ratio),
        canonical_scale=args.canonical_scale,
        canonical_scale_tolerance=float(args.scale_tolerance),
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
    print(f"Canonical size:   {reference.shape[1]}x{reference.shape[0]}")
    print(f"ROI config:       {Path(args.config).resolve() if args.config else 'OFF'}")
    print(f"Images:           {len(files)}")
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
                print(f"[{index}/{len(files)}] {rel.as_posix()} -> SKIP_ALIGNMENT | {record.error}")
            elif record.alignment_status == "OK":
                ecc = "-" if record.alignment_ecc is None else f"{record.alignment_ecc:.4f}"
                scale = "-" if record.canonical_scale is None else f"{record.canonical_scale:.5f}"
                rot = "-" if record.rotation_deg is None else f"{record.rotation_deg:.3f}"
                print(
                    f"[{index}/{len(files)}] {rel.as_posix()} -> {record.final_status} | "
                    f"scale={scale} rot={rot} ecc={ecc} | {record.total_time_sec:.3f}s"
                )
            else:
                print(f"[{index}/{len(files)}] {rel.as_posix()} -> {record.final_status} | {record.error}")

    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row["alignment_status"] == "OK"]
    scales = [float(row["canonical_scale"]) for row in ok_rows if row["canonical_scale"] not in (None, "")]
    summary = {
        "images": len(files),
        "counts": dict(counts),
        "alignment_ok": len(ok_rows),
        "alignment_skipped": counts.get("SKIP_ALIGNMENT", 0),
        "canonical_reference": str(reference_path),
        "canonical_size": [reference.shape[1], reference.shape[0]],
        "roi_config": str(Path(args.config).resolve()) if args.config else None,
        "detectors_plugged_in": False,
        "measured_scale": {
            "count": len(scales),
            "min": min(scales) if scales else None,
            "max": max(scales) if scales else None,
            "mean": sum(scales) / len(scales) if scales else None,
        },
        "next_detector_slots": ["S_presence", "E_empty", "spring", "surface_anomaly"],
        "logs": {"jsonl": str(jsonl_path), "csv": str(csv_path)},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=== Summary ===")
    print(f"Images:            {len(files)}")
    print(f"Alignment OK:      {summary['alignment_ok']}")
    print(f"Alignment skipped: {summary['alignment_skipped']}")
    if scales:
        print(
            "Scale range:       "
            f"{summary['measured_scale']['min']:.5f} .. {summary['measured_scale']['max']:.5f} "
            f"(mean {summary['measured_scale']['mean']:.5f})"
        )
    print(f"Canonical:         {output / 'canonical'}")
    print(f"Overlays:          {output / 'overlays'}")
    print(f"Restored:          {output / 'restored'}")
    print(f"CSV log:           {csv_path}")
    print(f"Summary:           {summary_path}")


if __name__ == "__main__":
    main()
