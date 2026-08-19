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
            "Main inspection framework runner. Current phase validates the production "
            "flow: image -> rigid alignment -> skip/log on alignment failure -> detector hook."
        )
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--foreground-threshold", type=int, default=238)
    parser.add_argument("--ecc-accept", type=float, default=0.70)
    parser.add_argument("--no-save-aligned", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    reference = read_image(Path(args.reference))
    align_cfg = ProductLocatorConfig(
        foreground_threshold=args.foreground_threshold,
        ecc_accept_score=args.ecc_accept,
    )

    # Detectors are intentionally empty in this framework phase. Later we plug in
    # independent S, E, spring, and surface detectors without changing the flow.
    pipeline = InspectionPipeline(
        reference=reference,
        output_dir=output,
        align_cfg=align_cfg,
        detectors=[],
        save_aligned=not args.no_save_aligned,
    )

    files = collect_images(input_root)
    if not files:
        raise SystemExit(f"No images found under: {input_root}")

    jsonl_path = output / "inspection_log.jsonl"
    csv_path = output / "inspection_summary.csv"
    summary_path = output / "summary.json"

    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()

    print("=== Inspection Pipeline Framework ===")
    print(f"Input root: {input_root}")
    print(f"Reference:  {Path(args.reference).resolve()}")
    print(f"Images:     {len(files)}")
    print("Flow: RAW -> rigid alignment -> [fail: SKIP+LOG] -> [ok: detector hook]")
    print("Detectors: not plugged in yet; aligned images become READY_FOR_DETECTION")
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
                    "rotation_deg": record.rotation_deg,
                    "tx": record.tx,
                    "ty": record.ty,
                    "ecc": record.alignment_ecc,
                    "alignment_time_sec": record.alignment_time_sec,
                    "total_time_sec": record.total_time_sec,
                    "aligned_path": record.aligned_path,
                    "error": record.error,
                }
            )

            if record.final_status == "SKIP_ALIGNMENT":
                print(
                    f"[{index}/{len(files)}] {rel.as_posix()} -> SKIP_ALIGNMENT | "
                    f"{record.error}"
                )
            else:
                ecc = "-" if record.alignment_ecc is None else f"{record.alignment_ecc:.4f}"
                print(
                    f"[{index}/{len(files)}] {rel.as_posix()} -> {record.final_status} | "
                    f"rot={record.rotation_deg:.3f} | ecc={ecc} | "
                    f"{record.total_time_sec:.3f}s"
                )

    fields = [
        "timestamp",
        "path",
        "final_status",
        "detection_run",
        "alignment_status",
        "alignment_method",
        "rotation_deg",
        "tx",
        "ty",
        "ecc",
        "alignment_time_sec",
        "total_time_sec",
        "aligned_path",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "images": len(files),
        "counts": dict(counts),
        "alignment_ok": sum(1 for row in rows if row["alignment_status"] == "OK"),
        "alignment_skipped": counts.get("SKIP_ALIGNMENT", 0),
        "detectors_plugged_in": False,
        "next_detector_slots": ["S_presence", "E_empty", "spring", "surface_anomaly"],
        "logs": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=== Summary ===")
    print(f"Images: {len(files)}")
    print(f"Alignment OK: {summary['alignment_ok']}")
    print(f"Alignment skipped: {summary['alignment_skipped']}")
    print(f"JSONL log: {jsonl_path}")
    print(f"CSV log:   {csv_path}")
    print(f"Summary:   {summary_path}")


if __name__ == "__main__":
    main()
