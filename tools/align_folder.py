from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow direct execution such as:
#   python tools\align_folder.py ...
# without requiring an editable package install first.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.io_utils import read_image, write_image, write_json

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-align a dataset folder to one product reference.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=int, default=238)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    reference = read_image(args.reference)
    cfg = ProductLocatorConfig(foreground_threshold=args.threshold)

    files = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise SystemExit(f"No images found in: {input_dir}")

    rows: list[dict[str, object]] = []
    ok_count = 0

    for index, path in enumerate(files, 1):
        rel = path.relative_to(input_dir)
        out_image = output_dir / rel.with_suffix(".png")
        out_json = output_dir / "meta" / rel.with_suffix(".json")
        try:
            image = read_image(path)
            result = align_to_reference(image, reference, cfg)
            write_image(out_image, result.aligned)
            write_json(out_json, result.to_dict())
            rows.append({
                "file": str(rel),
                "status": "ok",
                "method": result.method,
                "angle_deg": result.location.angle_deg,
                "fallback_rotation_deg": ""
                if result.fallback_rotation_deg is None
                else result.fallback_rotation_deg,
                "feature_inlier_ratio": result.feature_inlier_ratio,
                "ecc_score": "" if result.ecc_score is None else result.ecc_score,
                "error": "",
            })
            ok_count += 1
            status = (
                f"OK ({result.method}, ecc="
                f"{'n/a' if result.ecc_score is None else f'{result.ecc_score:.3f}'})"
            )
        except Exception as exc:  # batch processing should continue and report failures
            rows.append({
                "file": str(rel),
                "status": "failed",
                "method": "",
                "angle_deg": "",
                "fallback_rotation_deg": "",
                "feature_inlier_ratio": "",
                "ecc_score": "",
                "error": str(exc),
            })
            status = f"FAILED: {exc}"
        print(f"[{index}/{len(files)}] {rel} -> {status}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "alignment_report.csv"
    fieldnames = [
        "file",
        "status",
        "method",
        "angle_deg",
        "fallback_rotation_deg",
        "feature_inlier_ratio",
        "ecc_score",
        "error",
    ]
    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done: {ok_count}/{len(files)} succeeded")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
