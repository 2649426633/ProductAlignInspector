from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Allow direct execution: python tools\extract_roi_dataset.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.io_utils import read_image, write_image
from product_align_inspector.roi import crop_roi, validate_roi

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _collect_inputs(input_files: list[str], input_dir: str | None) -> list[Path]:
    files = [Path(p) for p in input_files]
    if input_dir:
        root = Path(input_dir)
        files.extend(sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in files:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Align GOOD images and extract configured ROI crops for model training.")
    parser.add_argument("--input", action="append", default=[], help="One GOOD image. May be repeated.")
    parser.add_argument("--input-dir", help="Directory containing GOOD images; searched recursively.")
    parser.add_argument("--reference", required=True, help="Canonical reference_aligned.png")
    parser.add_argument("--config", required=True, help="Product ROI JSON")
    parser.add_argument("--output", required=True, help="Dataset output directory")
    parser.add_argument("--threshold", type=int, default=238)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.25, help="Reject weak feature alignment below this ratio")
    args = parser.parse_args()

    inputs = _collect_inputs(args.input, args.input_dir)
    if not inputs:
        raise SystemExit("No input images. Use --input and/or --input-dir.")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    reference = read_image(args.reference)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg = ProductLocatorConfig(foreground_threshold=args.threshold)

    manifest: list[dict[str, object]] = []
    success = 0

    for index, image_path in enumerate(inputs, 1):
        try:
            image = read_image(image_path)
            result = align_to_reference(image, reference, cfg)

            if result.method.startswith("sift") and result.feature_inlier_ratio < args.min_inlier_ratio:
                raise RuntimeError(
                    f"Weak alignment: inlier_ratio={result.feature_inlier_ratio:.3f} < {args.min_inlier_ratio:.3f}"
                )

            aligned = result.aligned
            h, w = aligned.shape[:2]
            stem = image_path.stem

            for slot in config.get("screw_slots", []):
                slot_id = str(slot.get("id", "S?"))
                expected = str(slot.get("expected", "unknown"))
                roi = slot.get("roi")
                if roi is None or not validate_roi(roi, w, h):
                    raise RuntimeError(f"Invalid ROI {slot_id}: {roi} for aligned image {w}x{h}")
                crop = crop_roi(aligned, roi)
                crop_path = out / "screw" / expected / f"{stem}__{slot_id}.png"
                write_image(crop_path, crop)
                manifest.append(
                    {
                        "source": str(image_path),
                        "kind": "screw_slot",
                        "id": slot_id,
                        "label": expected,
                        "expected_count": "",
                        "crop": str(crop_path),
                        "alignment_method": result.method,
                        "feature_matches": result.feature_matches,
                        "feature_inliers": result.feature_inliers,
                        "feature_inlier_ratio": result.feature_inlier_ratio,
                        "ecc_score": "" if result.ecc_score is None else result.ecc_score,
                        "status": "ok",
                        "error": "",
                    }
                )

            for region in config.get("spring_regions", []):
                region_id = str(region.get("id", "SPRING?"))
                expected_count = int(region.get("expected_count", 0))
                roi = region.get("roi")
                if roi is None or not validate_roi(roi, w, h):
                    raise RuntimeError(f"Invalid ROI {region_id}: {roi} for aligned image {w}x{h}")
                crop = crop_roi(aligned, roi)
                crop_path = out / "spring" / f"count_{expected_count}" / f"{stem}__{region_id}.png"
                write_image(crop_path, crop)
                manifest.append(
                    {
                        "source": str(image_path),
                        "kind": "spring_region",
                        "id": region_id,
                        "label": f"count_{expected_count}",
                        "expected_count": expected_count,
                        "crop": str(crop_path),
                        "alignment_method": result.method,
                        "feature_matches": result.feature_matches,
                        "feature_inliers": result.feature_inliers,
                        "feature_inlier_ratio": result.feature_inlier_ratio,
                        "ecc_score": "" if result.ecc_score is None else result.ecc_score,
                        "status": "ok",
                        "error": "",
                    }
                )

            success += 1
            print(f"[{index}/{len(inputs)}] {image_path.name} -> OK ({result.method}, inliers={result.feature_inlier_ratio:.1%})")
        except Exception as exc:
            manifest.append(
                {
                    "source": str(image_path),
                    "kind": "",
                    "id": "",
                    "label": "",
                    "expected_count": "",
                    "crop": "",
                    "alignment_method": "",
                    "feature_matches": "",
                    "feature_inliers": "",
                    "feature_inlier_ratio": "",
                    "ecc_score": "",
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(inputs)}] {image_path.name} -> FAILED: {exc}")

    manifest_path = out / "manifest.csv"
    fieldnames = [
        "source",
        "kind",
        "id",
        "label",
        "expected_count",
        "crop",
        "alignment_method",
        "feature_matches",
        "feature_inliers",
        "feature_inlier_ratio",
        "ecc_score",
        "status",
        "error",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    print(f"Done: {success}/{len(inputs)} images succeeded")
    print(f"Dataset: {out}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
