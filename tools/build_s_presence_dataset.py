from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.io_utils import read_image, write_image
from product_align_inspector.roi import crop_roi, validate_roi

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the S-slot supervised presence dataset: GOOD S slots are screw, "
            "all_empty S slots are empty. E slots are intentionally excluded."
        )
    )
    parser.add_argument("--good-dir", required=True)
    parser.add_argument("--all-empty-dir", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/presence_s_dataset")
    parser.add_argument("--threshold", type=int, default=238)
    args = parser.parse_args()

    good_dir = Path(args.good_dir)
    empty_dir = Path(args.all_empty_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if not good_dir.is_dir():
        raise SystemExit(f"GOOD directory not found: {good_dir}")
    if not empty_dir.is_dir():
        raise SystemExit(f"all_empty directory not found: {empty_dir}")

    reference = read_image(args.reference)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    slots = [
        item
        for item in config.get("screw_slots", [])
        if bool(item.get("enabled", True)) and str(item.get("id", "")).upper().startswith("S")
    ]
    if not slots:
        raise SystemExit("No enabled Sxx slots found in config")

    align_cfg = ProductLocatorConfig(foreground_threshold=args.threshold)
    sources = [
        ("good", "screw", path) for path in collect_images(good_dir)
    ] + [
        ("all_empty", "empty", path) for path in collect_images(empty_dir)
    ]
    if not sources:
        raise SystemExit("No input images found")

    rows: list[dict[str, object]] = []
    success_images = 0
    crop_counts = {"screw": 0, "empty": 0}

    print("=== Build S Presence Dataset ===")
    print(f"S slots: {[str(s.get('id')) for s in slots]}")
    print(f"GOOD images: {sum(1 for group, _, _ in sources if group == 'good')}")
    print(f"all_empty images: {sum(1 for group, _, _ in sources if group == 'all_empty')}")
    print(f"Output: {output}")

    for index, (group, label, image_path) in enumerate(sources, 1):
        try:
            raw = read_image(image_path)
            result = align_to_reference(raw, reference, align_cfg)
            aligned = result.aligned
            h, w = aligned.shape[:2]
            source_id = f"{group}/{image_path.name}"

            for slot in slots:
                slot_id = str(slot.get("id"))
                roi = slot.get("roi")
                if roi is None or not validate_roi(roi, w, h):
                    raise RuntimeError(f"Invalid ROI {slot_id}: {roi}")
                crop = crop_roi(aligned, roi)
                filename = f"{group}__{image_path.stem}__{slot_id}.png"
                crop_path = output / "screw" / label / filename
                write_image(crop_path, crop)
                crop_counts[label] += 1
                rows.append(
                    {
                        "source": source_id,
                        "source_image": str(image_path),
                        "scenario": group,
                        "kind": "screw_slot",
                        "id": slot_id,
                        "label": label,
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

            success_images += 1
            print(
                f"[{index}/{len(sources)}] {group}/{image_path.name} -> OK "
                f"({result.method}, inliers={result.feature_inlier_ratio:.1%})"
            )
        except Exception as exc:
            rows.append(
                {
                    "source": f"{group}/{image_path.name}",
                    "source_image": str(image_path),
                    "scenario": group,
                    "kind": "",
                    "id": "",
                    "label": "",
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
            print(f"[{index}/{len(sources)}] {group}/{image_path.name} -> FAILED: {exc}")

    manifest_path = output / "manifest.csv"
    fieldnames = [
        "source",
        "source_image",
        "scenario",
        "kind",
        "id",
        "label",
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
        writer.writerows(rows)

    print("")
    print("=== S Dataset Ready ===")
    print(f"Aligned source images: {success_images}/{len(sources)}")
    print(f"SCREW crops: {crop_counts['screw']}")
    print(f"EMPTY crops: {crop_counts['empty']}")
    print(f"Manifest: {manifest_path}")
    print("E slots were not included; train the E model from its own labeled E dataset.")


if __name__ == "__main__":
    main()
