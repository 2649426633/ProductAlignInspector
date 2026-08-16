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


def _scenario_overrides(name: str) -> dict[str, str]:
    """Parse folder names such as normal, missing_S01, extra_E01, missing_S01+missing_S02."""
    name = name.strip()
    if name.lower() in {"normal", "good", "ok"}:
        return {}

    overrides: dict[str, str] = {}
    for token in name.split("+"):
        token = token.strip()
        lower = token.lower()
        if lower.startswith("missing_"):
            slot_id = token[len("missing_"):]
            overrides[slot_id] = "empty"
        elif lower.startswith("extra_"):
            slot_id = token[len("extra_"):]
            overrides[slot_id] = "screw"
        else:
            raise ValueError(
                f"Unsupported scenario folder '{name}'. Use normal, missing_S01, extra_E01, "
                "or combine with +, e.g. missing_S01+missing_S02."
            )
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a screw/empty ROI dataset from real assembly scenarios."
    )
    parser.add_argument("--input-dir", required=True, help="Root containing scenario subfolders")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=int, default=238)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.25)
    args = parser.parse_args()

    root = Path(args.input_dir)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise SystemExit(f"Input directory not found: {root}")

    reference = read_image(args.reference)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    align_cfg = ProductLocatorConfig(foreground_threshold=args.threshold)

    slots = [s for s in config.get("screw_slots", []) if bool(s.get("enabled", True))]
    if not slots:
        raise SystemExit("No enabled screw_slots in config")

    scenario_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not scenario_dirs:
        raise SystemExit(
            f"No scenario folders found under {root}. Create folders such as normal, missing_S01, missing_S02."
        )

    rows: list[dict[str, object]] = []
    source_count = 0
    crop_count = 0

    for scenario_dir in scenario_dirs:
        try:
            overrides = _scenario_overrides(scenario_dir.name)
        except ValueError as exc:
            print(f"SKIP {scenario_dir.name}: {exc}")
            continue

        images = sorted(
            p for p in scenario_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        print(f"Scenario {scenario_dir.name}: {len(images)} image(s), overrides={overrides}")

        for index, image_path in enumerate(images, 1):
            try:
                raw = read_image(image_path)
                result = align_to_reference(raw, reference, align_cfg)
                if result.method.startswith("sift") and result.feature_inlier_ratio < args.min_inlier_ratio:
                    raise RuntimeError(
                        f"weak alignment inlier_ratio={result.feature_inlier_ratio:.3f}"
                    )

                aligned = result.aligned
                h, w = aligned.shape[:2]
                source_id = f"{scenario_dir.name}/{image_path.relative_to(scenario_dir).as_posix()}"

                for slot in slots:
                    slot_id = str(slot.get("id"))
                    roi = slot.get("roi")
                    if roi is None or not validate_roi(roi, w, h):
                        raise RuntimeError(f"invalid ROI {slot_id}: {roi}")

                    expected = str(slot.get("expected", "screw"))
                    label = overrides.get(slot_id, expected)
                    if label not in {"screw", "empty"}:
                        raise RuntimeError(f"unsupported label for {slot_id}: {label}")

                    crop = crop_roi(aligned, roi)
                    safe_stem = image_path.stem.replace(" ", "_")
                    filename = f"{scenario_dir.name}__{safe_stem}__{slot_id}.png"
                    crop_path = out / "screw" / label / filename
                    write_image(crop_path, crop)

                    rows.append(
                        {
                            "source": source_id,
                            "scenario": scenario_dir.name,
                            "source_image": str(image_path),
                            "kind": "screw_slot",
                            "id": slot_id,
                            "label": label,
                            "config_expected": expected,
                            "scenario_override": overrides.get(slot_id, ""),
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
                    crop_count += 1

                source_count += 1
                print(
                    f"  [{index}/{len(images)}] {image_path.name} -> OK "
                    f"({result.method}, inliers={result.feature_inlier_ratio:.1%})"
                )
            except Exception as exc:
                rows.append(
                    {
                        "source": f"{scenario_dir.name}/{image_path.name}",
                        "scenario": scenario_dir.name,
                        "source_image": str(image_path),
                        "kind": "",
                        "id": "",
                        "label": "",
                        "config_expected": "",
                        "scenario_override": "",
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
                print(f"  [{index}/{len(images)}] {image_path.name} -> FAILED: {exc}")

    manifest_path = out / "manifest.csv"
    fieldnames = [
        "source",
        "scenario",
        "source_image",
        "kind",
        "id",
        "label",
        "config_expected",
        "scenario_override",
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

    screw_count = sum(1 for r in rows if r.get("status") == "ok" and r.get("label") == "screw")
    empty_count = sum(1 for r in rows if r.get("status") == "ok" and r.get("label") == "empty")

    print("")
    print("=== Dataset Summary ===")
    print(f"Source images: {source_count}")
    print(f"ROI crops:     {crop_count}")
    print(f"screw:         {screw_count}")
    print(f"empty:         {empty_count}")
    print(f"Manifest:      {manifest_path}")


if __name__ == "__main__":
    main()
