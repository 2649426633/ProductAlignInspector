from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.io_utils import read_image
from product_align_inspector.roi import config_reference_size, crop_roi, enabled_slots, roi_for_image
from product_align_inspector.se_presence import (
    PresenceDescriptorConfig,
    appearance_descriptor,
    leave_one_out_distances,
    threshold_from_good,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def save_model(
    output_dir: Path,
    *,
    expected: str,
    detector_name: str,
    banks: dict[str, list[np.ndarray]],
    config: dict,
    canonical_size: tuple[int, int],
    descriptor_cfg: PresenceDescriptorConfig,
    margin_factor: float,
    margin_abs: float,
    images_seen: int,
    alignment_ok: int,
    skipped: list[dict],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    banks_dir = output_dir / "banks"
    banks_dir.mkdir(parents=True, exist_ok=True)

    thresholds: dict[str, float] = {}
    slot_stats: dict[str, dict] = {}
    for slot_id, rows in banks.items():
        if len(rows) < 2:
            raise RuntimeError(f"Not enough aligned GOOD samples for {slot_id}: {len(rows)}")
        bank = np.stack(rows).astype(np.float32)
        np.save(banks_dir / f"{slot_id}.npy", bank)
        loo = leave_one_out_distances(bank)
        threshold = threshold_from_good(loo, margin_factor=margin_factor, margin_abs=margin_abs)
        thresholds[slot_id] = threshold
        slot_stats[slot_id] = {
            "samples": int(bank.shape[0]),
            "descriptor_dim": int(bank.shape[1]),
            "good_distance_min": float(np.min(loo)),
            "good_distance_mean": float(np.mean(loo)),
            "good_distance_max": float(np.max(loo)),
            "threshold": float(threshold),
        }

    roi_size = config_reference_size(config)
    model = {
        "schema_version": 1,
        "detector": detector_name,
        "expected": expected,
        "canonical_size": [int(canonical_size[0]), int(canonical_size[1])],
        "roi_coordinate_size": None if roi_size is None else [int(roi_size[0]), int(roi_size[1])],
        "descriptor": {
            "size": int(descriptor_cfg.size),
            "gray_weight": float(descriptor_cfg.gray_weight),
            "gradient_weight": float(descriptor_cfg.gradient_weight),
            "distance": "nearest_cosine",
        },
        "thresholds": thresholds,
        "slot_stats": slot_stats,
        "training": {
            "images_seen": int(images_seen),
            "alignment_ok": int(alignment_ok),
            "alignment_skipped": int(images_seen - alignment_ok),
            "skipped": skipped,
            "source": "GOOD only",
        },
        "decision": {
            "normal": "score <= threshold",
            "ng": "score > threshold",
            "ng_reason": "missing_screw" if expected == "screw" else "excess_screw",
        },
    }
    (output_dir / "model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build two independent GOOD-only appearance models: S screw-present and E empty. "
            "Training images are first aligned to the canonical reference."
        )
    )
    parser.add_argument("--train-good", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ecc-accept", type=float, default=0.70)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.10)
    parser.add_argument("--descriptor-size", type=int, default=40)
    parser.add_argument("--margin-factor", type=float, default=1.25)
    parser.add_argument("--margin-abs", type=float, default=0.002)
    args = parser.parse_args()

    train_root = Path(args.train_good).resolve()
    reference = read_image(Path(args.reference).resolve())
    config = json.loads(Path(args.config).resolve().read_text(encoding="utf-8"))
    output = Path(args.output).resolve()

    canonical_h, canonical_w = reference.shape[:2]
    s_slots = enabled_slots(config, expected="screw")
    e_slots = enabled_slots(config, expected="empty")
    if not s_slots:
        raise SystemExit("No enabled S/screw slots found in config.")
    if not e_slots:
        raise SystemExit("No enabled E/empty slots found in config.")

    descriptor_cfg = PresenceDescriptorConfig(size=max(16, int(args.descriptor_size)))
    align_cfg = ProductLocatorConfig(
        ecc_accept_score=float(args.ecc_accept),
        feature_min_inlier_ratio=float(args.min_inlier_ratio),
        canonical_scale=1.0,
    )

    s_banks: dict[str, list[np.ndarray]] = {str(slot["id"]): [] for slot in s_slots}
    e_banks: dict[str, list[np.ndarray]] = {str(slot["id"]): [] for slot in e_slots}
    files = collect_images(train_root)
    if not files:
        raise SystemExit(f"No training images found under: {train_root}")

    skipped: list[dict] = []
    alignment_ok = 0
    print("=== Build S/E GOOD-only presence models ===")
    print(f"Train GOOD:      {train_root}")
    print(f"Canonical:       {canonical_w}x{canonical_h}")
    print(f"S slots:         {[slot['id'] for slot in s_slots]}")
    print(f"E slots:         {[slot['id'] for slot in e_slots]}")
    print("Models:          S and E are built separately")
    print()

    for index, path in enumerate(files, 1):
        try:
            raw = read_image(path)
            alignment = align_to_reference(raw, reference, align_cfg)
            canonical = alignment.aligned
            for slot in s_slots:
                roi = roi_for_image(slot["roi"], config, canonical_w, canonical_h)
                s_banks[str(slot["id"])].append(appearance_descriptor(crop_roi(canonical, roi), descriptor_cfg))
            for slot in e_slots:
                roi = roi_for_image(slot["roi"], config, canonical_w, canonical_h)
                e_banks[str(slot["id"])].append(appearance_descriptor(crop_roi(canonical, roi), descriptor_cfg))
            alignment_ok += 1
            print(f"[{index}/{len(files)}] {path.name} -> OK | ecc={alignment.ecc_score:.4f}")
        except Exception as exc:
            skipped.append({"path": str(path), "error": str(exc)})
            print(f"[{index}/{len(files)}] {path.name} -> SKIP | {exc}")

    s_model = save_model(
        output / "S",
        expected="screw",
        detector_name="S_presence",
        banks=s_banks,
        config=config,
        canonical_size=(canonical_w, canonical_h),
        descriptor_cfg=descriptor_cfg,
        margin_factor=float(args.margin_factor),
        margin_abs=float(args.margin_abs),
        images_seen=len(files),
        alignment_ok=alignment_ok,
        skipped=skipped,
    )
    e_model = save_model(
        output / "E",
        expected="empty",
        detector_name="E_empty",
        banks=e_banks,
        config=config,
        canonical_size=(canonical_w, canonical_h),
        descriptor_cfg=descriptor_cfg,
        margin_factor=float(args.margin_factor),
        margin_abs=float(args.margin_abs),
        images_seen=len(files),
        alignment_ok=alignment_ok,
        skipped=skipped,
    )

    summary = {
        "images_seen": len(files),
        "alignment_ok": alignment_ok,
        "alignment_skipped": len(files) - alignment_ok,
        "S_model": str(output / "S" / "model.json"),
        "E_model": str(output / "E" / "model.json"),
        "S_thresholds": s_model["thresholds"],
        "E_thresholds": e_model["thresholds"],
    }
    (output / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=== Build Summary ===")
    print(f"GOOD images:       {len(files)}")
    print(f"Alignment OK:      {alignment_ok}")
    print(f"Alignment skipped: {len(files) - alignment_ok}")
    print(f"S model:           {output / 'S' / 'model.json'}")
    print(f"E model:           {output / 'E' / 'model.json'}")
    print(f"Summary:           {output / 'build_summary.json'}")


if __name__ == "__main__":
    main()
