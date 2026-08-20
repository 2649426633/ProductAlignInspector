from __future__ import annotations

import argparse
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
from product_align_inspector.roi import config_reference_size, crop_roi, enabled_slots, roi_for_image
from product_align_inspector.se_presence import PresenceDescriptorConfig, appearance_descriptor

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
THRESHOLD_PROFILES = {"strict", "recommended", "loose"}


def collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _gamma_variant(image: np.ndarray, gamma: float) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32) / 255.0
    y = np.power(np.clip(x, 0.0, 1.0), float(gamma))
    return np.clip(y * 255.0, 0.0, 255.0).astype(np.uint8)


def _gradient_variant(image: np.ndarray, axis: int, reverse: bool, strength: float = 0.35) -> np.ndarray:
    h, w = image.shape[:2]
    if axis == 0:
        ramp = np.linspace(1.0 - strength, 1.0 + strength, h, dtype=np.float32)[:, None]
        ramp = np.repeat(ramp, w, axis=1)
    else:
        ramp = np.linspace(1.0 - strength, 1.0 + strength, w, dtype=np.float32)[None, :]
        ramp = np.repeat(ramp, h, axis=0)
    if reverse:
        ramp = np.flip(ramp, axis=axis)
    if image.ndim == 3:
        ramp = ramp[..., None]
    return np.clip(np.asarray(image, dtype=np.float32) * ramp, 0.0, 255.0).astype(np.uint8)


def _lighting_variants(crop: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Training-only illumination augmentation; geometry is never changed."""
    return [
        ("identity", crop),
        ("gamma_065", _gamma_variant(crop, 0.65)),
        ("gamma_080", _gamma_variant(crop, 0.80)),
        ("gamma_125", _gamma_variant(crop, 1.25)),
        ("gamma_155", _gamma_variant(crop, 1.55)),
        ("shade_left_right", _gradient_variant(crop, axis=1, reverse=False)),
        ("shade_right_left", _gradient_variant(crop, axis=1, reverse=True)),
        ("shade_top_bottom", _gradient_variant(crop, axis=0, reverse=False)),
        ("shade_bottom_top", _gradient_variant(crop, axis=0, reverse=True)),
    ]


def _topk_distance_excluding_group(
    vector: np.ndarray,
    bank: np.ndarray,
    groups: np.ndarray,
    group_id: int,
    top_k: int,
) -> float:
    bank = np.asarray(bank, dtype=np.float32)
    groups = np.asarray(groups, dtype=np.int32).reshape(-1)
    if bank.ndim != 2 or bank.shape[0] != groups.size:
        raise RuntimeError(f"Invalid bank/group shapes: bank={bank.shape}, groups={groups.shape}")

    valid = groups != int(group_id)
    candidate = bank[valid]
    if candidate.shape[0] == 0:
        raise RuntimeError("No other GOOD image is available for slot calibration.")

    distances = 1.0 - candidate @ np.asarray(vector, dtype=np.float32)
    k = max(1, min(int(top_k), distances.size))
    return float(np.mean(np.partition(distances, k - 1)[:k]))


def _slot_calibration(
    original_vectors: np.ndarray,
    augmented_bank: np.ndarray,
    groups: np.ndarray,
    *,
    top_k: int,
) -> dict:
    if original_vectors.ndim != 2 or original_vectors.shape[0] < 3:
        raise RuntimeError(f"Not enough GOOD samples for slot calibration: {original_vectors.shape}")

    values = np.asarray(
        [
            _topk_distance_excluding_group(vector, augmented_bank, groups, i, top_k)
            for i, vector in enumerate(original_vectors)
        ],
        dtype=np.float32,
    )

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    p95 = float(np.quantile(values, 0.95))
    p99 = float(np.quantile(values, 0.99))
    max_good = float(np.max(values))

    # Per-slot recommendations. "recommended" deliberately leaves a margin over
    # the worst leave-one-image-out GOOD sample without using any test image.
    strict = max(max_good * 1.05 + 0.001, p99 + 0.001)
    recommended = max(max_good * 1.15 + 0.002, median + 6.0 * robust_sigma)
    loose = max(max_good * 1.30 + 0.004, median + 8.0 * robust_sigma)

    return {
        "base_samples": int(original_vectors.shape[0]),
        "bank_rows": int(augmented_bank.shape[0]),
        "descriptor_dim": int(augmented_bank.shape[1]),
        "top_k": int(top_k),
        "good_distance": {
            "min": float(np.min(values)),
            "median": median,
            "mean": float(np.mean(values)),
            "p95": p95,
            "p99": p99,
            "max": max_good,
            "mad": mad,
            "robust_sigma": float(robust_sigma),
        },
        "suggested_thresholds": {
            "strict": float(strict),
            "recommended": float(recommended),
            "loose": float(loose),
        },
    }


def _save_group_model(
    output_dir: Path,
    *,
    expected: str,
    detector_name: str,
    original_vectors: dict[str, np.ndarray],
    augmented_banks: dict[str, np.ndarray],
    group_ids: dict[str, np.ndarray],
    config: dict,
    canonical_size: tuple[int, int],
    descriptor_cfg: PresenceDescriptorConfig,
    top_k: int,
    threshold_profile: str,
    images_seen: int,
    alignment_ok: int,
    skipped: list[dict],
    augmentation_names: list[str],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    banks_dir = output_dir / "banks"
    banks_dir.mkdir(parents=True, exist_ok=True)
    for old in banks_dir.glob("*"):
        if old.is_file() and old.suffix.lower() in {".npy", ".npz"}:
            old.unlink()

    slot_models: dict[str, dict] = {}
    for slot_id, bank in augmented_banks.items():
        calibration = _slot_calibration(
            original_vectors[slot_id],
            bank,
            group_ids[slot_id],
            top_k=top_k,
        )
        suggestions = calibration["suggested_thresholds"]
        threshold = float(suggestions[threshold_profile])
        bank_rel = f"banks/{slot_id}.npy"
        np.save(output_dir / bank_rel, bank.astype(np.float32))
        slot_models[slot_id] = {
            "bank": bank_rel,
            "top_k": int(top_k),
            "threshold": threshold,
            "threshold_profile": threshold_profile,
            "suggested_thresholds": suggestions,
            "calibration": calibration,
        }

    roi_size = config_reference_size(config)
    model = {
        "schema_version": 6,
        "detector": detector_name,
        "expected": expected,
        "canonical_size": [int(canonical_size[0]), int(canonical_size[1])],
        "roi_coordinate_size": None if roi_size is None else [int(roi_size[0]), int(roi_size[1])],
        "descriptor": {
            "type": "slotwise_structure_v6",
            "size": int(descriptor_cfg.size),
            "center_crop_ratio": float(descriptor_cfg.center_crop_ratio),
            "radial_bins": int(descriptor_cfg.radial_bins),
            "hist_bins": int(descriptor_cfg.hist_bins),
            "note": "absolute brightness removed; local contrast + structure + LBP",
        },
        "classifier": {
            "type": "slotwise_augmented_good_topk_cosine",
            "top_k": int(top_k),
            "threshold_profile": threshold_profile,
            "decision": "PASS when illumination-invariant score <= threshold for that exact slot",
        },
        "training": {
            "images_seen": int(images_seen),
            "alignment_ok": int(alignment_ok),
            "alignment_skipped": int(images_seen - alignment_ok),
            "skipped": skipped,
            "source": "GOOD only; every S/E position has its own bank and threshold",
            "lighting_augmentation": augmentation_names,
            "geometry_augmentation": "none",
            "calibration": "leave one source image out, including all of its augmented copies",
        },
        "slots": slot_models,
    }
    (output_dir / "model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build v6 slot-wise S/E GOOD-only models. Every position keeps one independent bank and "
            "one independent threshold. Training uses photometric augmentation only; test NG images are never used."
        )
    )
    parser.add_argument("--train-good", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ecc-accept", type=float, default=0.70)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.10)
    parser.add_argument("--descriptor-size", type=int, default=64)
    parser.add_argument("--center-crop-ratio", type=float, default=0.72)
    parser.add_argument("--radial-bins", type=int, default=10)
    parser.add_argument("--hist-bins", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--threshold-profile",
        choices=sorted(THRESHOLD_PROFILES),
        default="recommended",
        help="Active threshold profile written for each independent slot. Default: recommended.",
    )
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

    descriptor_cfg = PresenceDescriptorConfig(
        size=max(32, int(args.descriptor_size)),
        center_crop_ratio=float(args.center_crop_ratio),
        radial_bins=max(4, int(args.radial_bins)),
        hist_bins=max(8, int(args.hist_bins)),
    )
    top_k = max(1, int(args.top_k))
    threshold_profile = str(args.threshold_profile)
    align_cfg = ProductLocatorConfig(
        ecc_accept_score=float(args.ecc_accept),
        feature_min_inlier_ratio=float(args.min_inlier_ratio),
        canonical_scale=1.0,
    )

    all_slots = s_slots + e_slots
    original_rows: dict[str, list[np.ndarray]] = {str(slot["id"]): [] for slot in all_slots}
    augmented_rows: dict[str, list[np.ndarray]] = {str(slot["id"]): [] for slot in all_slots}
    group_rows: dict[str, list[int]] = {str(slot["id"]): [] for slot in all_slots}

    files = collect_images(train_root)
    if not files:
        raise SystemExit(f"No training images found under: {train_root}")

    skipped: list[dict] = []
    alignment_ok = 0
    augmentation_names: list[str] | None = None

    print("=== Build slot-wise illumination-robust S/E models v6 ===")
    print(f"Train GOOD:        {train_root}")
    print(f"Canonical:         {canonical_w}x{canonical_h}")
    print(f"S slots:           {[slot['id'] for slot in s_slots]}")
    print(f"E slots:           {[slot['id'] for slot in e_slots]}")
    print("Model rule:        one slot = one bank = one threshold")
    print("Descriptor:        local-contrast structure + rotation-invariant LBP")
    print("Lighting augment:  gamma + smooth directional shading")
    print("Geometry augment:  NONE")
    print(f"Threshold profile: {threshold_profile}")
    print("Test NG used:      NO")
    print()

    for source_index, path in enumerate(files):
        index = source_index + 1
        try:
            raw = read_image(path)
            alignment = align_to_reference(raw, reference, align_cfg)
            canonical = alignment.aligned

            for slot in all_slots:
                slot_id = str(slot["id"])
                roi = roi_for_image(slot["roi"], config, canonical_w, canonical_h)
                crop = crop_roi(canonical, roi)
                variants = _lighting_variants(crop)
                if augmentation_names is None:
                    augmentation_names = [name for name, _ in variants]

                original_rows[slot_id].append(appearance_descriptor(crop, descriptor_cfg))
                for _, variant in variants:
                    augmented_rows[slot_id].append(appearance_descriptor(variant, descriptor_cfg))
                    group_rows[slot_id].append(source_index)

            alignment_ok += 1
            print(f"[{index}/{len(files)}] {path.name} -> OK | ecc={alignment.ecc_score:.4f}")
        except Exception as exc:
            skipped.append({"path": str(path), "error": str(exc)})
            print(f"[{index}/{len(files)}] {path.name} -> SKIP | {exc}")

    originals: dict[str, np.ndarray] = {}
    banks: dict[str, np.ndarray] = {}
    groups: dict[str, np.ndarray] = {}
    for slot in all_slots:
        slot_id = str(slot["id"])
        if len(original_rows[slot_id]) < 3:
            raise RuntimeError(f"Not enough aligned GOOD samples for {slot_id}: {len(original_rows[slot_id])}")
        originals[slot_id] = np.stack(original_rows[slot_id]).astype(np.float32)
        banks[slot_id] = np.stack(augmented_rows[slot_id]).astype(np.float32)
        groups[slot_id] = np.asarray(group_rows[slot_id], dtype=np.int32)

    if augmentation_names is None:
        raise RuntimeError("No augmentation variants were generated.")

    s_ids = {str(slot["id"]) for slot in s_slots}
    e_ids = {str(slot["id"]) for slot in e_slots}

    s_model = _save_group_model(
        output / "S",
        expected="screw",
        detector_name="S_presence",
        original_vectors={k: v for k, v in originals.items() if k in s_ids},
        augmented_banks={k: v for k, v in banks.items() if k in s_ids},
        group_ids={k: v for k, v in groups.items() if k in s_ids},
        config=config,
        canonical_size=(canonical_w, canonical_h),
        descriptor_cfg=descriptor_cfg,
        top_k=top_k,
        threshold_profile=threshold_profile,
        images_seen=len(files),
        alignment_ok=alignment_ok,
        skipped=skipped,
        augmentation_names=augmentation_names,
    )
    e_model = _save_group_model(
        output / "E",
        expected="empty",
        detector_name="E_empty",
        original_vectors={k: v for k, v in originals.items() if k in e_ids},
        augmented_banks={k: v for k, v in banks.items() if k in e_ids},
        group_ids={k: v for k, v in groups.items() if k in e_ids},
        config=config,
        canonical_size=(canonical_w, canonical_h),
        descriptor_cfg=descriptor_cfg,
        top_k=top_k,
        threshold_profile=threshold_profile,
        images_seen=len(files),
        alignment_ok=alignment_ok,
        skipped=skipped,
        augmentation_names=augmentation_names,
    )

    summary_slots: dict[str, dict] = {}
    for model in (s_model, e_model):
        for slot_id, row in model["slots"].items():
            summary_slots[slot_id] = {
                "active_threshold": row["threshold"],
                "profile": row["threshold_profile"],
                "suggested_thresholds": row["suggested_thresholds"],
                "good_distance": row["calibration"]["good_distance"],
                "base_samples": row["calibration"]["base_samples"],
                "bank_rows": row["calibration"]["bank_rows"],
            }

    summary = {
        "schema_version": 6,
        "descriptor": "slotwise_structure_v6",
        "classifier": "slotwise_augmented_good_topk_cosine",
        "threshold_profile": threshold_profile,
        "images_seen": len(files),
        "alignment_ok": alignment_ok,
        "alignment_skipped": len(files) - alignment_ok,
        "lighting_augmentation": augmentation_names,
        "slots": summary_slots,
        "S_model": str(output / "S" / "model.json"),
        "E_model": str(output / "E" / "model.json"),
    }
    (output / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=== Suggested thresholds by slot ===")
    print("slot   good_max    strict      recommended   loose        active      bank")
    for slot_id in [str(slot["id"]) for slot in all_slots]:
        row = summary_slots[slot_id]
        suggestions = row["suggested_thresholds"]
        good_max = row["good_distance"]["max"]
        print(
            f"{slot_id:<5}  {good_max:>9.6f}  {suggestions['strict']:>10.6f}  "
            f"{suggestions['recommended']:>12.6f}  {suggestions['loose']:>10.6f}  "
            f"{row['active_threshold']:>10.6f}  {row['bank_rows']:>5d}"
        )

    print()
    print("=== Build Summary ===")
    print(f"GOOD images:       {len(files)}")
    print(f"Alignment OK:      {alignment_ok}")
    print(f"Alignment skipped: {len(files) - alignment_ok}")
    print(f"Augmentations:     {len(augmentation_names)} per GOOD ROI")
    print(f"Threshold profile: {threshold_profile}")
    print(f"S model:           {output / 'S' / 'model.json'}")
    print(f"E model:           {output / 'E' / 'model.json'}")
    print(f"Summary:           {output / 'build_summary.json'}")
    print()
    print("Threshold guidance:")
    print("  strict      -> more sensitive, more GOOD false-alarm risk")
    print("  recommended -> default starting point")
    print("  loose       -> only after NG recall is verified")
    print("Each S/E slot still has one independent threshold in model.json.")


if __name__ == "__main__":
    main()
