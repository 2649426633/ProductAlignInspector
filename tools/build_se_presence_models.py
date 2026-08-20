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
from product_align_inspector.se_presence import PresenceDescriptorConfig, appearance_descriptor

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
THRESHOLD_PROFILES = {"strict", "recommended", "loose"}


def collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _topk_distance(vector: np.ndarray, bank: np.ndarray, top_k: int, exclude: int | None = None) -> float:
    distances = 1.0 - np.asarray(bank, dtype=np.float32) @ np.asarray(vector, dtype=np.float32)
    if exclude is not None and 0 <= exclude < distances.size:
        distances = distances.copy()
        distances[exclude] = np.inf
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        raise RuntimeError("No neighbors available for slot calibration.")
    k = max(1, min(int(top_k), finite.size))
    return float(np.mean(np.partition(finite, k - 1)[:k]))


def _slot_calibration(bank: np.ndarray, top_k: int) -> dict:
    if bank.ndim != 2 or bank.shape[0] < 3:
        raise RuntimeError(f"Not enough GOOD samples for slot calibration: {bank.shape}")

    distances = np.asarray(
        [_topk_distance(vector, bank, top_k, exclude=i) for i, vector in enumerate(bank)],
        dtype=np.float32,
    )
    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median)))
    robust_sigma = 1.4826 * mad
    p95 = float(np.quantile(distances, 0.95))
    p99 = float(np.quantile(distances, 0.99))
    max_good = float(np.max(distances))

    # Three practical operating points. All are calculated independently for each ROI.
    # strict:       lowest false-negative tolerance, higher false-alarm risk.
    # recommended:  default production starting point.
    # loose:        lower false-alarm risk, but may miss subtle defects.
    strict = max(max_good * 1.05 + 0.001, p99 + 0.001)
    recommended = max(max_good * 1.15 + 0.002, median + 6.0 * robust_sigma)
    loose = max(max_good * 1.30 + 0.004, median + 8.0 * robust_sigma)

    return {
        "samples": int(bank.shape[0]),
        "descriptor_dim": int(bank.shape[1]),
        "top_k": int(top_k),
        "good_distance": {
            "min": float(np.min(distances)),
            "median": median,
            "mean": float(np.mean(distances)),
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
    slot_banks: dict[str, np.ndarray],
    config: dict,
    canonical_size: tuple[int, int],
    descriptor_cfg: PresenceDescriptorConfig,
    top_k: int,
    threshold_profile: str,
    images_seen: int,
    alignment_ok: int,
    skipped: list[dict],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    banks_dir = output_dir / "banks"
    banks_dir.mkdir(parents=True, exist_ok=True)
    for old in banks_dir.glob("*.npy"):
        old.unlink()

    slot_models: dict[str, dict] = {}
    for slot_id, bank in slot_banks.items():
        calibration = _slot_calibration(bank, top_k)
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
        "schema_version": 4,
        "detector": detector_name,
        "expected": expected,
        "canonical_size": [int(canonical_size[0]), int(canonical_size[1])],
        "roi_coordinate_size": None if roi_size is None else [int(roi_size[0]), int(roi_size[1])],
        "descriptor": {
            "type": "slotwise_radial_v4",
            "size": int(descriptor_cfg.size),
            "center_crop_ratio": float(descriptor_cfg.center_crop_ratio),
            "radial_bins": int(descriptor_cfg.radial_bins),
            "hist_bins": int(descriptor_cfg.hist_bins),
        },
        "classifier": {
            "type": "slotwise_good_topk_cosine",
            "top_k": int(top_k),
            "threshold_profile": threshold_profile,
            "decision": "PASS when score <= the threshold stored for that exact slot",
        },
        "training": {
            "images_seen": int(images_seen),
            "alignment_ok": int(alignment_ok),
            "alignment_skipped": int(images_seen - alignment_ok),
            "skipped": skipped,
            "source": "GOOD only; every S/E position has its own bank and its own threshold",
        },
        "slots": slot_models,
    }
    (output_dir / "model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build slot-wise S/E GOOD-only models. S01/S02 and E01..E09 each receive an independent "
            "GOOD bank and an independent threshold. No test NG images are used for training."
        )
    )
    parser.add_argument("--train-good", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ecc-accept", type=float, default=0.70)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.10)
    parser.add_argument("--descriptor-size", type=int, default=64)
    parser.add_argument("--center-crop-ratio", type=float, default=0.78)
    parser.add_argument("--radial-bins", type=int, default=10)
    parser.add_argument("--hist-bins", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--threshold-profile",
        choices=sorted(THRESHOLD_PROFILES),
        default="recommended",
        help="Threshold set written as the active threshold for every slot. Default: recommended.",
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
    slot_rows: dict[str, list[np.ndarray]] = {str(slot["id"]): [] for slot in all_slots}
    files = collect_images(train_root)
    if not files:
        raise SystemExit(f"No training images found under: {train_root}")

    skipped: list[dict] = []
    alignment_ok = 0
    print("=== Build slot-wise S/E GOOD-only models v4 ===")
    print(f"Train GOOD:        {train_root}")
    print(f"Canonical:         {canonical_w}x{canonical_h}")
    print(f"S slots:           {[slot['id'] for slot in s_slots]}")
    print(f"E slots:           {[slot['id'] for slot in e_slots]}")
    print("Model rule:        one slot = one bank = one threshold")
    print(f"Threshold profile: {threshold_profile}")
    print("Test NG used:      NO")
    print()

    for index, path in enumerate(files, 1):
        try:
            raw = read_image(path)
            alignment = align_to_reference(raw, reference, align_cfg)
            canonical = alignment.aligned
            for slot in all_slots:
                slot_id = str(slot["id"])
                roi = roi_for_image(slot["roi"], config, canonical_w, canonical_h)
                descriptor = appearance_descriptor(crop_roi(canonical, roi), descriptor_cfg)
                slot_rows[slot_id].append(descriptor)
            alignment_ok += 1
            print(f"[{index}/{len(files)}] {path.name} -> OK | ecc={alignment.ecc_score:.4f}")
        except Exception as exc:
            skipped.append({"path": str(path), "error": str(exc)})
            print(f"[{index}/{len(files)}] {path.name} -> SKIP | {exc}")

    slot_banks: dict[str, np.ndarray] = {}
    for slot_id, rows in slot_rows.items():
        if len(rows) < 3:
            raise RuntimeError(f"Not enough aligned GOOD samples for {slot_id}: {len(rows)}")
        slot_banks[slot_id] = np.stack(rows).astype(np.float32)

    s_ids = {str(slot["id"]) for slot in s_slots}
    e_ids = {str(slot["id"]) for slot in e_slots}
    s_model = _save_group_model(
        output / "S",
        expected="screw",
        detector_name="S_presence",
        slot_banks={slot_id: bank for slot_id, bank in slot_banks.items() if slot_id in s_ids},
        config=config,
        canonical_size=(canonical_w, canonical_h),
        descriptor_cfg=descriptor_cfg,
        top_k=top_k,
        threshold_profile=threshold_profile,
        images_seen=len(files),
        alignment_ok=alignment_ok,
        skipped=skipped,
    )
    e_model = _save_group_model(
        output / "E",
        expected="empty",
        detector_name="E_empty",
        slot_banks={slot_id: bank for slot_id, bank in slot_banks.items() if slot_id in e_ids},
        config=config,
        canonical_size=(canonical_w, canonical_h),
        descriptor_cfg=descriptor_cfg,
        top_k=top_k,
        threshold_profile=threshold_profile,
        images_seen=len(files),
        alignment_ok=alignment_ok,
        skipped=skipped,
    )

    summary_slots: dict[str, dict] = {}
    for model in (s_model, e_model):
        for slot_id, row in model["slots"].items():
            summary_slots[slot_id] = {
                "active_threshold": row["threshold"],
                "profile": row["threshold_profile"],
                "suggested_thresholds": row["suggested_thresholds"],
                "good_distance": row["calibration"]["good_distance"],
            }

    summary = {
        "schema_version": 4,
        "descriptor": "slotwise_radial_v4",
        "classifier": "slotwise_good_topk_cosine",
        "threshold_profile": threshold_profile,
        "images_seen": len(files),
        "alignment_ok": alignment_ok,
        "alignment_skipped": len(files) - alignment_ok,
        "slots": summary_slots,
        "S_model": str(output / "S" / "model.json"),
        "E_model": str(output / "E" / "model.json"),
    }
    (output / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=== Suggested thresholds by slot ===")
    print("slot   good_max    strict      recommended   loose        active")
    for slot_id in [str(slot["id"]) for slot in all_slots]:
        row = summary_slots[slot_id]
        suggestions = row["suggested_thresholds"]
        good_max = row["good_distance"]["max"]
        print(
            f"{slot_id:<5}  {good_max:>9.6f}  {suggestions['strict']:>10.6f}  "
            f"{suggestions['recommended']:>12.6f}  {suggestions['loose']:>10.6f}  "
            f"{row['active_threshold']:>10.6f}"
        )

    print()
    print("=== Build Summary ===")
    print(f"GOOD images:       {len(files)}")
    print(f"Alignment OK:      {alignment_ok}")
    print(f"Alignment skipped: {len(files) - alignment_ok}")
    print(f"Threshold profile: {threshold_profile}")
    print(f"S model:           {output / 'S' / 'model.json'}")
    print(f"E model:           {output / 'E' / 'model.json'}")
    print(f"Summary:           {output / 'build_summary.json'}")
    print()
    print("Threshold guidance:")
    print("  strict      -> use if missing/extra screw defects are being missed; false alarms may increase")
    print("  recommended -> default starting point for production tuning")
    print("  loose       -> use only if GOOD false alarms remain after checking alignment/ROI quality")
    print("You can later edit only one slot's threshold in model.json without changing the other 10 slots.")


if __name__ == "__main__":
    main()
