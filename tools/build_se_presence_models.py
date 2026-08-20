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


def collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _topk_distance(vector: np.ndarray, bank: np.ndarray, top_k: int, exclude: int | None = None) -> float:
    similarity = np.asarray(bank, dtype=np.float32) @ np.asarray(vector, dtype=np.float32)
    distances = 1.0 - similarity
    if exclude is not None and 0 <= exclude < distances.size:
        distances = distances.copy()
        distances[exclude] = np.inf
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        raise RuntimeError("No neighbors available for calibration.")
    k = max(1, min(int(top_k), finite.size))
    return float(np.mean(np.partition(finite, k - 1)[:k]))


def _calibrate_thresholds(screw_bank: np.ndarray, empty_bank: np.ndarray, top_k: int) -> tuple[float, float, dict]:
    screw_margins = []
    for i, vector in enumerate(screw_bank):
        d_screw = _topk_distance(vector, screw_bank, top_k, exclude=i)
        d_empty = _topk_distance(vector, empty_bank, top_k)
        screw_margins.append(d_empty - d_screw)

    empty_margins = []
    for i, vector in enumerate(empty_bank):
        d_screw = _topk_distance(vector, screw_bank, top_k)
        d_empty = _topk_distance(vector, empty_bank, top_k, exclude=i)
        empty_margins.append(d_empty - d_screw)

    screw_margins = np.asarray(screw_margins, dtype=np.float32)
    empty_margins = np.asarray(empty_margins, dtype=np.float32)

    screw_low = float(np.quantile(screw_margins, 0.01))
    empty_high = float(np.quantile(empty_margins, 0.99))
    if empty_high < screw_low:
        gap = screw_low - empty_high
        s_threshold = screw_low - 0.15 * gap
        e_threshold = empty_high + 0.15 * gap
        calibration = "separated"
    else:
        screw_med = float(np.median(screw_margins))
        empty_med = float(np.median(empty_margins))
        boundary = 0.5 * (screw_med + empty_med)
        s_threshold = boundary
        e_threshold = boundary
        calibration = "overlap"

    stats = {
        "calibration": calibration,
        "margin_definition": "distance_empty - distance_screw; positive=screw-like, negative=empty-like",
        "screw": {
            "count": int(screw_margins.size),
            "min": float(screw_margins.min()),
            "p01": screw_low,
            "median": float(np.median(screw_margins)),
            "max": float(screw_margins.max()),
        },
        "empty": {
            "count": int(empty_margins.size),
            "min": float(empty_margins.min()),
            "median": float(np.median(empty_margins)),
            "p99": empty_high,
            "max": float(empty_margins.max()),
        },
        "S_threshold": float(s_threshold),
        "E_threshold": float(e_threshold),
    }
    return float(s_threshold), float(e_threshold), stats


def _save_model(
    output_dir: Path,
    *,
    expected: str,
    detector_name: str,
    threshold: float,
    screw_bank: np.ndarray,
    empty_bank: np.ndarray,
    config: dict,
    canonical_size: tuple[int, int],
    descriptor_cfg: PresenceDescriptorConfig,
    top_k: int,
    images_seen: int,
    alignment_ok: int,
    skipped: list[dict],
    calibration_stats: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    banks_dir = output_dir / "banks"
    banks_dir.mkdir(parents=True, exist_ok=True)
    for old in banks_dir.glob("*.npy"):
        old.unlink()
    np.save(banks_dir / "screw.npy", screw_bank.astype(np.float32))
    np.save(banks_dir / "empty.npy", empty_bank.astype(np.float32))

    roi_size = config_reference_size(config)
    if expected == "screw":
        normal = "margin >= threshold"
        ng = "margin < threshold"
        ng_reason = "missing_screw"
    else:
        normal = "margin <= threshold"
        ng = "margin > threshold"
        ng_reason = "excess_screw"

    model = {
        "schema_version": 3,
        "detector": detector_name,
        "expected": expected,
        "canonical_size": [int(canonical_size[0]), int(canonical_size[1])],
        "roi_coordinate_size": None if roi_size is None else [int(roi_size[0]), int(roi_size[1])],
        "descriptor": {
            "type": "radial_presence_v3",
            "size": int(descriptor_cfg.size),
            "center_crop_ratio": float(descriptor_cfg.center_crop_ratio),
            "radial_bins": int(descriptor_cfg.radial_bins),
            "hist_bins": int(descriptor_cfg.hist_bins),
        },
        "classifier": {
            "type": "dual_bank_topk_cosine",
            "top_k": int(top_k),
            "threshold": float(threshold),
            "margin": "distance_empty - distance_screw",
        },
        "training": {
            "images_seen": int(images_seen),
            "alignment_ok": int(alignment_ok),
            "alignment_skipped": int(images_seen - alignment_ok),
            "skipped": skipped,
            "source": "GOOD only: real S slots provide screw class; real E slots provide empty class",
            "screw_samples": int(screw_bank.shape[0]),
            "empty_samples": int(empty_bank.shape[0]),
        },
        "calibration": calibration_stats,
        "decision": {
            "normal": normal,
            "ng": ng,
            "ng_reason": ng_reason,
        },
    }
    (output_dir / "model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build independent S/E models from GOOD images only. Each GOOD image already contains "
            "real screw examples in S01/S02 and real empty examples in E01..E09, so no NG test "
            "images are used for training."
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
    align_cfg = ProductLocatorConfig(
        ecc_accept_score=float(args.ecc_accept),
        feature_min_inlier_ratio=float(args.min_inlier_ratio),
        canonical_scale=1.0,
    )

    screw_rows: list[np.ndarray] = []
    empty_rows: list[np.ndarray] = []
    files = collect_images(train_root)
    if not files:
        raise SystemExit(f"No training images found under: {train_root}")

    skipped: list[dict] = []
    alignment_ok = 0
    print("=== Build S/E dual-class GOOD-only models v3 ===")
    print(f"Train GOOD:      {train_root}")
    print(f"Canonical:       {canonical_w}x{canonical_h}")
    print(f"S slots:         {[slot['id'] for slot in s_slots]}")
    print(f"E slots:         {[slot['id'] for slot in e_slots]}")
    print("Training:        S crops = real screw class; E crops = real empty class")
    print("Test NG used:    NO")
    print(f"Classifier:      dual-bank top-{top_k} cosine margin")
    print()

    for index, path in enumerate(files, 1):
        try:
            raw = read_image(path)
            alignment = align_to_reference(raw, reference, align_cfg)
            canonical = alignment.aligned
            for slot in s_slots:
                roi = roi_for_image(slot["roi"], config, canonical_w, canonical_h)
                screw_rows.append(appearance_descriptor(crop_roi(canonical, roi), descriptor_cfg))
            for slot in e_slots:
                roi = roi_for_image(slot["roi"], config, canonical_w, canonical_h)
                empty_rows.append(appearance_descriptor(crop_roi(canonical, roi), descriptor_cfg))
            alignment_ok += 1
            print(f"[{index}/{len(files)}] {path.name} -> OK | ecc={alignment.ecc_score:.4f}")
        except Exception as exc:
            skipped.append({"path": str(path), "error": str(exc)})
            print(f"[{index}/{len(files)}] {path.name} -> SKIP | {exc}")

    if len(screw_rows) < 4 or len(empty_rows) < 4:
        raise RuntimeError("Not enough aligned screw/empty samples to build S/E models.")

    screw_bank = np.stack(screw_rows).astype(np.float32)
    empty_bank = np.stack(empty_rows).astype(np.float32)
    s_threshold, e_threshold, calibration = _calibrate_thresholds(screw_bank, empty_bank, top_k)

    s_model = _save_model(
        output / "S",
        expected="screw",
        detector_name="S_presence",
        threshold=s_threshold,
        screw_bank=screw_bank,
        empty_bank=empty_bank,
        config=config,
        canonical_size=(canonical_w, canonical_h),
        descriptor_cfg=descriptor_cfg,
        top_k=top_k,
        images_seen=len(files),
        alignment_ok=alignment_ok,
        skipped=skipped,
        calibration_stats=calibration,
    )
    e_model = _save_model(
        output / "E",
        expected="empty",
        detector_name="E_empty",
        threshold=e_threshold,
        screw_bank=screw_bank,
        empty_bank=empty_bank,
        config=config,
        canonical_size=(canonical_w, canonical_h),
        descriptor_cfg=descriptor_cfg,
        top_k=top_k,
        images_seen=len(files),
        alignment_ok=alignment_ok,
        skipped=skipped,
        calibration_stats=calibration,
    )

    summary = {
        "schema_version": 3,
        "descriptor": "radial_presence_v3",
        "classifier": "dual_bank_topk_cosine",
        "images_seen": len(files),
        "alignment_ok": alignment_ok,
        "alignment_skipped": len(files) - alignment_ok,
        "screw_samples": int(screw_bank.shape[0]),
        "empty_samples": int(empty_bank.shape[0]),
        "S_threshold": float(s_model["classifier"]["threshold"]),
        "E_threshold": float(e_model["classifier"]["threshold"]),
        "calibration": calibration,
        "S_model": str(output / "S" / "model.json"),
        "E_model": str(output / "E" / "model.json"),
    }
    (output / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=== Build Summary ===")
    print(f"GOOD images:       {len(files)}")
    print(f"Alignment OK:      {alignment_ok}")
    print(f"Alignment skipped: {len(files) - alignment_ok}")
    print(f"Screw samples:     {screw_bank.shape[0]}")
    print(f"Empty samples:     {empty_bank.shape[0]}")
    print(f"Calibration:       {calibration['calibration']}")
    print(f"S threshold:       {s_threshold:.6f}")
    print(f"E threshold:       {e_threshold:.6f}")
    print(f"S model:           {output / 'S' / 'model.json'}")
    print(f"E model:           {output / 'E' / 'model.json'}")
    print(f"Summary:           {output / 'build_summary.json'}")


if __name__ == "__main__":
    main()
