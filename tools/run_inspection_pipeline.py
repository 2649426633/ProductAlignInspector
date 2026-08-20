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
from product_align_inspector.roi import config_reference_size, enabled_slots, roi_for_image, validate_roi
from product_align_inspector.se_presence import (
    EEmptyDetector,
    SPresenceDetector,
    SharedSemanticPresenceModel,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_images(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_config(config: dict, reference_width: int, reference_height: int) -> str:
    """Validate ROI mapping into the current canonical/reference image."""
    source_size = config_reference_size(config)
    if source_size is None:
        note = f"ROI size not declared; coordinates treated as {reference_width}x{reference_height}"
    elif source_size == (reference_width, reference_height):
        note = f"ROI coordinates already canonical: {source_size[0]}x{source_size[1]}"
    else:
        sx = reference_width / float(source_size[0])
        sy = reference_height / float(source_size[1])
        note = (
            f"fixed ROI coordinate mapping {source_size[0]}x{source_size[1]} -> "
            f"{reference_width}x{reference_height} (sx={sx:.6f}, sy={sy:.6f})"
        )

    for slot in enabled_slots(config):
        roi = roi_for_image(slot["roi"], config, reference_width, reference_height)
        if not validate_roi(roi, reference_width, reference_height):
            raise SystemExit(
                f"ROI OUT OF CANONICAL BOUNDS for {slot.get('id', '?')}: "
                f"source={slot.get('roi')} mapped={roi}; reference={reference_width}x{reference_height}"
            )
    return note


def _resolve_semantic_models(args: argparse.Namespace) -> tuple[Path | None, Path | None, Path | None]:
    """Resolve new shared model CLI while retaining a narrow compatibility path."""

    if args.se_model:
        shared = Path(args.se_model).resolve()
        if args.s_model or args.e_model:
            raise SystemExit("Use --se-model alone; do not combine it with --s-model/--e-model.")
        return shared, shared, shared

    s_model = None if not args.s_model else Path(args.s_model).resolve()
    e_model = None if not args.e_model else Path(args.e_model).resolve()
    if s_model is not None and e_model is not None and s_model == e_model:
        return s_model, s_model, s_model
    return None, s_model, e_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RAW -> rigid canonical reference -> shared semantic screw/empty CNN -> "
            "11 per-slot probability thresholds -> inverse-map results to RAW. "
            "Per-frame geometry is rotation + X/Y translation only, scale=1."
        )
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument(
        "--reference",
        required=True,
        help="Canonical detection reference, e.g. artifacts/reference/brunei_preview_reference.png",
    )
    parser.add_argument(
        "--config",
        help="ROI JSON, e.g. configs/brunei_preview_template.json",
    )
    parser.add_argument(
        "--se-model",
        help=(
            "Shared semantic S/E model directory containing model.json + presence_classifier.onnx. "
            "Enables both S_presence and E_empty."
        ),
    )
    parser.add_argument(
        "--s-model",
        help="Compatibility option: semantic model directory for S only. Prefer --se-model.",
    )
    parser.add_argument(
        "--e-model",
        help="Compatibility option: semantic model directory for E only. Prefer --se-model.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--ecc-accept", type=float, default=0.70)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.10)
    parser.add_argument("--no-save-canonical", action="store_true")
    parser.add_argument("--no-save-overlay", action="store_true")
    parser.add_argument("--no-save-restored", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    reference_path = Path(args.reference).resolve()
    reference = read_image(reference_path)
    ref_h, ref_w = reference.shape[:2]

    shared_dir, s_model_dir, e_model_dir = _resolve_semantic_models(args)
    any_model = shared_dir or s_model_dir or e_model_dir

    config = None
    config_note = "OFF"
    if args.config:
        config_path = Path(args.config).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config_note = validate_config(config, ref_w, ref_h)
    elif any_model:
        raise SystemExit("--config is required when S/E semantic detection is enabled.")

    align_cfg = ProductLocatorConfig(
        ecc_accept_score=float(args.ecc_accept),
        feature_min_inlier_ratio=float(args.min_inlier_ratio),
        canonical_scale=1.0,
    )

    detectors = []
    if shared_dir is not None:
        shared_model = SharedSemanticPresenceModel(shared_dir)
        detectors.append(
            SPresenceDetector(shared_dir, config or {}, shared_model=shared_model)
        )
        detectors.append(
            EEmptyDetector(shared_dir, config or {}, shared_model=shared_model)
        )
    else:
        if s_model_dir is not None:
            detectors.append(SPresenceDetector(s_model_dir, config or {}))
        if e_model_dir is not None:
            detectors.append(EEmptyDetector(e_model_dir, config or {}))

    pipeline = InspectionPipeline(
        reference=reference,
        output_dir=output,
        align_cfg=align_cfg,
        detectors=detectors,
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
    detector_counts: dict[str, Counter[str]] = {
        getattr(detector, "name", detector.__class__.__name__): Counter() for detector in detectors
    }

    print("=== ProductAlignInspector Main Framework ===")
    print(f"Input root:       {input_root}")
    print(f"Canonical ref:    {reference_path}")
    print(f"Canonical size:   {ref_w}x{ref_h}")
    print(f"ROI config:       {Path(args.config).resolve() if args.config else 'OFF'}")
    if args.config:
        print(f"ROI config check: {config_note}")
    print(f"Images:           {len(files)}")
    print("Geometry:         rotation + X/Y translation ONLY")
    print("Scale:            FIXED 1.00000")
    if shared_dir is not None:
        print(f"Shared S/E CNN:    {shared_dir}")
        print("S/E score:         P(screw)")
        print("Thresholds:        11 independent slot thresholds")
    else:
        print(f"S detector:       {s_model_dir if s_model_dir else 'OFF'}")
        print(f"E detector:       {e_model_dir if e_model_dir else 'OFF'}")
    print("Flow: RAW -> canonical -> semantic S/E -> inverse-map result to RAW")
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
            detector_summary = []
            for detection in record.detections:
                name = str(detection.get("detector", "?"))
                status = str(detection.get("status", "?"))
                detector_counts.setdefault(name, Counter())[status] += 1
                detector_summary.append(f"{name}={status}")

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
                    "S_status": next(
                        (
                            d.get("status", "")
                            for d in record.detections
                            if d.get("detector") == "S_presence"
                        ),
                        "",
                    ),
                    "E_status": next(
                        (
                            d.get("status", "")
                            for d in record.detections
                            if d.get("detector") == "E_empty"
                        ),
                        "",
                    ),
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
                detect_text = " | ".join(detector_summary) if detector_summary else "detectors=OFF"
                print(
                    f"[{index}/{len(files)}] {rel.as_posix()} -> {record.final_status} | "
                    f"ecc={ecc} | {detect_text} | {record.total_time_sec:.3f}s"
                )
            else:
                print(f"[{index}/{len(files)}] {rel.as_posix()} -> {record.final_status} | {record.error}")

    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row["alignment_status"] == "OK"]
    summary = {
        "images": len(files),
        "counts": dict(counts),
        "alignment_ok": len(ok_rows),
        "alignment_skipped": counts.get("SKIP_ALIGNMENT", 0),
        "canonical_reference": str(reference_path),
        "canonical_size": [ref_w, ref_h],
        "roi_config": str(Path(args.config).resolve()) if args.config else None,
        "roi_mapping": config_note,
        "shared_se_model": str(shared_dir) if shared_dir else None,
        "score_semantics": "P(screw)" if any_model else None,
        "detectors": {
            name: dict(counter) for name, counter in detector_counts.items()
        },
        "logs": {"jsonl": str(jsonl_path), "csv": str(csv_path)},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=== Summary ===")
    print(f"Images:            {len(files)}")
    print(f"Alignment OK:      {summary['alignment_ok']}")
    print(f"Alignment skipped: {summary['alignment_skipped']}")
    for name, counter in detector_counts.items():
        print(f"{name}: {dict(counter)}")
    print(f"Canonical:         {output / 'canonical'}")
    print(f"Overlays:          {output / 'overlays'}")
    print(f"Restored:          {output / 'restored'}")
    print(f"CSV log:           {csv_path}")
    print(f"Summary:           {summary_path}")


if __name__ == "__main__":
    main()
