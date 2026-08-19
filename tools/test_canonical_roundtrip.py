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
from product_align_inspector.canonical_frame import CanonicalFrameMapper
from product_align_inspector.io_utils import read_image, write_image, write_json


def draw_config_on_original(
    raw: np.ndarray,
    mapper: CanonicalFrameMapper,
    input_to_raw_reference: np.ndarray,
    config: dict,
) -> np.ndarray:
    out = raw.copy()
    for slot in config.get("screw_slots", []):
        if not bool(slot.get("enabled", True)):
            continue
        roi = slot.get("roi")
        if roi is None:
            continue
        expected = str(slot.get("expected", ""))
        color = (0, 190, 0) if expected == "screw" else (0, 150, 220)
        polygon = np.round(
            mapper.canonical_roi_on_raw(roi, input_to_raw_reference)
        ).astype(np.int32)
        cv2.polylines(out, [polygon], True, color, 4, cv2.LINE_AA)
        label = f"{slot.get('id', '?')}:{expected}"
        x, y = polygon[0].tolist()
        cv2.putText(
            out,
            label,
            (int(x), max(30, int(y) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test the desired coordinate chain: RAW input -> rigid RAW reference -> "
            "existing preview canonical -> map canonical ROI back onto the original RAW image."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--raw-reference", required=True)
    parser.add_argument("--canonical-reference", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--foreground-threshold", type=int, default=238)
    parser.add_argument("--ecc-accept", type=float, default=0.70)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    raw = read_image(input_path)
    raw_reference = read_image(Path(args.raw_reference))
    canonical_reference = read_image(Path(args.canonical_reference))
    mapper = CanonicalFrameMapper.from_json(args.calibration)

    cfg = ProductLocatorConfig(
        foreground_threshold=args.foreground_threshold,
        ecc_accept_score=args.ecc_accept,
    )
    alignment = align_to_reference(raw, raw_reference, cfg)
    if alignment.feature_matrix is None:
        raise RuntimeError("Rigid alignment did not expose input->RAW-reference matrix.")

    canonical, raw_to_canonical = mapper.warp_to_canonical(
        raw,
        alignment.feature_matrix,
    )
    if canonical.shape[:2] != canonical_reference.shape[:2]:
        raise RuntimeError(
            f"Canonical size mismatch: mapped={canonical.shape[1]}x{canonical.shape[0]}, "
            f"template={canonical_reference.shape[1]}x{canonical_reference.shape[0]}"
        )

    overlay = cv2.addWeighted(canonical_reference, 0.5, canonical, 0.5, 0.0)
    write_image(out / "canonical.png", canonical)
    write_image(out / "canonical_overlay.png", overlay)

    restored = raw.copy()
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        restored = draw_config_on_original(
            raw,
            mapper,
            alignment.feature_matrix,
            config,
        )
    write_image(out / "original_with_mapped_rois.jpg", restored)

    report = {
        "input": str(input_path),
        "raw_reference": str(Path(args.raw_reference).resolve()),
        "canonical_reference": str(Path(args.canonical_reference).resolve()),
        "calibration": str(Path(args.calibration).resolve()),
        "alignment": alignment.to_dict(),
        "raw_input_to_canonical": raw_to_canonical.tolist(),
        "canonical_to_raw_input": cv2.invertAffineTransform(raw_to_canonical).tolist(),
        "outputs": {
            "canonical": str(out / "canonical.png"),
            "canonical_overlay": str(out / "canonical_overlay.png"),
            "original_with_mapped_rois": str(out / "original_with_mapped_rois.jpg"),
        },
    }
    write_json(out / "roundtrip.json", report)

    print("=== Canonical round-trip test ===")
    print(f"Input:             {input_path}")
    print(f"Rigid alignment:   {alignment.method}")
    print(f"ECC:               {alignment.ecc_score:.4f}")
    print(f"Canonical overlay: {out / 'canonical_overlay.png'}")
    print(f"Mapped to RAW:     {out / 'original_with_mapped_rois.jpg'}")
    print(f"Report:            {out / 'roundtrip.json'}")


if __name__ == "__main__":
    main()
