from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference, make_overlay
from product_align_inspector.io_utils import read_image, write_image


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose the current fixed-camera alignment path for one or more images.")
    p.add_argument("--reference", required=True)
    p.add_argument("--input", action="append", required=True, help="May be supplied multiple times")
    p.add_argument("--output", default="artifacts/alignment_diagnose")
    p.add_argument("--foreground-threshold", type=int, default=238)
    args = p.parse_args()

    reference = read_image(Path(args.reference))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = ProductLocatorConfig(foreground_threshold=args.foreground_threshold)

    rows = []
    print("=== Fixed-camera alignment diagnostic ===")
    for input_text in args.input:
        path = Path(input_text)
        stem_out = output / path.stem
        stem_out.mkdir(parents=True, exist_ok=True)
        try:
            result = align_to_reference(read_image(path), reference, cfg)
            write_image(stem_out / "aligned.png", result.aligned)
            write_image(stem_out / "overlay.png", make_overlay(reference, result.aligned))
            row = {
                "input": str(path),
                "status": "ALIGN_OK",
                "method": result.method,
                "ecc": result.ecc_score,
                "feature_matches": result.feature_matches,
                "feature_inliers": result.feature_inliers,
                "feature_inlier_ratio": result.feature_inlier_ratio,
                "fallback_rotation_deg": result.fallback_rotation_deg,
                "overlay": str(stem_out / "overlay.png"),
                "error": "",
            }
            print(
                f"{path.name} -> ALIGN_OK | method={result.method} | "
                f"ecc={result.ecc_score} | inlier_ratio={result.feature_inlier_ratio:.3f}"
            )
        except Exception as exc:
            row = {
                "input": str(path),
                "status": "RETRY",
                "method": "",
                "ecc": None,
                "feature_matches": 0,
                "feature_inliers": 0,
                "feature_inlier_ratio": 0.0,
                "fallback_rotation_deg": None,
                "overlay": "",
                "error": str(exc),
            }
            print(f"{path.name} -> RETRY: {exc}")
        rows.append(row)

    summary = output / "summary.json"
    summary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
