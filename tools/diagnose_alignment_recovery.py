from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.alignment import (
    ProductLocatorConfig,
    _ecc_refine,
    _feature_align,
    make_overlay,
)
from product_align_inspector.io_utils import read_image, write_image


def try_preset(
    image,
    reference,
    name: str,
    cfg: ProductLocatorConfig,
    ecc_accept: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "preset": name,
        "ok": False,
        "matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "ecc": None,
        "ecc_accept": float(ecc_accept),
        "accepted": False,
        "error": "",
    }
    try:
        aligned, _location, matrix, matches, inliers, ratio = _feature_align(image, reference, cfg)
        refined, ecc, ecc_matrix = _ecc_refine(
            reference,
            aligned,
            iterations=300,
            epsilon=1e-6,
            motion=cv2.MOTION_AFFINE,
        )
        # Weak/relaxed SIFT is never trusted alone. ECC must independently
        # confirm the whole aligned image before the recovery is accepted.
        accepted = ecc is not None and ecc >= ecc_accept
        final = refined if accepted else aligned
        row.update(
            {
                "ok": True,
                "matches": int(matches),
                "inliers": int(inliers),
                "inlier_ratio": float(ratio),
                "ecc": None if ecc is None else float(ecc),
                "accepted": bool(accepted),
                "feature_matrix": matrix.tolist(),
                "ecc_matrix": None if ecc_matrix is None else ecc_matrix.tolist(),
                "aligned": final,
            }
        )
    except Exception as exc:
        row["error"] = str(exc)
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="Try staged SIFT/ECC recovery on alignment failures.")
    p.add_argument("--reference", required=True)
    p.add_argument("--input", action="append", required=True, help="May be supplied multiple times")
    p.add_argument("--output", default="artifacts/alignment_recovery")
    args = p.parse_args()

    reference = read_image(Path(args.reference))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    base = ProductLocatorConfig()
    presets = [
        ("strict", base, 0.80),
        (
            "detail",
            replace(
                base,
                feature_max_dim=2600,
                feature_nfeatures=8000,
                feature_ratio_test=0.76,
                feature_min_matches=10,
                feature_min_inliers=8,
                feature_min_inlier_ratio=0.20,
                feature_ransac_threshold_px=6.0,
            ),
            0.80,
        ),
        (
            "relaxed",
            replace(
                base,
                feature_max_dim=3200,
                feature_nfeatures=12000,
                feature_ratio_test=0.80,
                feature_min_matches=8,
                feature_min_inliers=6,
                feature_min_inlier_ratio=0.15,
                feature_ransac_threshold_px=8.0,
            ),
            0.80,
        ),
        (
            "ultra",
            replace(
                base,
                feature_max_dim=3600,
                feature_nfeatures=16000,
                feature_ratio_test=0.82,
                feature_min_matches=8,
                feature_min_inliers=6,
                feature_min_inlier_ratio=0.12,
                feature_ransac_threshold_px=10.0,
            ),
            0.85,
        ),
    ]

    summary: list[dict[str, object]] = []
    print("=== Alignment Recovery Diagnostic ===", flush=True)
    print("Acceptance: strict/detail/relaxed ECC >= 0.80; ultra ECC >= 0.85", flush=True)

    for input_text in args.input:
        path = Path(input_text)
        image = read_image(path)
        print(f"\nInput: {path}", flush=True)
        print(
            f"{'Preset':<10} {'Matches':>8} {'Inliers':>8} {'Ratio':>9} "
            f"{'ECC':>9} {'Need':>7} {'Accepted':>10}",
            flush=True,
        )
        print("-" * 70, flush=True)

        image_rows: list[dict[str, object]] = []
        best: dict[str, object] | None = None
        for name, cfg, ecc_accept in presets:
            row = try_preset(image, reference, name, cfg, ecc_accept)
            image_rows.append(row)
            if row["ok"]:
                ecc_text = "-" if row["ecc"] is None else f"{float(row['ecc']):.4f}"
                print(
                    f"{name:<10} {int(row['matches']):>8} {int(row['inliers']):>8} "
                    f"{float(row['inlier_ratio']):>8.1%} {ecc_text:>9} "
                    f"{float(row['ecc_accept']):>7.2f} {str(row['accepted']):>10}",
                    flush=True,
                )
                if row["accepted"] and (best is None or float(row["ecc"]) > float(best["ecc"])):
                    best = row
            else:
                print(f"{name:<10} FAILED: {row['error']}", flush=True)

        stem_out = out / path.stem
        stem_out.mkdir(parents=True, exist_ok=True)
        if best is not None:
            aligned = best.pop("aligned")
            write_image(stem_out / "aligned.png", aligned)
            write_image(stem_out / "overlay.png", make_overlay(reference, aligned))
            print(
                f"RECOVERED: preset={best['preset']} ECC={float(best['ecc']):.4f} -> {stem_out / 'overlay.png'}",
                flush=True,
            )
        else:
            for row in image_rows:
                row.pop("aligned", None)
            print("NOT RECOVERED: no preset passed its ECC acceptance threshold", flush=True)

        for row in image_rows:
            row.pop("aligned", None)
        summary.append({"input": str(path), "presets": image_rows, "recovered": best is not None})

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
