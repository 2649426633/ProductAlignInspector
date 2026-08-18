from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild model.json from existing ROI NPZ memory banks without retraining.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--config", default="configs/brunei.json")
    p.add_argument("--dino-repo", required=True)
    p.add_argument("--dino-weights", required=True)
    p.add_argument("--foreground-threshold", type=int, default=238)
    p.add_argument("--min-inlier-ratio", type=float, default=0.25)
    args = p.parse_args()

    model_dir = Path(args.model_dir).resolve()
    banks_dir = model_dir / "banks"
    reference = Path(args.reference).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    if not banks_dir.is_dir():
        raise SystemExit(f"Banks directory not found: {banks_dir}")
    if not reference.is_file():
        raise SystemExit(f"Reference image not found: {reference}")

    bank_files = sorted(banks_dir.glob("*.npz"))
    if not bank_files:
        raise SystemExit(f"No NPZ banks found in: {banks_dir}")

    config = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    rois = []
    patch_grids = set()
    feature_dims = set()
    score_fractions = set()

    for path in bank_files:
        with np.load(path, allow_pickle=False) as data:
            memory = np.asarray(data["memory"], dtype=np.float32)
            roi = [int(v) for v in data["roi"].tolist()]
            threshold_value = float(data["threshold"][0])
            calibration_scores = [float(v) for v in data["calibration_scores"].tolist()]
            score_top_fraction = float(data["score_top_fraction"][0])
            patch_grid = int(data["patch_grid"][0])
            feature_dim = int(data["feature_dim"][0])
            source_group = str(data["source_group"][0])

        patch_grids.add(patch_grid)
        feature_dims.add(feature_dim)
        score_fractions.add(round(score_top_fraction, 8))

        rois.append({
            "id": path.stem,
            "roi": roi,
            "source_group": source_group,
            "raw_patch_features": None,
            "memory_features": int(memory.shape[0]),
            "memory_mb_float32": float(memory.nbytes / (1024 * 1024)),
            "threshold": None if np.isnan(threshold_value) else threshold_value,
            "calibration_scores": calibration_scores,
            "calibration_detail": [],
            "bank_file": f"banks/{path.name}",
            "coreset_seconds": None,
        })

    if len(patch_grids) != 1:
        raise SystemExit(f"Inconsistent patch_grid values in banks: {sorted(patch_grids)}")
    if len(feature_dims) != 1:
        raise SystemExit(f"Inconsistent feature_dim values in banks: {sorted(feature_dims)}")

    patch_grid = next(iter(patch_grids))
    feature_dim = next(iter(feature_dims))
    score_top_fraction = next(iter(score_fractions)) if len(score_fractions) == 1 else 0.05
    image_size = patch_grid * 14

    payload = {
        "schema_version": 1,
        "model_type": "roi_dinov2_patchcore",
        "product": config.get("product", "brunei"),
        "reference_image": str(reference),
        "config_file": str(config_path),
        "training_data": {
            "recovered_manifest": True,
            "note": "model.json reconstructed from existing NPZ banks; memory banks were not retrained"
        },
        "alignment": {
            "min_inlier_ratio": float(args.min_inlier_ratio),
            "foreground_threshold": int(args.foreground_threshold),
        },
        "dino": {
            "model_name": "dinov2_vits14",
            "image_size": image_size,
            "patch_size": 14,
            "patch_grid": patch_grid,
            "embedding_dim": feature_dim,
            "repo_dir": str(Path(args.dino_repo).resolve()),
            "weights_path": str(Path(args.dino_weights).resolve()),
            "preprocess": {
                "pad_to_square": True,
                "pad_value": 255,
                "resize": [image_size, image_size],
                "interpolation": "OpenCV INTER_LINEAR",
                "color_order": "RGB",
                "scale": "uint8 / 255.0",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "layout": "NCHW",
            },
        },
        "patchcore": {
            "metric": "cosine_distance_on_l2_normalized_patch_tokens",
            "score_top_fraction": score_top_fraction,
            "threshold_rule": "stored_per_roi_threshold",
            "manifest_recovered": True,
        },
        "rois": rois,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    output = model_dir / "model.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== ROI model manifest repaired ===")
    print(f"Banks found: {len(rois)}")
    print(f"Patch grid:  {patch_grid}x{patch_grid}")
    print(f"Feature dim: {feature_dim}")
    print(f"Model JSON:  {output}")
    print("Memory banks were NOT retrained or modified.")


if __name__ == "__main__":
    main()
