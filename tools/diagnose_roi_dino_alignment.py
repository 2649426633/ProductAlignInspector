from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference, make_overlay
from product_align_inspector.anomaly.dinov2_adapter import DINOv2Adapter, DINOv2Config
from product_align_inspector.anomaly.roi_patchcore import load_roi_model, read_model_manifest, score_patch_tokens
from product_align_inspector.io_utils import read_image, write_image
from product_align_inspector.roi import crop_roi, validate_roi


def save_heatmap(crop: np.ndarray, anomaly_map: np.ndarray, threshold: float | None, path: Path) -> None:
    h, w = crop.shape[:2]
    up = cv2.resize(anomaly_map.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    scale = float(threshold) if threshold is not None and threshold > 1e-12 else float(np.max(up))
    scale = max(scale, 1e-8)
    norm = np.clip(up / scale, 0.0, 1.0)
    heat = cv2.applyColorMap(np.round(norm * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(crop, 0.58, heat, 0.42, 0.0)
    write_image(path, overlay)


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose alignment and ROI DINO/PatchCore scores for one image.")
    p.add_argument("--input", required=True)
    p.add_argument("--model-dir", default="artifacts/roi_dino_patchcore")
    p.add_argument("--reference", required=True)
    p.add_argument("--dino-repo")
    p.add_argument("--dino-weights")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    input_path = Path(args.input)
    model_dir = Path(args.model_dir)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    manifest = read_model_manifest(model_dir)
    dino_meta = manifest["dino"]
    align_meta = manifest.get("alignment", {})
    models = [load_roi_model(model_dir, str(item["id"])) for item in manifest.get("rois", [])]
    if not models:
        raise RuntimeError("No ROI models found in model.json")

    reference = read_image(Path(args.reference))
    raw = read_image(input_path)
    align_cfg = ProductLocatorConfig(foreground_threshold=int(align_meta.get("foreground_threshold", 238)))
    alignment = align_to_reference(raw, reference, align_cfg)

    print("=== Alignment Diagnostic ===")
    print(f"Input:   {input_path}")
    print(f"Method:  {alignment.method}")
    print(f"Matches: {alignment.feature_matches}")
    print(f"Inliers: {alignment.feature_inliers}")
    print(f"Ratio:   {alignment.feature_inlier_ratio:.1%}")
    print(f"ECC:     {'-' if alignment.ecc_score is None else f'{alignment.ecc_score:.6f}'}")

    write_image(out / "aligned.png", alignment.aligned)
    write_image(out / "overlay.png", make_overlay(reference, alignment.aligned))

    h, w = alignment.aligned.shape[:2]
    crops = []
    for model in models:
        if not validate_roi(model.roi, w, h):
            raise RuntimeError(f"Invalid ROI {model.roi_id}: {model.roi}")
        crop = crop_roi(alignment.aligned, model.roi)
        crops.append(crop)
        write_image(out / "crops" / f"{model.roi_id}.png", crop)

    dino_cfg = DINOv2Config(
        model_name=str(dino_meta.get("model_name", "dinov2_vits14")),
        image_size=int(dino_meta.get("image_size", 224)),
        embedding_dim=int(dino_meta.get("embedding_dim", 384)),
        repo_dir=str(args.dino_repo or dino_meta.get("repo_dir", "third_party/dinov2")),
        weights_path=str(args.dino_weights or dino_meta.get("weights_path", "weights/dinov2_vits14_pretrain.pth")),
        pad_value=int(dino_meta.get("preprocess", {}).get("pad_value", 255)),
    )
    dino = DINOv2Adapter(device=args.device, config=dino_cfg, project_root=REPO_ROOT)
    dino.load()
    tokens_batch = dino.patch_tokens_batch(crops)

    results = []
    print("\n=== ROI Scores ===")
    print(f"{'ROI':<10} {'Score':>10} {'Threshold':>10} {'MaxPatch':>10} {'Ratio':>8}")
    print("-" * 54)
    for model, crop, tokens in zip(models, crops, tokens_batch):
        score, anomaly_map, stats = score_patch_tokens(
            tokens,
            model.memory,
            patch_grid=model.patch_grid,
            top_fraction=model.score_top_fraction,
        )
        threshold = model.threshold
        score_ratio = None if threshold is None or threshold <= 1e-12 else score / threshold
        print(
            f"{model.roi_id:<10} {score:>10.6f} "
            f"{('-' if threshold is None else f'{threshold:.6f}'):>10} "
            f"{stats['max']:>10.6f} "
            f"{('-' if score_ratio is None else f'{score_ratio:.2f}x'):>8}"
        )
        save_heatmap(crop, anomaly_map, threshold, out / "heatmaps" / f"{model.roi_id}.png")
        results.append({
            "id": model.roi_id,
            "score": float(score),
            "threshold": threshold,
            "score_over_threshold": score_ratio,
            "patch_stats": stats,
        })

    payload = {
        "input": str(input_path),
        "alignment": alignment.to_dict(),
        "rois": results,
        "output": str(out),
    }
    (out / "diagnostic.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nOverlay: {out / 'overlay.png'}")
    print(f"Crops:   {out / 'crops'}")
    print(f"Heatmaps:{out / 'heatmaps'}")


if __name__ == "__main__":
    main()
