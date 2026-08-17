from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.anomaly.dinov2_adapter import DINOv2Adapter, DINOv2Config
from product_align_inspector.anomaly.roi_patchcore import load_roi_model, read_model_manifest, score_patch_tokens
from product_align_inspector.io_utils import read_image, write_image
from product_align_inspector.roi import crop_roi, validate_roi


def to_gray32(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0


def local_translation_refine(reference_crop: np.ndarray, test_crop: np.ndarray) -> tuple[np.ndarray, float | None, float, float]:
    h, w = reference_crop.shape[:2]
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
    try:
        ecc, warp = cv2.findTransformECC(
            to_gray32(reference_crop),
            to_gray32(test_crop),
            warp,
            cv2.MOTION_TRANSLATION,
            criteria,
            None,
            3,
        )
        refined = cv2.warpAffine(
            test_crop,
            warp,
            (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT101,
        )
        return refined, float(ecc), float(warp[0, 2]), float(warp[1, 2])
    except cv2.error:
        return test_crop.copy(), None, 0.0, 0.0


def label_panel(image: np.ndarray, text: str) -> np.ndarray:
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    bar_h = 30
    out = cv2.copyMakeBorder(canvas, bar_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Compare global ROI crops with local translation refinement.")
    p.add_argument("--good", required=True, help="Known GOOD full-resolution image")
    p.add_argument("--test", required=True, help="Test/NG full-resolution image")
    p.add_argument("--model-dir", default="artifacts/roi_dino_full")
    p.add_argument("--reference", required=True)
    p.add_argument("--dino-repo")
    p.add_argument("--dino-weights")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="artifacts/diagnose_roi_local_refinement")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    manifest = read_model_manifest(model_dir)
    models = [load_roi_model(model_dir, str(item["id"])) for item in manifest.get("rois", [])]
    if not models:
        raise RuntimeError("No ROI models found in model.json")

    reference = read_image(Path(args.reference))
    align_meta = manifest.get("alignment", {})
    align_cfg = ProductLocatorConfig(foreground_threshold=int(align_meta.get("foreground_threshold", 238)))

    good_raw = read_image(Path(args.good))
    test_raw = read_image(Path(args.test))
    good_alignment = align_to_reference(good_raw, reference, align_cfg)
    test_alignment = align_to_reference(test_raw, reference, align_cfg)

    print("=== Global alignment ===")
    print(f"GOOD: method={good_alignment.method}, ratio={good_alignment.feature_inlier_ratio:.1%}, ECC={good_alignment.ecc_score}")
    print(f"TEST: method={test_alignment.method}, ratio={test_alignment.feature_inlier_ratio:.1%}, ECC={test_alignment.ecc_score}")

    dino_meta = manifest["dino"]
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

    out = Path(args.output)
    compare_dir = out / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)

    good_aligned = good_alignment.aligned
    test_aligned = test_alignment.aligned
    h, w = test_aligned.shape[:2]

    good_crops: list[np.ndarray] = []
    test_crops: list[np.ndarray] = []
    refined_crops: list[np.ndarray] = []
    local_meta: list[tuple[float | None, float, float]] = []

    for model in models:
        if not validate_roi(model.roi, w, h):
            raise RuntimeError(f"Invalid ROI {model.roi_id}: {model.roi}")
        gc = crop_roi(good_aligned, model.roi)
        tc = crop_roi(test_aligned, model.roi)
        rc, ecc, dx, dy = local_translation_refine(gc, tc)
        good_crops.append(gc)
        test_crops.append(tc)
        refined_crops.append(rc)
        local_meta.append((ecc, dx, dy))

    tokens = dino.patch_tokens_batch(good_crops + test_crops + refined_crops)
    n = len(models)
    good_tokens = tokens[:n]
    test_tokens = tokens[n:2*n]
    refined_tokens = tokens[2*n:]

    rows: list[dict[str, object]] = []
    print("\n=== ROI local-refinement diagnostic ===")
    print(f"{'ROI':<10} {'GOOD':>9} {'TEST':>9} {'REFINED':>9} {'THR':>9} {'ECC':>8} {'dx':>7} {'dy':>7} {'DROP':>8}")
    print("-" * 90)

    for model, gc, tc, rc, gt, tt, rt, meta in zip(models, good_crops, test_crops, refined_crops, good_tokens, test_tokens, refined_tokens, local_meta):
        good_score, _, _ = score_patch_tokens(gt, model.memory, patch_grid=model.patch_grid, top_fraction=model.score_top_fraction)
        test_score, _, _ = score_patch_tokens(tt, model.memory, patch_grid=model.patch_grid, top_fraction=model.score_top_fraction)
        refined_score, _, _ = score_patch_tokens(rt, model.memory, patch_grid=model.patch_grid, top_fraction=model.score_top_fraction)
        ecc, dx, dy = meta
        threshold = None if model.threshold is None else float(model.threshold)
        drop = float(test_score - refined_score)
        rows.append({
            "roi_id": model.roi_id,
            "good_score": good_score,
            "test_score": test_score,
            "refined_score": refined_score,
            "threshold": threshold,
            "local_ecc": "" if ecc is None else ecc,
            "dx": dx,
            "dy": dy,
            "score_drop": drop,
        })
        ecc_text = "-" if ecc is None else f"{ecc:.4f}"
        thr_text = "-" if threshold is None else f"{threshold:.4f}"
        print(f"{model.roi_id:<10} {good_score:>9.4f} {test_score:>9.4f} {refined_score:>9.4f} {thr_text:>9} {ecc_text:>8} {dx:>7.2f} {dy:>7.2f} {drop:>8.4f}")

        panel = cv2.hconcat([
            label_panel(gc, f"GOOD {good_score:.3f}"),
            label_panel(tc, f"TEST {test_score:.3f}"),
            label_panel(rc, f"LOCAL {refined_score:.3f} dx={dx:.1f} dy={dy:.1f}"),
        ])
        write_image(compare_dir / f"{model.roi_id}.png", panel)

    csv_path = out / "local_refinement.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "good": str(args.good),
        "test": str(args.test),
        "good_alignment": {
            "method": good_alignment.method,
            "feature_inlier_ratio": good_alignment.feature_inlier_ratio,
            "ecc_score": good_alignment.ecc_score,
        },
        "test_alignment": {
            "method": test_alignment.method,
            "feature_inlier_ratio": test_alignment.feature_inlier_ratio,
            "ecc_score": test_alignment.ecc_score,
        },
        "rows": rows,
    }
    (out / "local_refinement.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nCSV:     {csv_path}")
    print(f"Compare: {compare_dir}")


if __name__ == "__main__":
    main()
