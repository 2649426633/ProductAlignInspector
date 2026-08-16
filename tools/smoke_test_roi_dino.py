from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.anomaly.dinov2_adapter import DINOv2Adapter, DINOv2Config
from product_align_inspector.anomaly.roi_patchcore import select_regions
from product_align_inspector.io_utils import read_image, write_image
from product_align_inspector.roi import crop_roi, validate_roi


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test aligned fixed ROIs through local DINOv2 patch-token extraction.")
    parser.add_argument("--input", required=True, help="One full-resolution GOOD image")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--roi-id", action="append", default=[])
    parser.add_argument("--dino-repo", default="third_party/dinov2")
    parser.add_argument("--dino-weights", default="weights/dinov2_vits14_pretrain.pth")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", default="artifacts/roi_dino_smoke")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    regions = select_regions(config, args.roi_id or None)
    reference = read_image(args.reference)
    raw = read_image(args.input)

    print("=== ROI DINOv2 Smoke Test ===", flush=True)
    t0 = time.perf_counter()
    alignment = align_to_reference(raw, reference, ProductLocatorConfig())
    alignment_seconds = time.perf_counter() - t0
    aligned = alignment.aligned
    print(
        f"Alignment: {alignment.method}, ratio={alignment.feature_inlier_ratio:.1%}, "
        f"ECC={'-' if alignment.ecc_score is None else f'{alignment.ecc_score:.4f}'}, "
        f"time={alignment_seconds:.3f}s",
        flush=True,
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_image(out / "aligned.png", aligned)

    h, w = aligned.shape[:2]
    crops = []
    for region in regions:
        if not validate_roi(region.roi, w, h):
            raise RuntimeError(f"Invalid ROI {region.id}: {region.roi}")
        crop = crop_roi(aligned, region.roi)
        crops.append(crop)
        write_image(out / "crops" / f"{region.id}.png", crop)

    dino = DINOv2Adapter(
        device=args.device,
        config=DINOv2Config(
            image_size=args.image_size,
            repo_dir=args.dino_repo,
            weights_path=args.dino_weights,
        ),
        project_root=REPO_ROOT,
    )
    dino.load()

    t0 = time.perf_counter()
    tokens = dino.patch_tokens_batch(crops)
    dino_seconds = time.perf_counter() - t0
    norms = np.linalg.norm(tokens, axis=-1)

    print(f"ROI IDs:       {[r.id for r in regions]}", flush=True)
    print(f"Token shape:   {tokens.shape}", flush=True)
    print(f"Patch grid:    {dino.patch_grid}x{dino.patch_grid}", flush=True)
    print(f"Feature dim:   {tokens.shape[-1]}", flush=True)
    print(f"Mean L2 norm:  {float(norms.mean()):.6f}", flush=True)
    print(f"DINO time:     {dino_seconds:.3f}s for {len(regions)} ROI(s)", flush=True)
    print(f"Crops:         {out / 'crops'}", flush=True)

    expected_shape = (len(regions), dino.patch_grid * dino.patch_grid, 384)
    if tuple(tokens.shape) != expected_shape:
        raise RuntimeError(f"Unexpected token shape {tokens.shape}; expected {expected_shape}")
    if not np.isfinite(tokens).all():
        raise RuntimeError("DINO tokens contain non-finite values")
    if abs(float(norms.mean()) - 1.0) > 1e-3:
        raise RuntimeError(f"Patch tokens are not L2-normalized; mean norm={norms.mean():.6f}")

    print("SMOKE TEST OK.", flush=True)


if __name__ == "__main__":
    main()
