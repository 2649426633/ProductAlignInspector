from __future__ import annotations

import argparse
from pathlib import Path

from product_align_inspector.alignment import ProductLocatorConfig, build_foreground_mask, coarse_align
from product_align_inspector.io_utils import read_image, write_image, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a canonical GOOD reference image.")
    parser.add_argument("--input", required=True, help="GOOD product image")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--threshold", type=int, default=238, help="Foreground grayscale threshold")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cfg = ProductLocatorConfig(foreground_threshold=args.threshold)
    image = read_image(args.input)
    aligned, location, _ = coarse_align(image, cfg)
    aligned_mask = build_foreground_mask(aligned, cfg)

    write_image(out / "reference_aligned.png", aligned)
    write_image(out / "reference_mask.png", aligned_mask)
    write_json(
        out / "reference_meta.json",
        {
            "source": str(Path(args.input)),
            "reference_shape": list(aligned.shape),
            "locator": location.to_dict(),
            "config": {
                "foreground_threshold": cfg.foreground_threshold,
                "border_margin_ratio": cfg.border_margin_ratio,
                "min_component_area_ratio": cfg.min_component_area_ratio,
                "close_kernel_ratio": cfg.close_kernel_ratio,
                "crop_padding_ratio": cfg.crop_padding_ratio,
            },
        },
    )
    print(f"Reference created: {out / 'reference_aligned.png'}")
    print(f"Reference size: {aligned.shape[1]} x {aligned.shape[0]}")
    print(f"Detected angle: {location.angle_deg:.3f} deg")


if __name__ == "__main__":
    main()
