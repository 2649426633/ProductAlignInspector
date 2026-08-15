from __future__ import annotations

import argparse
from pathlib import Path

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference, make_overlay
from product_align_inspector.io_utils import read_image, write_image, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Align one product image to a canonical reference.")
    parser.add_argument("--input", required=True, help="Input image")
    parser.add_argument("--reference", required=True, help="reference_aligned.png")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--threshold", type=int, default=238, help="Foreground grayscale threshold")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    image = read_image(args.input)
    reference = read_image(args.reference)
    cfg = ProductLocatorConfig(foreground_threshold=args.threshold)
    result = align_to_reference(image, reference, cfg)

    write_image(out / "aligned.png", result.aligned)
    write_image(out / "coarse.png", result.coarse)
    write_image(out / "foreground_mask.png", result.foreground_mask)
    write_image(out / "overlay.png", make_overlay(reference, result.aligned))
    write_json(out / "alignment.json", result.to_dict())

    print(f"Aligned image: {out / 'aligned.png'}")
    print(f"Detected angle: {result.location.angle_deg:.3f} deg")
    if result.ecc_score is None:
        print("ECC: failed; coarse alignment was saved as fallback")
    else:
        print(f"ECC score: {result.ecc_score:.6f}")


if __name__ == "__main__":
    main()
