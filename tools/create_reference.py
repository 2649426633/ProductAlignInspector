from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.io_utils import read_image, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create the fixed-camera RAW geometry reference. "
            "No crop, resize, rotation, inpaint, or normalization is performed."
        )
    )
    parser.add_argument("--input", required=True, help="One real RAW GOOD image")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    source = Path(args.input).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    image = read_image(source)
    suffix = source.suffix.lower() or ".bmp"
    reference_path = out / f"reference_raw{suffix}"

    # Byte-for-byte copy whenever possible: the canonical geometry reference is
    # exactly one real camera frame.
    shutil.copy2(source, reference_path)

    write_json(
        out / "reference_meta.json",
        {
            "source": str(source),
            "reference": str(reference_path),
            "reference_width": int(image.shape[1]),
            "reference_height": int(image.shape[0]),
            "geometry": "rigid",
            "allowed_transform": ["rotation", "translation_x", "translation_y"],
            "forbidden_transform": ["scale", "resize_x_y", "shear", "perspective"],
            "note": "ROI coordinates must be annotated on this exact raw reference coordinate system.",
        },
    )

    print(f"RAW reference created: {reference_path}")
    print(f"Reference size: {image.shape[1]} x {image.shape[0]}")
    print("No crop / resize / rotation / inpaint was applied.")


if __name__ == "__main__":
    main()
