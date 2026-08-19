from __future__ import annotations

import argparse
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def center_focus(image, ratio: float):
    h, w = image.shape[:2]
    side = max(8, int(round(min(h, w) * ratio)))
    cx, cy = w // 2, h // 2
    x0 = max(0, min(w - side, cx - side // 2))
    y0 = max(0, min(h - side, cy - side // 2))
    return image[y0:y0 + side, x0:x0 + side].copy()


def main() -> None:
    p = argparse.ArgumentParser(description='Preview exactly what the verified-crop ResNet18 will see after center focusing. No training is performed.')
    p.add_argument('--crop-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--center-ratio', type=float, default=0.72)
    p.add_argument('--limit-per-roi', type=int, default=12)
    args = p.parse_args()
    if not 0.50 <= args.center_ratio <= 1.0:
        raise SystemExit('--center-ratio must be 0.50..1.0')
    root = Path(args.crop_root).resolve()
    out = Path(args.output).resolve()
    total = 0
    for roi_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.upper().startswith(('S','E'))):
        files = [p for p in sorted(roi_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        for src in files[:max(1, args.limit_per_roi)]:
            image = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if image is None:
                continue
            focus = center_focus(image, args.center_ratio)
            target = out / roi_dir.name / src.name
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), focus)
            total += 1
    print('=== Verified ROI focus preview ===')
    print(f'Source: {root}')
    print(f'Center ratio: {args.center_ratio:.2f}')
    print(f'Written: {total}')
    print(f'Output: {out}')
    print('No model was trained. Inspect these images first.')


if __name__ == '__main__':
    main()
