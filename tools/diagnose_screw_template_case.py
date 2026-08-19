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

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.io_utils import read_image, write_image
from product_align_inspector.roi import crop_roi, validate_roi

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def center_focus(image: np.ndarray, ratio: float) -> np.ndarray:
    h, w = image.shape[:2]
    side = max(12, int(round(min(h, w) * ratio)))
    cx, cy = w // 2, h // 2
    x0 = max(0, min(w - side, cx - side // 2))
    y0 = max(0, min(h - side, cy - side // 2))
    return image[y0:y0 + side, x0:x0 + side].copy()


def feature_image(image: np.ndarray, size: int, center_ratio: float) -> np.ndarray:
    image = center_focus(image, center_ratio)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.resize(mag, (size, size), interpolation=cv2.INTER_AREA)
    mag -= float(mag.mean())
    std = float(mag.std())
    if std > 1e-6:
        mag /= std
    return mag.astype(np.float32)


def shifted_corr(a: np.ndarray, b: np.ndarray, max_shift: int) -> float:
    best = -1.0
    h, w = a.shape
    shifts = range(-max_shift, max_shift + 1) if max_shift > 0 else (0,)
    for dy in shifts:
        for dx in shifts:
            ax0, ay0 = max(0, dx), max(0, dy)
            bx0, by0 = max(0, -dx), max(0, -dy)
            ww = w - abs(dx)
            hh = h - abs(dy)
            if ww < w // 2 or hh < h // 2:
                continue
            aa = a[ay0:ay0 + hh, ax0:ax0 + ww]
            bb = b[by0:by0 + hh, bx0:bx0 + ww]
            denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
            if denom <= 1e-8:
                continue
            score = float(np.sum(aa * bb) / denom)
            best = max(best, score)
    return best


def score_against_bank(query: np.ndarray, bank: list[np.ndarray], max_shift: int, topk: int) -> tuple[float, list[tuple[float, int]]]:
    scored = sorted(((shifted_corr(query, feat, max_shift), i) for i, feat in enumerate(bank)), reverse=True)
    k = max(1, min(topk, len(scored)))
    return float(np.mean([s for s, _ in scored[:k]])), scored


def load_bank(folder: Path, size: int, center_ratio: float) -> tuple[list[np.ndarray], list[Path]]:
    paths = collect_images(folder)
    feats = [feature_image(read_image(p), size, center_ratio) for p in paths]
    if len(feats) < 8:
        raise RuntimeError(f'need at least 8 templates in {folder}, found {len(feats)}')
    return feats, paths


def calibrate(bank: list[np.ndarray], max_shift: int, topk: int, quantile: float, margin: float) -> tuple[float, list[float]]:
    loo = []
    for i, q in enumerate(bank):
        others = [x for j, x in enumerate(bank) if j != i]
        score, _ = score_against_bank(q, others, max_shift, topk)
        loo.append(score)
    q = float(np.quantile(np.asarray(loo, dtype=np.float32), quantile))
    return q - abs(float(margin)), loo


def shifted_roi(roi, dx: int, dy: int, width: int, height: int) -> list[int] | None:
    x, y, w, h = [int(v) for v in roi]
    nx, ny = x + dx, y + dy
    candidate = [nx, ny, w, h]
    return candidate if validate_roi(candidate, width, height) else None


def draw_rect(canvas: np.ndarray, roi, color, label: str) -> None:
    x, y, w, h = [int(v) for v in roi]
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
    cv2.putText(canvas, label, (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def main() -> None:
    p = argparse.ArgumentParser(description='Diagnose one GOOD/NG image by searching the real S ROI locally around the nominal aligned position.')
    p.add_argument('--image', required=True)
    p.add_argument('--good-crop-root', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--center-ratio', type=float, default=0.58)
    p.add_argument('--feature-size', type=int, default=96)
    p.add_argument('--feature-shift', type=int, default=5)
    p.add_argument('--topk', type=int, default=1)
    p.add_argument('--good-quantile', type=float, default=0.02)
    p.add_argument('--threshold-margin', type=float, default=0.02)
    p.add_argument('--roi-search-px', type=int, default=20)
    p.add_argument('--roi-search-step', type=int, default=2)
    p.add_argument('--foreground-threshold', type=int, default=238)
    args = p.parse_args()

    image_path = Path(args.image).resolve()
    crop_root = Path(args.good_crop_root).resolve()
    reference = read_image(Path(args.reference).resolve())
    config = json.loads(Path(args.config).resolve().read_text(encoding='utf-8'))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    slots = [x for x in config.get('screw_slots', []) if bool(x.get('enabled', True)) and str(x.get('id', '')).upper().startswith('S')]
    if not slots:
        raise SystemExit('No S ROIs found')

    raw = read_image(image_path)
    result = align_to_reference(raw, reference, ProductLocatorConfig(foreground_threshold=args.foreground_threshold))
    aligned = result.aligned
    h, w = aligned.shape[:2]
    canvas = aligned.copy()

    print('=== Screw template single-case diagnostic ===')
    print(f'Image: {image_path}')
    print(f'Alignment: {result.method} | ECC={result.ecc_score} | inlier={result.feature_inlier_ratio}')
    print(f'Actual ROI search: +/-{args.roi_search_px}px step={args.roi_search_step}px')

    summary = {'image': str(image_path), 'alignment_method': result.method, 'ecc_score': result.ecc_score, 'feature_inlier_ratio': result.feature_inlier_ratio, 'rois': {}}

    for slot in slots:
        rid = str(slot['id'])
        nominal_roi = [int(v) for v in slot['roi']]
        bank, template_paths = load_bank(crop_root / rid, args.feature_size, args.center_ratio)
        threshold, loo = calibrate(bank, args.feature_shift, args.topk, args.good_quantile, args.threshold_margin)

        nominal_crop = crop_roi(aligned, nominal_roi)
        nominal_feat = feature_image(nominal_crop, args.feature_size, args.center_ratio)
        nominal_score, nominal_rank = score_against_bank(nominal_feat, bank, args.feature_shift, args.topk)

        best = (nominal_score, 0, 0, nominal_roi, nominal_rank)
        for dy in range(-args.roi_search_px, args.roi_search_px + 1, args.roi_search_step):
            for dx in range(-args.roi_search_px, args.roi_search_px + 1, args.roi_search_step):
                roi = shifted_roi(nominal_roi, dx, dy, w, h)
                if roi is None:
                    continue
                feat = feature_image(crop_roi(aligned, roi), args.feature_size, args.center_ratio)
                score, ranked = score_against_bank(feat, bank, args.feature_shift, args.topk)
                if score > best[0]:
                    best = (score, dx, dy, roi, ranked)

        best_score, best_dx, best_dy, best_roi, best_rank = best
        nearest_idx = int(best_rank[0][1])
        nearest_path = template_paths[nearest_idx]
        best_crop = crop_roi(aligned, best_roi)
        nearest_img = read_image(nearest_path)

        write_image(output / f'{rid}_01_nominal.png', nominal_crop)
        write_image(output / f'{rid}_02_best_local.png', best_crop)
        write_image(output / f'{rid}_03_nearest_good_{nearest_path.stem}.png', nearest_img)
        write_image(output / f'{rid}_04_nominal_focus.png', center_focus(nominal_crop, args.center_ratio))
        write_image(output / f'{rid}_05_best_focus.png', center_focus(best_crop, args.center_ratio))
        write_image(output / f'{rid}_06_nearest_good_focus.png', center_focus(nearest_img, args.center_ratio))

        draw_rect(canvas, nominal_roi, (0, 165, 255), f'{rid} nominal {nominal_score:.3f}')
        if best_dx != 0 or best_dy != 0:
            draw_rect(canvas, best_roi, (0, 220, 0), f'{rid} best {best_score:.3f} d=({best_dx},{best_dy})')

        nominal_status = 'PRESENT' if nominal_score >= threshold else 'MISSING'
        best_status = 'PRESENT' if best_score >= threshold else 'MISSING'
        print(f'{rid}: threshold={threshold:.4f} LOOmin={min(loo):.4f} LOOmedian={float(np.median(loo)):.4f}')
        print(f'  nominal={nominal_score:.4f} -> {nominal_status}')
        print(f'  local_best={best_score:.4f} at dx={best_dx},dy={best_dy} -> {best_status}')
        print(f'  nearest GOOD={nearest_path.name} corr={best_rank[0][0]:.4f}')

        summary['rois'][rid] = {
            'threshold': threshold,
            'loo_min': min(loo),
            'loo_median': float(np.median(loo)),
            'nominal_score': nominal_score,
            'nominal_status': nominal_status,
            'best_score': best_score,
            'best_status': best_status,
            'best_dx': best_dx,
            'best_dy': best_dy,
            'nearest_good': nearest_path.name,
            'nearest_good_corr': float(best_rank[0][0]),
        }

    write_image(output / 'overlay_nominal_vs_best.png', canvas)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Output: {output}')


if __name__ == '__main__':
    main()
