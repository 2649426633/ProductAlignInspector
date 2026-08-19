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
from product_align_inspector.io_utils import read_image, write_image
from product_align_inspector.roi import crop_roi, validate_roi

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def scenario_of(relative: Path) -> str:
    parts = relative.parts
    if parts and parts[0].lower() == 'good':
        return 'good'
    if len(parts) >= 2 and parts[0].lower() == 'ng':
        return parts[1]
    return parts[0] if parts else 'unknown'


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
    # Compare standardized gradient maps with a small translation tolerance.
    if max_shift <= 0:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.sum(a * b) / denom) if denom > 1e-8 else -1.0
    best = -1.0
    h, w = a.shape
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
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
            if score > best:
                best = score
    return best


def bank_score(query: np.ndarray, bank: list[np.ndarray], max_shift: int, topk: int) -> float:
    scores = sorted((shifted_corr(query, x, max_shift) for x in bank), reverse=True)
    k = max(1, min(topk, len(scores)))
    return float(np.mean(scores[:k]))


def calibrate_threshold(bank: list[np.ndarray], max_shift: int, topk: int, quantile: float, margin: float) -> tuple[float, list[float]]:
    loo = []
    for i, q in enumerate(bank):
        others = [x for j, x in enumerate(bank) if j != i]
        loo.append(bank_score(q, others, max_shift, topk))
    q = float(np.quantile(np.asarray(loo, dtype=np.float32), quantile))
    threshold = q - abs(float(margin))
    return threshold, loo


def load_bank(crop_root: Path, roi_id: str, size: int, center_ratio: float) -> tuple[list[np.ndarray], list[str]]:
    folder = crop_root / roi_id
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    feats, names = [], []
    for p in collect_images(folder):
        img = read_image(p)
        feats.append(feature_image(img, size, center_ratio))
        names.append(p.name)
    if len(feats) < 8:
        raise RuntimeError(f'{roi_id}: need at least 8 verified GOOD crops, found {len(feats)}')
    return feats, names


def draw_box(canvas: np.ndarray, roi, label: str, status: str) -> None:
    x, y, w, h = [int(v) for v in roi]
    color = (0, 190, 0) if status == 'PRESENT' else (0, 0, 255)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
    cv2.putText(canvas, label, (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main() -> None:
    p = argparse.ArgumentParser(description='Evaluate S01/S02 screw presence using verified GOOD crop template banks only. No neural training and no NG/test calibration.')
    p.add_argument('--good-crop-root', required=True)
    p.add_argument('--test-root', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--scenario', action='append')
    p.add_argument('--center-ratio', type=float, default=0.58)
    p.add_argument('--feature-size', type=int, default=96)
    p.add_argument('--max-shift', type=int, default=3)
    p.add_argument('--topk', type=int, default=3)
    p.add_argument('--good-quantile', type=float, default=0.02)
    p.add_argument('--threshold-margin', type=float, default=0.02)
    p.add_argument('--foreground-threshold', type=int, default=238)
    args = p.parse_args()

    if not 0.35 <= args.center_ratio <= 1.0:
        raise SystemExit('--center-ratio must be 0.35..1.0')
    crop_root = Path(args.good_crop_root).resolve()
    test_root = Path(args.test_root).resolve()
    reference_path = Path(args.reference).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    overlay_dir = output / 'overlays_aligned'
    output.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    reference = read_image(reference_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    s_slots = [x for x in config.get('screw_slots', []) if bool(x.get('enabled', True)) and str(x.get('id', '')).upper().startswith('S')]
    if not s_slots:
        raise SystemExit('No S ROIs in config')

    banks: dict[str, list[np.ndarray]] = {}
    thresholds: dict[str, float] = {}
    calibration_rows = []
    print('=== Verified GOOD screw-template calibration ===')
    print(f'GOOD crop root: {crop_root}')
    print(f'Center ratio: {args.center_ratio:.2f} | feature size: {args.feature_size} | shift: +/-{args.max_shift}')
    print('NG/test images used for calibration: 0')
    for slot in s_slots:
        rid = str(slot['id'])
        bank, names = load_bank(crop_root, rid, args.feature_size, args.center_ratio)
        threshold, loo = calibrate_threshold(bank, args.max_shift, args.topk, args.good_quantile, args.threshold_margin)
        banks[rid] = bank
        thresholds[rid] = threshold
        print(f'{rid}: bank={len(bank)} GOOD, LOO min={min(loo):.4f} median={np.median(loo):.4f} threshold={threshold:.4f}')
        for name, score in zip(names, loo):
            calibration_rows.append({'roi_id': rid, 'source': name, 'good_loo_score': score, 'threshold': threshold})

    filters = {str(x).strip().lower() for x in (args.scenario or []) if str(x).strip()}
    selected = []
    for path in collect_images(test_root):
        rel = path.relative_to(test_root)
        scenario = scenario_of(rel)
        if filters and scenario.lower() not in filters:
            continue
        selected.append((path, rel, scenario))

    align_cfg = ProductLocatorConfig(foreground_threshold=args.foreground_threshold)
    image_rows, score_rows = [], []
    print(f'\n=== Evaluate S screw presence: {len(selected)} images ===')
    for idx, (path, rel, scenario) in enumerate(selected, 1):
        try:
            raw = read_image(path)
            result = align_to_reference(raw, reference, align_cfg)
            aligned = result.aligned
            h, w = aligned.shape[:2]
            canvas = aligned.copy()
            missing = []
            for slot in s_slots:
                rid = str(slot['id'])
                roi = slot['roi']
                if not validate_roi(roi, w, h):
                    raise RuntimeError(f'invalid ROI {rid}: {roi}')
                crop = crop_roi(aligned, roi)
                feat = feature_image(crop, args.feature_size, args.center_ratio)
                score = bank_score(feat, banks[rid], args.max_shift, args.topk)
                status = 'PRESENT' if score >= thresholds[rid] else 'MISSING'
                if status == 'MISSING':
                    missing.append(rid)
                draw_box(canvas, roi, f'{rid} {status} score={score:.3f} thr={thresholds[rid]:.3f}', status)
                score_rows.append({'relative_path': rel.as_posix(), 'scenario': scenario, 'roi_id': rid, 'score': score, 'threshold': thresholds[rid], 'status': status, 'alignment_method': result.method, 'ecc_score': '' if result.ecc_score is None else result.ecc_score})

            final_status = 'NG' if missing else 'PASS'
            overlay_rel = Path('overlays_aligned') / rel.with_suffix('.jpg')
            overlay_path = output / overlay_rel
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            write_image(overlay_path, canvas)
            image_rows.append({'relative_path': rel.as_posix(), 'scenario': scenario, 'final_status': final_status, 'missing_s_rois': ','.join(missing), 'alignment_method': result.method, 'ecc_score': '' if result.ecc_score is None else result.ecc_score, 'overlay': overlay_rel.as_posix(), 'error': ''})
            print(f'[{idx}/{len(selected)}] {rel.as_posix()} -> {final_status} | missing={missing or "-"} | align={result.method} ecc={result.ecc_score}')
        except Exception as exc:
            image_rows.append({'relative_path': rel.as_posix(), 'scenario': scenario, 'final_status': 'RETRY', 'missing_s_rois': '', 'alignment_method': '', 'ecc_score': '', 'overlay': '', 'error': str(exc)})
            print(f'[{idx}/{len(selected)}] {rel.as_posix()} -> RETRY: {exc}')

    with (output / 'good_loo_scores.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['roi_id', 'source', 'good_loo_score', 'threshold'])
        writer.writeheader(); writer.writerows(calibration_rows)
    with (output / 's_presence_scores.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['relative_path', 'scenario', 'roi_id', 'score', 'threshold', 'status', 'alignment_method', 'ecc_score'])
        writer.writeheader(); writer.writerows(score_rows)
    with (output / 'image_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['relative_path', 'scenario', 'final_status', 'missing_s_rois', 'alignment_method', 'ecc_score', 'overlay', 'error'])
        writer.writeheader(); writer.writerows(image_rows)

    counts = {k: sum(1 for r in image_rows if r['final_status'] == k) for k in ('PASS', 'NG', 'RETRY')}
    summary = {
        'images': len(image_rows), 'counts': counts, 'method': 'gradient_template_bank',
        'center_ratio': args.center_ratio, 'feature_size': args.feature_size, 'max_shift': args.max_shift,
        'topk': args.topk, 'good_quantile': args.good_quantile, 'threshold_margin': args.threshold_margin,
        'thresholds': thresholds, 'ng_test_used_for_calibration': 0,
    }
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n=== Complete ===')
    print(f"PASS/NG/RETRY: {counts['PASS']}/{counts['NG']}/{counts['RETRY']}")
    print(f'GOOD LOO scores: {output / "good_loo_scores.csv"}')
    print(f'Test S scores:   {output / "s_presence_scores.csv"}')
    print(f'Overlays:        {overlay_dir}')


if __name__ == '__main__':
    main()
