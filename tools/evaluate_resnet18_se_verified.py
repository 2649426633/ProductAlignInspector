from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.io_utils import read_image, write_image
from product_align_inspector.roi import crop_roi, validate_roi

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def collect_images(root: Path):
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def scenario_of(relative: Path) -> str:
    parts = relative.parts
    if parts and parts[0].lower() == 'good':
        return 'good'
    if len(parts) >= 2 and parts[0].lower() == 'ng':
        return parts[1]
    return parts[0] if parts else 'unknown'


def center_focus_bgr(image: np.ndarray, ratio: float) -> np.ndarray:
    h, w = image.shape[:2]
    side = max(8, int(round(min(h, w) * ratio)))
    cx, cy = w // 2, h // 2
    x0 = max(0, min(w - side, cx - side // 2))
    y0 = max(0, min(h - side, cy - side // 2))
    return image[y0:y0 + side, x0:x0 + side].copy()


def load_checkpoint(path: Path, device: torch.device):
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location='cpu')
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.to(device).eval()
    return model, ckpt


def make_tf(input_size: int):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def predict(model, crops: list[np.ndarray], tf, ratio: float, device: torch.device) -> np.ndarray:
    tensors = []
    for crop in crops:
        focus = center_focus_bgr(crop, ratio)
        rgb = cv2.cvtColor(focus, cv2.COLOR_BGR2RGB)
        tensors.append(tf(Image.fromarray(rgb)))
    batch = torch.stack(tensors).to(device)
    with torch.inference_mode():
        return torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy().astype(np.float32)


def candidate_rois(roi, width: int, height: int, amount: int):
    x, y, w, h = [int(v) for v in roi]
    offsets = [(0, 0)]
    if amount > 0:
        offsets += [(-amount, 0), (amount, 0), (0, -amount), (0, amount), (-amount, -amount), (-amount, amount), (amount, -amount), (amount, amount)]
    out = []
    for dx, dy in offsets:
        nx = max(0, min(width - w, x + dx))
        ny = max(0, min(height - h, y + dy))
        out.append(([nx, ny, w, h], dx, dy))
    return out


def draw_box(canvas: np.ndarray, roi, label: str, status: str):
    x, y, w, h = [int(v) for v in roi]
    color = (0, 190, 0) if status == 'PASS' else (0, 0, 255)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
    cv2.putText(canvas, label, (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def main() -> None:
    p = argparse.ArgumentParser(description='Evaluate S/E ResNet18 trained from verified debug_good_crops. Jitter, if used, is aggregated by MEDIAN; it is never chosen by max/min best-case scoring.')
    p.add_argument('--test-root', required=True)
    p.add_argument('--model-root', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--scenario', action='append')
    p.add_argument('--local-jitter', type=int, default=0)
    p.add_argument('--foreground-threshold', type=int, default=238)
    args = p.parse_args()

    device = torch.device('cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    root = Path(args.model_root).resolve()
    s_model, s_ckpt = load_checkpoint(root / 'S' / 'best.pt', device)
    e_model, e_ckpt = load_checkpoint(root / 'E' / 'best.pt', device)
    s_thr = float(s_ckpt.get('threshold_screw', 0.5)); e_thr = float(e_ckpt.get('threshold_screw', 0.5))
    s_ratio = float(s_ckpt.get('center_ratio', 0.72)); e_ratio = float(e_ckpt.get('center_ratio', 0.72))
    s_tf = make_tf(int(s_ckpt.get('input_size', 224))); e_tf = make_tf(int(e_ckpt.get('input_size', 224)))

    reference = read_image(Path(args.reference).resolve())
    config = json.loads(Path(args.config).resolve().read_text(encoding='utf-8'))
    slots = [x for x in config.get('screw_slots', []) if bool(x.get('enabled', True))]
    s_slots = [x for x in slots if str(x.get('id', '')).upper().startswith('S')]
    e_slots = [x for x in slots if str(x.get('id', '')).upper().startswith('E')]
    filters = {str(v).strip().lower() for v in (args.scenario or []) if str(v).strip()}
    test_root = Path(args.test_root).resolve()
    selected = []
    for path in collect_images(test_root):
        rel = path.relative_to(test_root); sc = scenario_of(rel)
        if filters and sc.lower() not in filters:
            continue
        selected.append((path, rel, sc))

    out = Path(args.output).resolve(); overlay_dir = out / 'overlays_aligned'; overlay_dir.mkdir(parents=True, exist_ok=True)
    align_cfg = ProductLocatorConfig(foreground_threshold=args.foreground_threshold)
    image_rows, roi_rows = [], []
    print('=== Verified-crop ResNet18 S/E Evaluation ===')
    print(f'Device: {device} | Images: {len(selected)}')
    print(f'S threshold={s_thr:.3f}, center_ratio={s_ratio:.2f}')
    print(f'E threshold={e_thr:.3f}, center_ratio={e_ratio:.2f}')
    print(f'Local jitter={args.local_jitter}px; aggregation=MEDIAN (not best-case max/min)')

    for i, (path, rel, scenario) in enumerate(selected, 1):
        try:
            raw = read_image(path)
            al = align_to_reference(raw, reference, align_cfg)
            aligned = al.aligned; h, w = aligned.shape[:2]
            canvas = aligned.copy(); ng = []
            for group, items, model, tf, ratio, thr in [('S', s_slots, s_model, s_tf, s_ratio, s_thr), ('E', e_slots, e_model, e_tf, e_ratio, e_thr)]:
                for item in items:
                    roi = item['roi']
                    if not validate_roi(roi, w, h):
                        raise RuntimeError(f"invalid ROI {item['id']}: {roi}")
                    candidates = candidate_rois(roi, w, h, args.local_jitter)
                    probs = predict(model, [crop_roi(aligned, r[0]) for r in candidates], tf, ratio, device)
                    p_screw = float(np.median(probs))
                    status = 'PASS' if (p_screw >= thr if group == 'S' else p_screw < thr) else 'NG'
                    if status == 'NG':
                        ng.append(str(item['id']))
                    draw_box(canvas, roi, f"{item['id']} {status} P(screw)={p_screw:.3f}", status)
                    roi_rows.append({'relative_path': rel.as_posix(), 'scenario': scenario, 'roi_id': item['id'], 'group': group,
                                     'p_screw': p_screw, 'threshold': thr, 'status': status,
                                     'jitter_px': args.local_jitter, 'aggregation': 'median'})
            final = 'NG' if ng else 'PASS'
            overlay_rel = Path('overlays_aligned') / rel.with_suffix('.jpg'); overlay_path = out / overlay_rel; overlay_path.parent.mkdir(parents=True, exist_ok=True)
            write_image(overlay_path, canvas)
            image_rows.append({'relative_path': rel.as_posix(), 'scenario': scenario, 'final_status': final, 'ng_rois': ','.join(ng),
                               'alignment_method': al.method, 'ecc_score': '' if al.ecc_score is None else al.ecc_score,
                               'overlay': overlay_rel.as_posix(), 'error': ''})
            print(f"[{i}/{len(selected)}] {rel.as_posix()} -> {final} | NG={ng or '-'} | align={al.method} ecc={al.ecc_score}")
        except Exception as exc:
            image_rows.append({'relative_path': rel.as_posix(), 'scenario': scenario, 'final_status': 'RETRY', 'ng_rois': '', 'alignment_method': '', 'ecc_score': '', 'overlay': '', 'error': str(exc)})
            print(f"[{i}/{len(selected)}] {rel.as_posix()} -> RETRY: {exc}")

    out.mkdir(parents=True, exist_ok=True)
    with (out / 'image_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['relative_path','scenario','final_status','ng_rois','alignment_method','ecc_score','overlay','error']); writer.writeheader(); writer.writerows(image_rows)
    with (out / 'roi_scores.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['relative_path','scenario','roi_id','group','p_screw','threshold','status','jitter_px','aggregation']); writer.writeheader(); writer.writerows(roi_rows)
    counts = {k: sum(1 for r in image_rows if r['final_status'] == k) for k in ('PASS','NG','RETRY')}
    (out / 'summary.json').write_text(json.dumps({'images': len(image_rows), 'counts': counts, 'jitter_px': args.local_jitter, 'aggregation': 'median'}, indent=2), encoding='utf-8')
    print(f"\nPASS/NG/RETRY: {counts['PASS']}/{counts['NG']}/{counts['RETRY']}")
    print(f'Overlays: {overlay_dir}')
    print(f'ROI scores: {out / "roi_scores.csv"}')


if __name__ == '__main__':
    main()
