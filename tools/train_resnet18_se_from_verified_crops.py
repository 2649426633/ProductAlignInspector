from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Sample:
    path: Path
    source: str
    roi_id: str
    label: int  # 0=empty, 1=screw


def center_focus_bgr(image: np.ndarray, ratio: float) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError('empty crop')
    h, w = image.shape[:2]
    side = max(8, int(round(min(h, w) * ratio)))
    cx, cy = w // 2, h // 2
    x0 = max(0, min(w - side, cx - side // 2))
    y0 = max(0, min(h - side, cy - side // 2))
    return image[y0:y0 + side, x0:x0 + side].copy()


class CropDataset(Dataset):
    def __init__(self, samples: list[Sample], input_size: int, center_ratio: float, train: bool) -> None:
        self.samples = samples
        self.center_ratio = center_ratio
        aug = []
        if train:
            aug = [
                transforms.RandomAffine(degrees=4.0, translate=(0.035, 0.035), scale=(0.96, 1.04), fill=255),
                transforms.ColorJitter(brightness=0.12, contrast=0.12),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.8))], p=0.10),
            ]
        self.tf = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            *aug,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]
        image = cv2.imread(str(s.path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(s.path)
        image = center_focus_bgr(image, self.center_ratio)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.tf(Image.fromarray(image)), int(s.label)


def load_samples(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        roi_id = d.name.upper()
        if roi_id.startswith('S'):
            label = 1
        elif roi_id.startswith('E'):
            label = 0
        else:
            continue
        for p in sorted(x for x in d.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS):
            samples.append(Sample(p, p.stem, roi_id, label))
    return samples


def build_model(weights_path: Path, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    try:
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def set_trainable(model: nn.Module, phase: str) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if phase == 'head':
        for p in model.fc.parameters():
            p.requires_grad = True
    else:
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True


def eval_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probs, labels = [], []
    loss_sum, count = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += float(loss.item()) * int(y.numel())
            count += int(y.numel())
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            labels.append(y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels), loss_sum / max(1, count)


def choose_threshold(p: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, float]]:
    best = None
    for t in np.linspace(0.05, 0.95, 181):
        pred = (p >= t).astype(np.int64)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        sr = tp / max(1, tp + fn)
        er = tn / max(1, tn + fp)
        bal = 0.5 * (sr + er)
        key = (bal, min(sr, er), -abs(float(t) - 0.5))
        if best is None or key > best[0]:
            best = (key, float(t), sr, er, tp, tn, fp, fn)
    assert best is not None
    _, t, sr, er, tp, tn, fp, fn = best
    return t, {'balanced_accuracy': 0.5 * (sr + er), 'screw_recall': sr, 'empty_recall': er,
               'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def balanced_subset(samples: list[Sample], seed: int) -> list[Sample]:
    screws = [s for s in samples if s.label == 1]
    empties = [s for s in samples if s.label == 0]
    rng = random.Random(seed)
    rng.shuffle(empties)
    n = min(len(screws), len(empties))
    return screws + empties[:n]


def train_role(role: str, train_samples: list[Sample], val_samples: list[Sample], weights: Path,
               output: Path, device: torch.device, input_size: int, center_ratio: float,
               batch_size: int, epochs: int, freeze_epochs: int, lr: float, seed: int) -> None:
    model = build_model(weights, device)
    if role == 'S':
        train_samples = balanced_subset(train_samples, seed)
        val_samples = balanced_subset(val_samples, seed + 1)
    train_loader = DataLoader(CropDataset(train_samples, input_size, center_ratio, True), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(CropDataset(val_samples, input_size, center_ratio, False), batch_size=batch_size, shuffle=False, num_workers=0)
    counts = np.bincount([s.label for s in train_samples], minlength=2).astype(np.float32)
    weights_ce = counts.sum() / np.maximum(1.0, 2.0 * counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights_ce, dtype=torch.float32, device=device))
    best_score = -1.0
    best = None
    history = []
    for epoch in range(1, epochs + 1):
        phase = 'head' if epoch <= freeze_epochs else 'layer4'
        set_trainable(model, phase)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr if phase == 'head' else lr * 0.25, weight_decay=1e-4)
        model.train()
        loss_sum, count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * int(y.numel())
            count += int(y.numel())
        p, y, val_loss = eval_probs(model, val_loader, device)
        threshold, metrics = choose_threshold(p, y)
        score = float(metrics['balanced_accuracy'])
        row = {'epoch': epoch, 'phase': phase, 'train_loss': loss_sum / max(1, count), 'val_loss': val_loss,
               'threshold': threshold, **metrics}
        history.append(row)
        print(f"{role} epoch {epoch:02d}/{epochs} {phase}: loss={row['train_loss']:.4f} val={val_loss:.4f} bal_acc={score:.3f} thr={threshold:.3f} screw_recall={metrics['screw_recall']:.3f} empty_recall={metrics['empty_recall']:.3f}")
        if score > best_score:
            best_score = score
            best = {
                'architecture': 'resnet18', 'role': role, 'classes': ['empty', 'screw'],
                'input_size': input_size, 'center_ratio': center_ratio, 'threshold_screw': threshold,
                'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'training': {'source': 'verified_debug_good_crops', 'train_samples': len(train_samples),
                             'val_samples': len(val_samples), 'best_balanced_accuracy': best_score,
                             'ng_source_images': 0, 'test_source_images': 0, 'metrics': metrics},
            }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(best, output / 'best.pt')
    with (output / 'history.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader(); writer.writerows(history)
    (output / 'summary.json').write_text(json.dumps({k: v for k, v in best.items() if k != 'state_dict'}, indent=2), encoding='utf-8')


def main() -> None:
    p = argparse.ArgumentParser(description='Train independent S/E ResNet18 models directly from already-verified debug_good_crops. No synthetic images are created.')
    p.add_argument('--crop-root', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--input-size', type=int, default=224)
    p.add_argument('--center-ratio', type=float, default=0.72)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--freeze-epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=8e-4)
    p.add_argument('--val-ratio', type=float, default=0.20)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if not 0.50 <= args.center_ratio <= 1.0:
        raise SystemExit('--center-ratio must be 0.50..1.0')
    root = Path(args.crop_root).resolve()
    weights = Path(args.weights).resolve()
    output = Path(args.output).resolve()
    samples = load_samples(root)
    s = [x for x in samples if x.label == 1]
    e = [x for x in samples if x.label == 0]
    if not s or not e:
        raise SystemExit(f'Need S and E crop folders. Found screw={len(s)}, empty={len(e)}')
    sources = sorted({x.source for x in samples})
    rng = random.Random(args.seed); rng.shuffle(sources)
    val_n = max(2, int(round(len(sources) * args.val_ratio)))
    val_sources = set(sources[:val_n]); train_sources = set(sources[val_n:])
    train_samples = [x for x in samples if x.source in train_sources]
    val_samples = [x for x in samples if x.source in val_sources]
    device = torch.device('cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    print('=== Verified crop training ===')
    print(f'Crop root: {root}')
    print(f'Screw crops: {len(s)} | Empty crops: {len(e)}')
    print(f'Train/val source images: {len(train_sources)}/{len(val_sources)}')
    print(f'Center focus ratio: {args.center_ratio:.2f}')
    print('Synthetic/counterfactual images: 0')
    print('NG/test source images: 0')
    split = {'train_sources': sorted(train_sources), 'val_sources': sorted(val_sources), 'center_ratio': args.center_ratio,
             'synthetic_images': 0, 'ng_source_images': 0, 'test_source_images': 0}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'split.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    print('\n=== Train S model ===')
    train_role('S', train_samples, val_samples, weights, output / 'S', device, args.input_size, args.center_ratio,
               args.batch_size, args.epochs, args.freeze_epochs, args.lr, args.seed)
    print('\n=== Train E model ===')
    train_role('E', train_samples, val_samples, weights, output / 'E', device, args.input_size, args.center_ratio,
               args.batch_size, args.epochs, args.freeze_epochs, args.lr, args.seed + 1000)
    print(f'\nComplete: {output}')


if __name__ == '__main__':
    main()
