from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.io_utils import read_image
from product_align_inspector.roi import crop_roi, validate_roi

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASSES = ("empty", "screw")


@dataclass
class Sample:
    source: str
    roi_id: str
    image_bgr: np.ndarray
    label: int  # 0 empty, 1 screw


class RoiDataset(Dataset):
    def __init__(self, samples: list[Sample], input_size: int, train: bool) -> None:
        self.samples = samples
        if train:
            self.tf = transforms.Compose([
                transforms.Lambda(self._pad_square),
                transforms.Resize((input_size, input_size)),
                transforms.RandomAffine(degrees=5.0, translate=(0.04, 0.04), scale=(0.95, 1.05), fill=255),
                transforms.ColorJitter(brightness=0.18, contrast=0.18),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Lambda(self._pad_square),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    @staticmethod
    def _pad_square(image: Image.Image) -> Image.Image:
        w, h = image.size
        side = max(w, h)
        out = Image.new("RGB", (side, side), (255, 255, 255))
        out.paste(image, ((side - w) // 2, (side - h) // 2))
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]
        rgb = cv2.cvtColor(s.image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        return self.tf(image), int(s.label)


def collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_local_resnet18(weights_path: Path, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def set_trainable(model: nn.Module, phase: str) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if phase == "head":
        for p in model.fc.parameters():
            p.requires_grad = True
    elif phase == "layer4":
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True
    else:
        raise ValueError(phase)


def class_weights(samples: list[Sample], device: torch.device) -> torch.Tensor:
    counts = np.bincount([s.label for s in samples], minlength=2).astype(np.float32)
    total = float(counts.sum())
    weights = total / np.maximum(1.0, 2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    probs = []
    labels = []
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            losses.append(float(criterion(logits, y).item()) * int(y.numel()))
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.append(p.cpu().numpy())
            labels.append(y.cpu().numpy())
    if not labels:
        return 0.0, np.empty(0), np.empty(0, dtype=np.int64)
    y_true = np.concatenate(labels)
    p_screw = np.concatenate(probs)
    return sum(losses) / max(1, len(y_true)), p_screw, y_true


def choose_threshold(p_screw: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, float]]:
    best = None
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (p_screw >= threshold).astype(np.int64)
        tp = int(((pred == 1) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        tpr = tp / max(1, tp + fn)
        tnr = tn / max(1, tn + fp)
        bal = 0.5 * (tpr + tnr)
        key = (bal, min(tpr, tnr), -abs(float(threshold) - 0.5))
        if best is None or key > best[0]:
            best = (key, float(threshold), tp, tn, fp, fn, tpr, tnr)
    assert best is not None
    _, threshold, tp, tn, fp, fn, tpr, tnr = best
    return threshold, {
        "balanced_accuracy": float(0.5 * (tpr + tnr)),
        "screw_recall": float(tpr),
        "empty_recall": float(tnr),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def train_one(
    role: str,
    train_samples: list[Sample],
    val_samples: list[Sample],
    weights_path: Path,
    output_dir: Path,
    device: torch.device,
    input_size: int,
    batch_size: int,
    epochs: int,
    freeze_epochs: int,
    lr: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_local_resnet18(weights_path, device)
    train_loader = DataLoader(RoiDataset(train_samples, input_size, True), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(RoiDataset(val_samples, input_size, False), batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_samples, device))

    history = []
    best_score = -1.0
    best_state = None
    best_threshold = 0.5
    best_metrics: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        phase = "head" if epoch <= freeze_epochs else "layer4"
        set_trainable(model, phase)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr if phase == "head" else lr * 0.25, weight_decay=1e-4)

        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * int(y.numel())
            train_count += int(y.numel())

        val_loss, p_screw, y_true = evaluate(model, val_loader, device)
        threshold, metrics = choose_threshold(p_screw, y_true)
        score = float(metrics["balanced_accuracy"])
        history.append({
            "epoch": epoch,
            "phase": phase,
            "train_loss": train_loss_sum / max(1, train_count),
            "val_loss": val_loss,
            "threshold": threshold,
            **metrics,
        })
        print(
            f"{role} epoch {epoch:02d}/{epochs} {phase}: "
            f"loss={history[-1]['train_loss']:.4f} val={val_loss:.4f} "
            f"bal_acc={score:.3f} thr={threshold:.3f} "
            f"screw_recall={metrics['screw_recall']:.3f} empty_recall={metrics['empty_recall']:.3f}"
        )
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")

    checkpoint = {
        "architecture": "resnet18",
        "classes": list(CLASSES),
        "role": role,
        "input_size": int(input_size),
        "threshold_screw": float(best_threshold),
        "state_dict": best_state,
        "preprocess": {"pad_to_square": True, "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "training": {
            "ng_source_images": 0,
            "test_source_images": 0,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "best_balanced_accuracy": float(best_score),
            "best_metrics": best_metrics,
        },
    }
    torch.save(checkpoint, output_dir / "best.pt")

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    (output_dir / "summary.json").write_text(json.dumps(checkpoint["training"] | {"threshold_screw": best_threshold, "role": role}, indent=2), encoding="utf-8")
    return checkpoint


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Train two independent ResNet18 semantic models using ONLY GOOD images. "
            "S ROIs provide screw examples; E ROIs provide empty examples. "
            "No NG/test images are used."
        )
    )
    p.add_argument("--good-dir", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--output", default="artifacts/resnet18_se_proxy")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--freeze-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--foreground-threshold", type=int, default=238)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    good_dir = Path(args.good_dir).resolve()
    reference_path = Path(args.reference).resolve()
    config_path = Path(args.config).resolve()
    weights_path = Path(args.weights).resolve()
    output = Path(args.output).resolve()

    images = collect_images(good_dir)
    if len(images) < 10:
        raise SystemExit(f"Need at least 10 GOOD images, found {len(images)}")
    reference = read_image(reference_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    slots = [x for x in config.get("screw_slots", []) if bool(x.get("enabled", True))]
    s_slots = [x for x in slots if str(x.get("id", "")).upper().startswith("S")]
    e_slots = [x for x in slots if str(x.get("id", "")).upper().startswith("E")]
    if not s_slots or not e_slots:
        raise SystemExit("Config must contain both S and E ROIs")

    align_cfg = ProductLocatorConfig(foreground_threshold=args.foreground_threshold)
    by_source: dict[str, list[Sample]] = {}
    print("=== GOOD-only semantic crop extraction ===")
    print(f"GOOD images: {len(images)} | S ROIs: {[x['id'] for x in s_slots]} | E ROIs: {[x['id'] for x in e_slots]}")
    print("NG/test source images used: 0")

    for i, path in enumerate(images, 1):
        try:
            raw = read_image(path)
            result = align_to_reference(raw, reference, align_cfg)
            aligned = result.aligned
            h, w = aligned.shape[:2]
            samples: list[Sample] = []
            for item in s_slots:
                roi = item["roi"]
                if not validate_roi(roi, w, h):
                    raise RuntimeError(f"invalid ROI {item['id']}: {roi}")
                samples.append(Sample(path.name, str(item["id"]), crop_roi(aligned, roi), 1))
            for item in e_slots:
                roi = item["roi"]
                if not validate_roi(roi, w, h):
                    raise RuntimeError(f"invalid ROI {item['id']}: {roi}")
                samples.append(Sample(path.name, str(item["id"]), crop_roi(aligned, roi), 0))
            by_source[path.name] = samples
            print(f"[{i}/{len(images)}] {path.name} -> OK ({result.method}, ecc={result.ecc_score})")
        except Exception as exc:
            print(f"[{i}/{len(images)}] {path.name} -> SKIP: {exc}")

    sources = sorted(by_source)
    rng = random.Random(args.seed)
    rng.shuffle(sources)
    val_count = max(2, int(round(len(sources) * args.val_ratio)))
    val_sources = set(sources[:val_count])
    train_sources = set(sources[val_count:])

    train_all = [s for src in train_sources for s in by_source[src]]
    val_all = [s for src in val_sources for s in by_source[src]]

    # S model: balanced semantic task, exactly 2 screw and 2 empty proxy crops per source.
    def balanced_s(samples: list[Sample]) -> list[Sample]:
        grouped: dict[str, list[Sample]] = {}
        for s in samples:
            grouped.setdefault(s.source, []).append(s)
        out: list[Sample] = []
        for src, group in grouped.items():
            screws = [s for s in group if s.label == 1]
            empties = [s for s in group if s.label == 0]
            local = random.Random(f"{args.seed}:{src}")
            local.shuffle(empties)
            out.extend(screws)
            out.extend(empties[:len(screws)])
        return out

    s_train = balanced_s(train_all)
    s_val = balanced_s(val_all)
    e_train = train_all[:]  # E model sees all empty positions plus real screw proxies from S.
    e_val = val_all[:]

    output.mkdir(parents=True, exist_ok=True)
    split_info = {
        "train_sources": sorted(train_sources),
        "val_sources": sorted(val_sources),
        "ng_source_images": 0,
        "test_source_images": 0,
    }
    (output / "split.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    print("\n=== Train independent S model ===")
    s_ckpt = train_one("S_presence", s_train, s_val, weights_path, output / "S", device, args.input_size, args.batch_size, args.epochs, args.freeze_epochs, args.lr)
    print("\n=== Train independent E model ===")
    e_ckpt = train_one("E_intrusion", e_train, e_val, weights_path, output / "E", device, args.input_size, args.batch_size, args.epochs, args.freeze_epochs, args.lr)

    summary = {
        "model_type": "resnet18_good_semantic_proxy",
        "good_source_images": len(by_source),
        "ng_source_images": 0,
        "test_source_images": 0,
        "S_checkpoint": str(output / "S" / "best.pt"),
        "E_checkpoint": str(output / "E" / "best.pt"),
        "S_threshold_screw": float(s_ckpt["threshold_screw"]),
        "E_threshold_screw": float(e_ckpt["threshold_screw"]),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== Training complete ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
