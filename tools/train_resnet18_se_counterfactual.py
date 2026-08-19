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
from product_align_inspector.io_utils import read_image, write_image
from product_align_inspector.roi import crop_roi, validate_roi

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASSES = ("empty", "screw")


@dataclass
class CropSample:
    source: str
    roi_id: str
    image_bgr: np.ndarray
    label: int  # 0=empty, 1=screw
    kind: str


class RoiDataset(Dataset):
    def __init__(self, samples: list[CropSample], input_size: int, train: bool) -> None:
        self.samples = samples
        if train:
            self.tf = transforms.Compose([
                transforms.Lambda(self._pad_square),
                transforms.Resize((input_size, input_size)),
                transforms.RandomAffine(
                    degrees=4.0,
                    translate=(0.035, 0.035),
                    scale=(0.96, 1.04),
                    fill=255,
                ),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.10
                ),
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
        sample = self.samples[index]
        rgb = cv2.cvtColor(sample.image_bgr, cv2.COLOR_BGR2RGB)
        return self.tf(Image.fromarray(rgb)), int(sample.label)


def collect_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def load_resnet18(weights_path: Path, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def set_trainable(model: nn.Module, phase: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if phase == "head":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    elif phase == "layer4":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(phase)


def class_weights(samples: list[CropSample], device: torch.device) -> torch.Tensor:
    counts = np.bincount([s.label for s in samples], minlength=2).astype(np.float32)
    total = float(counts.sum())
    weights = total / np.maximum(1.0, 2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            losses.append(float(criterion(logits, y).item()) * int(y.numel()))
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            labels.append(y.cpu().numpy())
    if not labels:
        return 0.0, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
    y_true = np.concatenate(labels)
    p_screw = np.concatenate(probs)
    return sum(losses) / max(1, len(y_true)), p_screw, y_true


def choose_threshold(p_screw: np.ndarray, labels: np.ndarray):
    best = None
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (p_screw >= threshold).astype(np.int64)
        tp = int(((pred == 1) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        tpr = tp / max(1, tp + fn)
        tnr = tn / max(1, tn + fp)
        balanced = 0.5 * (tpr + tnr)
        key = (balanced, min(tpr, tnr), -abs(float(threshold) - 0.5))
        if best is None or key > best[0]:
            best = (key, float(threshold), tp, tn, fp, fn, tpr, tnr)
    assert best is not None
    _, threshold, tp, tn, fp, fn, tpr, tnr = best
    return threshold, {
        "balanced_accuracy": float(0.5 * (tpr + tnr)),
        "screw_recall": float(tpr),
        "empty_recall": float(tnr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _soft_center_mask(height: int, width: int, radius_ratio: float, feather_ratio: float) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    axes = (
        max(3, int(round(width * radius_ratio))),
        max(3, int(round(height * radius_ratio))),
    )
    cv2.ellipse(mask, (width // 2, height // 2), axes, 0.0, 0.0, 360.0, 255, -1)
    k = max(3, int(round(min(height, width) * feather_ratio)))
    if k % 2 == 0:
        k += 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask.astype(np.float32) / 255.0


def _resize_donor(donor: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    return cv2.resize(donor, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def counterfactual_paste(
    target: np.ndarray,
    donor: np.ndarray,
    *,
    radius_ratio: float,
    feather_ratio: float,
) -> np.ndarray:
    """Replace only the central semantic core while preserving the target ROI context.

    S-missing synthesis: target=S screw ROI, donor=E empty ROI.
    E-extra synthesis: target=E empty ROI, donor=S screw ROI.
    """
    h, w = target.shape[:2]
    donor_rs = _resize_donor(donor, (h, w)).astype(np.float32)
    target_f = target.astype(np.float32)
    alpha = _soft_center_mask(h, w, radius_ratio, feather_ratio)[..., None]
    blended = target_f * (1.0 - alpha) + donor_rs * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def train_one(
    role: str,
    train_samples: list[CropSample],
    val_samples: list[CropSample],
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
    model = load_resnet18(weights_path, device)
    train_loader = DataLoader(
        RoiDataset(train_samples, input_size, True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        RoiDataset(val_samples, input_size, False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_samples, device))

    history: list[dict[str, object]] = []
    best_score = -1.0
    best_state = None
    best_threshold = 0.5
    best_metrics: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        phase = "head" if epoch <= freeze_epochs else "layer4"
        set_trainable(model, phase)
        phase_lr = lr if phase == "head" else lr * 0.20
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=phase_lr,
            weight_decay=1e-4,
        )

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
        row = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": train_loss_sum / max(1, train_count),
            "val_loss": val_loss,
            "threshold": threshold,
            **metrics,
        }
        history.append(row)
        print(
            f"{role} epoch {epoch:02d}/{epochs} {phase}: "
            f"loss={row['train_loss']:.4f} val={val_loss:.4f} "
            f"bal_acc={score:.3f} thr={threshold:.3f} "
            f"screw_recall={metrics['screw_recall']:.3f} empty_recall={metrics['empty_recall']:.3f}"
        )
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")

    checkpoint = {
        "architecture": "resnet18",
        "classes": list(CLASSES),
        "role": role,
        "model_type": "resnet18_same_roi_counterfactual",
        "input_size": int(input_size),
        "threshold_screw": float(best_threshold),
        "state_dict": best_state,
        "preprocess": {
            "pad_to_square": True,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "training": {
            "ng_source_images": 0,
            "test_source_images": 0,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "best_balanced_accuracy": float(best_score),
            "best_metrics": best_metrics,
            "note": "Both classes are generated at the SAME target ROI context to prevent learning S-vs-E location/background shortcuts.",
        },
    }
    torch.save(checkpoint, output_dir / "best.pt")

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    (output_dir / "summary.json").write_text(
        json.dumps(
            checkpoint["training"]
            | {"threshold_screw": best_threshold, "role": role, "model_type": checkpoint["model_type"]},
            indent=2,
        ),
        encoding="utf-8",
    )
    return checkpoint


def _save_debug_samples(output: Path, role: str, samples: list[CropSample], limit_per_kind: int = 12) -> None:
    root = output / "debug_counterfactuals" / role
    counts: dict[str, int] = {}
    for sample in samples:
        count = counts.get(sample.kind, 0)
        if count >= limit_per_kind:
            continue
        filename = f"{sample.kind}_{count+1:02d}_{sample.source}_{sample.roi_id}.png"
        write_image(root / filename, sample.image_bgr)
        counts[sample.kind] = count + 1


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Train independent S and E ResNet18 models from GOOD images only, using same-ROI "
            "counterfactual synthesis so the network cannot solve the task by memorizing ROI location/background."
        )
    )
    p.add_argument("--good-dir", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--output", default="artifacts/resnet18_se_counterfactual")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--freeze-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--val-ratio", type=float, default=0.20)
    p.add_argument("--variants", type=int, default=3, help="Synthetic counterfactuals per target ROI")
    p.add_argument("--core-radius", type=float, default=0.34)
    p.add_argument("--feather", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--foreground-threshold", type=int, default=238)
    args = p.parse_args()

    if args.variants < 1:
        raise SystemExit("--variants must be >= 1")
    if not 0.20 <= args.core_radius <= 0.48:
        raise SystemExit("--core-radius must be between 0.20 and 0.48")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    good_dir = Path(args.good_dir).resolve()
    reference_path = Path(args.reference).resolve()
    config_path = Path(args.config).resolve()
    weights_path = Path(args.weights).resolve()
    output = Path(args.output).resolve()

    if not weights_path.is_file():
        raise SystemExit(f"Weights not found: {weights_path}")
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
    by_source: dict[str, dict[str, list[tuple[str, np.ndarray]]]] = {}

    print("=== GOOD-only same-ROI counterfactual extraction ===")
    print(f"GOOD images: {len(images)}")
    print(f"S ROIs: {[x['id'] for x in s_slots]}")
    print(f"E ROIs: {[x['id'] for x in e_slots]}")
    print("NG/test source images used: 0")

    for index, path in enumerate(images, 1):
        try:
            raw = read_image(path)
            result = align_to_reference(raw, reference, align_cfg)
            if result.feature_matrix is None:
                raise RuntimeError(
                    f"alignment method {result.method} has no feature_matrix; excluded from semantic training"
                )
            aligned = result.aligned
            h, w = aligned.shape[:2]
            s_crops: list[tuple[str, np.ndarray]] = []
            e_crops: list[tuple[str, np.ndarray]] = []
            for item in s_slots:
                roi = item["roi"]
                if not validate_roi(roi, w, h):
                    raise RuntimeError(f"invalid ROI {item['id']}: {roi}")
                s_crops.append((str(item["id"]), crop_roi(aligned, roi)))
            for item in e_slots:
                roi = item["roi"]
                if not validate_roi(roi, w, h):
                    raise RuntimeError(f"invalid ROI {item['id']}: {roi}")
                e_crops.append((str(item["id"]), crop_roi(aligned, roi)))
            by_source[path.name] = {"S": s_crops, "E": e_crops}
            print(
                f"[{index}/{len(images)}] {path.name} -> OK "
                f"({result.method}, ecc={result.ecc_score})"
            )
        except Exception as exc:
            print(f"[{index}/{len(images)}] {path.name} -> SKIP: {exc}")

    sources = sorted(by_source)
    if len(sources) < 10:
        raise SystemExit(f"Only {len(sources)} GOOD images survived alignment")
    rng = random.Random(args.seed)
    rng.shuffle(sources)
    val_count = max(2, int(round(len(sources) * args.val_ratio)))
    val_sources = set(sources[:val_count])
    train_sources = set(sources[val_count:])

    def synthesize(source_names: set[str]) -> tuple[list[CropSample], list[CropSample]]:
        s_samples: list[CropSample] = []
        e_samples: list[CropSample] = []
        for source in sorted(source_names):
            groups = by_source[source]
            screws = groups["S"]
            empties = groups["E"]
            local = random.Random(f"{args.seed}:{source}")

            # S model: SAME S target context for both classes.
            # Original S = screw. Replace only center with a real E empty core = synthetic missing.
            for s_id, s_crop in screws:
                s_samples.append(CropSample(source, s_id, s_crop.copy(), 1, "screw_real"))
                donor_order = list(range(len(empties)))
                local.shuffle(donor_order)
                for variant in range(args.variants):
                    _, donor = empties[donor_order[variant % len(donor_order)]]
                    fake_missing = counterfactual_paste(
                        s_crop,
                        donor,
                        radius_ratio=args.core_radius,
                        feather_ratio=args.feather,
                    )
                    s_samples.append(
                        CropSample(source, s_id, fake_missing, 0, "missing_synthetic")
                    )

            # E model: SAME E target context for both classes.
            # Original E = empty. Replace only center with a real S screw core = synthetic extra screw.
            for e_id, e_crop in empties:
                e_samples.append(CropSample(source, e_id, e_crop.copy(), 0, "empty_real"))
                donor_order = list(range(len(screws)))
                local.shuffle(donor_order)
                for variant in range(args.variants):
                    _, donor = screws[donor_order[variant % len(donor_order)]]
                    fake_extra = counterfactual_paste(
                        e_crop,
                        donor,
                        radius_ratio=args.core_radius,
                        feather_ratio=args.feather,
                    )
                    e_samples.append(
                        CropSample(source, e_id, fake_extra, 1, "extra_screw_synthetic")
                    )
        return s_samples, e_samples

    s_train, e_train = synthesize(train_sources)
    s_val, e_val = synthesize(val_sources)

    output.mkdir(parents=True, exist_ok=True)
    split_info = {
        "train_sources": sorted(train_sources),
        "val_sources": sorted(val_sources),
        "ng_source_images": 0,
        "test_source_images": 0,
        "counterfactual_variants": int(args.variants),
        "core_radius": float(args.core_radius),
        "feather": float(args.feather),
    }
    (output / "split.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    _save_debug_samples(output, "S", s_train)
    _save_debug_samples(output, "E", e_train)

    print("")
    print("=== Dataset summary ===")
    print(f"Accepted GOOD sources: {len(sources)}/{len(images)}")
    print(f"Train/val sources: {len(train_sources)}/{len(val_sources)}")
    print(f"S samples train/val: {len(s_train)}/{len(s_val)}")
    print(f"E samples train/val: {len(e_train)}/{len(e_val)}")
    print(f"Debug counterfactuals: {output / 'debug_counterfactuals'}")
    print("IMPORTANT: inspect debug_counterfactuals before trusting training results.")

    print("\n=== Train independent S model ===")
    s_ckpt = train_one(
        "S_presence_same_roi",
        s_train,
        s_val,
        weights_path,
        output / "S",
        device,
        args.input_size,
        args.batch_size,
        args.epochs,
        args.freeze_epochs,
        args.lr,
    )
    print("\n=== Train independent E model ===")
    e_ckpt = train_one(
        "E_intrusion_same_roi",
        e_train,
        e_val,
        weights_path,
        output / "E",
        device,
        args.input_size,
        args.batch_size,
        args.epochs,
        args.freeze_epochs,
        args.lr,
    )

    summary = {
        "model_type": "resnet18_same_roi_counterfactual",
        "good_source_images": len(sources),
        "ng_source_images": 0,
        "test_source_images": 0,
        "S_checkpoint": str(output / "S" / "best.pt"),
        "E_checkpoint": str(output / "E" / "best.pt"),
        "S_threshold_screw": float(s_ckpt["threshold_screw"]),
        "E_threshold_screw": float(e_ckpt["threshold_screw"]),
        "debug_counterfactuals": str(output / "debug_counterfactuals"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== Training complete ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
