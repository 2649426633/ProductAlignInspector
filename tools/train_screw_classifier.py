from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.screw_classifier import (
    DEFAULT_CLASSES,
    build_model,
    build_transforms,
    save_checkpoint,
    set_backbone_trainable,
)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    class_name: str
    source: str


class ScrewDataset(Dataset):
    def __init__(self, samples: list[Sample], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, sample.label


def _load_samples(dataset_dir: Path, class_names: list[str]) -> list[Sample]:
    class_to_index = {name: i for i, name in enumerate(class_names)}
    manifest_path = dataset_dir / "manifest.csv"
    samples: list[Sample] = []
    seen: set[str] = set()

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ok" or row.get("kind") != "screw_slot":
                    continue
                class_name = str(row.get("label", "")).strip()
                if class_name not in class_to_index:
                    continue
                crop_value = str(row.get("crop", "")).strip()
                if not crop_value:
                    continue
                crop_path = Path(crop_value)
                if not crop_path.is_absolute():
                    candidate = dataset_dir / crop_path
                    if candidate.exists():
                        crop_path = candidate
                if not crop_path.exists():
                    candidate = REPO_ROOT / crop_value
                    if candidate.exists():
                        crop_path = candidate
                if not crop_path.exists():
                    continue
                key = str(crop_path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                source = str(row.get("source", "")).strip() or crop_path.stem.split("__", 1)[0]
                samples.append(
                    Sample(
                        path=crop_path,
                        label=class_to_index[class_name],
                        class_name=class_name,
                        source=source,
                    )
                )

    if not samples:
        for class_name in class_names:
            class_dir = dataset_dir / "screw" / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.rglob("*")):
                if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                    continue
                source = path.stem.split("__", 1)[0]
                samples.append(
                    Sample(
                        path=path,
                        label=class_to_index[class_name],
                        class_name=class_name,
                        source=source,
                    )
                )

    return samples


def _has_all_classes(samples: list[Sample], num_classes: int) -> bool:
    return {s.label for s in samples} == set(range(num_classes))


def _stratified_sample_split(
    samples: list[Sample],
    val_ratio: float,
    seed: int,
    num_classes: int,
) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(seed)
    train: list[Sample] = []
    val: list[Sample] = []

    for class_id in range(num_classes):
        items = [s for s in samples if s.label == class_id]
        rng.shuffle(items)
        if len(items) < 2:
            raise RuntimeError(
                f"Class {class_id} has only {len(items)} sample(s). "
                "At least 2 per class are required for train/validation."
            )
        val_count = max(1, int(round(len(items) * val_ratio)))
        val_count = min(val_count, len(items) - 1)
        val.extend(items[:val_count])
        train.extend(items[val_count:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _grouped_split(
    samples: list[Sample],
    val_ratio: float,
    seed: int,
    num_classes: int,
) -> tuple[list[Sample], list[Sample], str]:
    groups: dict[str, list[Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.source, []).append(sample)

    group_names = list(groups)
    if len(group_names) < 2:
        train, val = _stratified_sample_split(samples, val_ratio, seed, num_classes)
        return train, val, "sample_fallback_single_source"

    val_count = max(1, int(round(len(group_names) * val_ratio)))
    val_count = min(val_count, len(group_names) - 1)

    for attempt in range(200):
        rng = random.Random(seed + attempt)
        shuffled = group_names[:]
        rng.shuffle(shuffled)
        val_groups = set(shuffled[:val_count])
        train = [s for s in samples if s.source not in val_groups]
        val = [s for s in samples if s.source in val_groups]
        if _has_all_classes(train, num_classes) and _has_all_classes(val, num_classes):
            return train, val, "grouped_by_source"

    train, val = _stratified_sample_split(samples, val_ratio, seed, num_classes)
    return train, val, "sample_fallback_unbalanced_groups"


def _class_counts(samples: list[Sample], class_names: list[str]) -> dict[str, int]:
    counts = {name: 0 for name in class_names}
    for sample in samples:
        counts[sample.class_name] += 1
    return counts


def _class_weights(samples: list[Sample], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = [sum(1 for s in samples if s.label == i) for i in range(num_classes)]
    total = sum(counts)
    weights = []
    for count in counts:
        if count <= 0:
            raise RuntimeError("A training class has zero samples.")
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _metrics_from_confusion(confusion: list[list[int]]) -> dict[str, float]:
    num_classes = len(confusion)
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[i][i] for i in range(num_classes))
    accuracy = correct / total if total else 0.0

    f1_values: list[float] = []
    result: dict[str, float] = {"accuracy": accuracy}
    for i in range(num_classes):
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(num_classes) if r != i)
        fn = sum(confusion[i][c] for c in range(num_classes) if c != i)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[f"class_{i}_precision"] = precision
        result[f"class_{i}_recall"] = recall
        result[f"class_{i}_f1"] = f1
        f1_values.append(f1)

    result["macro_f1"] = sum(f1_values) / len(f1_values) if f1_values else 0.0
    return result


@torch.no_grad()
def _evaluate(model, loader, criterion, device: torch.device, num_classes: int):
    model.eval()
    total_loss = 0.0
    total_items = 0
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

        predictions = torch.argmax(logits, dim=1)
        for truth, pred in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
            confusion[int(truth)][int(pred)] += 1

    metrics = _metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / max(1, total_items)
    return metrics, confusion


def _make_optimizer(model: nn.Module, head_lr: float, backbone_lr: float, weight_decay: float, frozen: bool):
    if frozen:
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=head_lr,
            weight_decay=weight_decay,
        )

    return torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": backbone_lr},
            {"params": model.classifier.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small-sample screw/empty ROI classifier.")
    parser.add_argument("--dataset", default="artifacts/roi_dataset", help="ROI dataset directory")
    parser.add_argument("--output", default="artifacts/screw_classifier", help="Training output directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze-epochs", type=int, default=5, help="Train classifier head only for first N epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--backbone-lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, help="Keep 0 on Windows unless you need more loader workers")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-pretrained", action="store_true", help="Do not use ImageNet pretrained weights")
    args = parser.parse_args()

    if not 0.05 <= args.val_ratio <= 0.5:
        raise SystemExit("--val-ratio should be between 0.05 and 0.5")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False.")
    use_cuda = (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset_dir = Path(args.dataset)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = list(DEFAULT_CLASSES)
    samples = _load_samples(dataset_dir, class_names)
    if not samples:
        raise SystemExit(
            f"No screw ROI samples found under {dataset_dir}. "
            "Run tools\\extract_roi_dataset.py first."
        )

    total_counts = _class_counts(samples, class_names)
    missing = [name for name, count in total_counts.items() if count == 0]
    if missing:
        raise SystemExit(f"Missing training class(es): {', '.join(missing)}")

    train_samples, val_samples, split_method = _grouped_split(
        samples,
        args.val_ratio,
        args.seed,
        len(class_names),
    )

    train_counts = _class_counts(train_samples, class_names)
    val_counts = _class_counts(val_samples, class_names)

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"All samples: {len(samples)} {total_counts}")
    print(f"Train: {len(train_samples)} {train_counts}")
    print(f"Val: {len(val_samples)} {val_counts}")
    print(f"Split: {split_method}")
    if split_method != "grouped_by_source":
        print("WARNING: validation is not source-isolated; accuracy may look better than real production accuracy.")
    if min(total_counts.values()) < 10:
        print("WARNING: one class has fewer than 10 real crops. Treat this run as a baseline, not final validation.")

    train_dataset = ScrewDataset(train_samples, build_transforms(args.input_size, train=True))
    val_dataset = ScrewDataset(val_samples, build_transforms(args.input_size, train=False))

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    try:
        model = build_model(num_classes=len(class_names), pretrained=not args.no_pretrained)
    except Exception as exc:
        if args.no_pretrained:
            raise
        raise RuntimeError(
            "Could not load ImageNet pretrained MobileNetV3-Small weights. "
            "Check internet/cache, or rerun with --no-pretrained (not recommended for very small datasets)."
        ) from exc

    model.to(device)
    class_weight_tensor = _class_weights(train_samples, len(class_names), device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)

    frozen = args.freeze_epochs > 0
    set_backbone_trainable(model, not frozen)
    optimizer = _make_optimizer(model, args.head_lr, args.backbone_lr, args.weight_decay, frozen=frozen)

    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        if frozen and epoch == args.freeze_epochs + 1:
            set_backbone_trainable(model, True)
            frozen = False
            optimizer = _make_optimizer(
                model,
                args.head_lr * 0.6,
                args.backbone_lr,
                args.weight_decay,
                frozen=False,
            )
            print("Backbone unfrozen: fine-tuning all MobileNetV3 layers.")

        model.train()
        running_loss = 0.0
        running_items = 0
        train_confusion = [[0 for _ in class_names] for _ in class_names]

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            running_items += batch_size

            predictions = torch.argmax(logits.detach(), dim=1)
            for truth, pred in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
                train_confusion[int(truth)][int(pred)] += 1

        train_metrics = _metrics_from_confusion(train_confusion)
        train_metrics["loss"] = running_loss / max(1, running_items)
        val_metrics, val_confusion = _evaluate(model, val_loader, criterion, device, len(class_names))

        row: dict[str, object] = {
            "epoch": epoch,
            "frozen_backbone": frozen,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        current_f1 = float(val_metrics["macro_f1"])
        if current_f1 > best_f1 + 1e-6:
            best_f1 = current_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                epoch=epoch,
                class_names=class_names,
                input_size=args.input_size,
                metrics={k: float(v) for k, v in val_metrics.items()},
                extra={
                    "split_method": split_method,
                    "train_counts": train_counts,
                    "val_counts": val_counts,
                    "val_confusion": val_confusion,
                    "pretrained": not args.no_pretrained,
                },
            )
        else:
            epochs_without_improvement += 1

        if epoch > args.freeze_epochs and epochs_without_improvement >= args.patience:
            print(f"Early stopping: no val macro-F1 improvement for {args.patience} epochs.")
            break

    history_path = out_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "architecture": "mobilenet_v3_small",
        "classes": class_names,
        "input_size": args.input_size,
        "device": str(device),
        "sample_counts": total_counts,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "split_method": split_method,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "best_checkpoint": str(best_path),
        "history": str(history_path),
    }
    (out_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("")
    print(f"Best checkpoint: {best_path}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val macro-F1: {best_f1:.4f}")
    print(f"History: {history_path}")
    print("Next: export the checkpoint with tools\\export_screw_onnx.py")


if __name__ == "__main__":
    main()
