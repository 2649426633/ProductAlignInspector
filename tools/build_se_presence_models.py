from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.io_utils import read_image
from product_align_inspector.roi import config_reference_size, crop_roi, enabled_slots, roi_for_image
from product_align_inspector.se_presence import SemanticPreprocessConfig, preprocess_presence_crop

LABEL_MAP = {
    "screw": 1,
    "present": 1,
    "1": 1,
    "empty": 0,
    "absent": 0,
    "0": 0,
}
UNKNOWN_LABELS = {"", "?", "unknown", "na", "n/a", "skip", "-"}
VALID_SPLITS = {"train", "val", "test"}


@dataclass
class ManifestImage:
    path: Path
    source: str
    split: str
    labels: dict[str, int]


@dataclass
class LoadedSample:
    crop: np.ndarray
    label: int
    slot_id: str
    image_key: str
    source: str
    split: str


class TinyPresenceCNN(nn.Module):
    """Small shared binary classifier designed for CPU ONNX deployment."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 48),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def _parse_label(value: str) -> int | None:
    text = str(value or "").strip().lower()
    if text in UNKNOWN_LABELS:
        return None
    if text not in LABEL_MAP:
        raise ValueError(
            f"Unsupported manifest label {value!r}. Use screw/empty or ? for unknown."
        )
    return int(LABEL_MAP[text])


def _read_manifest(path: Path, slot_ids: list[str]) -> list[ManifestImage]:
    rows: list[ManifestImage] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "image" not in reader.fieldnames:
            raise RuntimeError("Manifest must contain an 'image' column.")
        missing = [slot_id for slot_id in slot_ids if slot_id not in reader.fieldnames]
        if missing:
            raise RuntimeError(f"Manifest is missing slot columns: {missing}")

        for line_no, row in enumerate(reader, 2):
            raw_image = str(row.get("image", "")).strip()
            if not raw_image:
                continue
            image_path = Path(raw_image)
            if not image_path.is_absolute():
                image_path = (path.parent / image_path).resolve()
            else:
                image_path = image_path.resolve()

            split = str(row.get("split", "")).strip().lower()
            if split and split not in VALID_SPLITS:
                raise RuntimeError(
                    f"Manifest line {line_no}: split must be train/val/test or blank, got {split!r}"
                )

            labels: dict[str, int] = {}
            for slot_id in slot_ids:
                label = _parse_label(str(row.get(slot_id, "")))
                if label is not None:
                    labels[slot_id] = label

            if not labels:
                continue
            rows.append(
                ManifestImage(
                    path=image_path,
                    source=str(row.get("source", "")).strip(),
                    split=split,
                    labels=labels,
                )
            )

    if not rows:
        raise RuntimeError(f"No labeled images found in manifest: {path}")
    return rows


def _coverage(rows: list[ManifestImage], slot_ids: list[str]) -> dict[str, Counter[int]]:
    result = {slot_id: Counter() for slot_id in slot_ids}
    for row in rows:
        for slot_id, label in row.labels.items():
            result[slot_id][int(label)] += 1
    return result


def _coverage_text(coverage: dict[str, Counter[int]]) -> str:
    parts = []
    for slot_id, counter in coverage.items():
        parts.append(
            f"{slot_id}: empty={counter.get(0, 0)}, screw={counter.get(1, 0)}"
        )
    return "; ".join(parts)


def _split_is_usable(
    train_rows: list[ManifestImage],
    val_rows: list[ManifestImage],
    slot_ids: list[str],
    *,
    min_val_per_class: int,
) -> bool:
    train_cov = _coverage(train_rows, slot_ids)
    val_cov = _coverage(val_rows, slot_ids)
    for slot_id in slot_ids:
        for label in (0, 1):
            if train_cov[slot_id].get(label, 0) < 1:
                return False
            if val_cov[slot_id].get(label, 0) < min_val_per_class:
                return False
    return True


def _assign_splits(
    rows: list[ManifestImage],
    slot_ids: list[str],
    *,
    val_ratio: float,
    seed: int,
    min_val_per_class: int,
    allow_incomplete: bool,
) -> list[ManifestImage]:
    explicit = [row for row in rows if row.split]
    blank = [row for row in rows if not row.split]
    if explicit and blank:
        raise RuntimeError(
            "Manifest mixes explicit and blank split values. Either fill split for every row "
            "or leave split blank for every row."
        )

    full_cov = _coverage(rows, slot_ids)
    missing_full = [
        (slot_id, label)
        for slot_id in slot_ids
        for label in (0, 1)
        if full_cov[slot_id].get(label, 0) == 0
    ]
    if missing_full and not allow_incomplete:
        detail = ", ".join(
            f"{slot_id}:{'screw' if label == 1 else 'empty'}"
            for slot_id, label in missing_full
        )
        raise RuntimeError(
            "Per-slot probability calibration requires both screw and empty examples for every "
            f"position. Missing labeled coverage: {detail}. Fill '?' cells in the manifest or "
            "add training data. Use --allow-incomplete-calibration only for preliminary testing."
        )

    if explicit:
        result = copy.deepcopy(rows)
        train_rows = [row for row in result if row.split == "train"]
        val_rows = [row for row in result if row.split == "val"]
        if not train_rows or not val_rows:
            raise RuntimeError("Explicit manifest splits require at least one train and one val image.")
        if not allow_incomplete and not _split_is_usable(
            train_rows,
            val_rows,
            slot_ids,
            min_val_per_class=min_val_per_class,
        ):
            raise RuntimeError(
                "Explicit train/val split does not provide both classes for every slot in both "
                "training and validation. Adjust the split or add labels."
            )
        return result

    n = len(rows)
    if n < 4:
        raise RuntimeError("At least four labeled source images are required for automatic splitting.")
    n_val = max(2, min(n - 2, int(round(n * float(val_ratio)))))

    rng = random.Random(seed)
    indices = list(range(n))
    best: tuple[int, list[int]] | None = None

    for _ in range(1000):
        rng.shuffle(indices)
        val_idx = set(indices[:n_val])
        train_rows = [rows[i] for i in range(n) if i not in val_idx]
        val_rows = [rows[i] for i in range(n) if i in val_idx]

        train_cov = _coverage(train_rows, slot_ids)
        val_cov = _coverage(val_rows, slot_ids)
        score = 0
        for slot_id in slot_ids:
            for label in (0, 1):
                score += int(train_cov[slot_id].get(label, 0) >= 1)
                score += int(val_cov[slot_id].get(label, 0) >= min_val_per_class)

        if best is None or score > best[0]:
            best = (score, sorted(val_idx))

        if allow_incomplete or _split_is_usable(
            train_rows,
            val_rows,
            slot_ids,
            min_val_per_class=min_val_per_class,
        ):
            result = copy.deepcopy(rows)
            for i, row in enumerate(result):
                row.split = "val" if i in val_idx else "train"
            return result

    if best is None:
        raise RuntimeError("Unable to create train/val split.")

    if not allow_incomplete:
        raise RuntimeError(
            "Could not create an image-level train/val split with enough screw/empty examples "
            "for every slot. Add more labeled samples or provide an explicit split."
        )

    result = copy.deepcopy(rows)
    val_idx = set(best[1])
    for i, row in enumerate(result):
        row.split = "val" if i in val_idx else "train"
    return result


def _augment_crop(crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Photometric + small X/Y jitter only. No scale or rotation augmentation."""

    image = np.asarray(crop, dtype=np.uint8).copy()

    gamma = float(rng.uniform(0.72, 1.38))
    x = image.astype(np.float32) / 255.0
    image = np.clip(np.power(x, gamma) * 255.0, 0.0, 255.0).astype(np.uint8)

    gain = float(rng.uniform(0.82, 1.18))
    offset = float(rng.uniform(-18.0, 18.0))
    image = np.clip(image.astype(np.float32) * gain + offset, 0.0, 255.0).astype(np.uint8)

    if rng.random() < 0.70:
        h, w = image.shape[:2]
        axis = int(rng.integers(0, 2))
        strength = float(rng.uniform(-0.22, 0.22))
        if axis == 0:
            ramp = np.linspace(1.0 - strength, 1.0 + strength, h, dtype=np.float32)[:, None]
            ramp = np.repeat(ramp, w, axis=1)
        else:
            ramp = np.linspace(1.0 - strength, 1.0 + strength, w, dtype=np.float32)[None, :]
            ramp = np.repeat(ramp, h, axis=0)
        if image.ndim == 3:
            ramp = ramp[..., None]
        image = np.clip(image.astype(np.float32) * ramp, 0.0, 255.0).astype(np.uint8)

    if rng.random() < 0.70:
        h, w = image.shape[:2]
        max_dx = max(1, int(round(w * 0.04)))
        max_dy = max(1, int(round(h * 0.04)))
        dx = int(rng.integers(-max_dx, max_dx + 1))
        dy = int(rng.integers(-max_dy, max_dy + 1))
        matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        image = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    return image


class PresenceDataset(Dataset):
    def __init__(
        self,
        samples: list[LoadedSample],
        preprocess_cfg: SemanticPreprocessConfig,
        *,
        augment: bool,
        seed: int,
    ) -> None:
        self.samples = samples
        self.preprocess_cfg = preprocess_cfg
        self.augment = bool(augment)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[index]
        crop = sample.crop
        if self.augment:
            worker = torch.utils.data.get_worker_info()
            worker_id = 0 if worker is None else int(worker.id)
            rng = np.random.default_rng(
                self.seed + index * 1009 + worker_id * 1000003 + random.randint(0, 2**16)
            )
            crop = _augment_crop(crop, rng)

        tensor = preprocess_presence_crop(crop, self.preprocess_cfg)[0]
        x = torch.from_numpy(tensor)
        y = torch.tensor(float(sample.label), dtype=torch.float32)
        return x, y, sample.slot_id


def _load_samples(
    rows: list[ManifestImage],
    reference: np.ndarray,
    config: dict[str, Any],
    align_cfg: ProductLocatorConfig,
) -> tuple[list[LoadedSample], list[dict[str, str]]]:
    h, w = reference.shape[:2]
    slot_lookup = {str(slot["id"]): slot for slot in enabled_slots(config)}
    samples: list[LoadedSample] = []
    skipped: list[dict[str, str]] = []

    print("=== Align and extract labeled ROI samples ===")
    for index, row in enumerate(rows, 1):
        try:
            raw = read_image(row.path)
            alignment = align_to_reference(raw, reference, align_cfg)
            canonical = alignment.aligned
            for slot_id, label in row.labels.items():
                slot = slot_lookup.get(slot_id)
                if slot is None:
                    raise RuntimeError(f"Manifest references unknown slot {slot_id}")
                roi = roi_for_image(slot["roi"], config, w, h)
                crop = crop_roi(canonical, roi)
                samples.append(
                    LoadedSample(
                        crop=crop,
                        label=int(label),
                        slot_id=slot_id,
                        image_key=str(row.path),
                        source=row.source,
                        split=row.split,
                    )
                )
            print(
                f"[{index}/{len(rows)}] {row.path.name} -> OK | split={row.split} "
                f"| labels={len(row.labels)} | ecc={alignment.ecc_score:.4f}"
            )
        except Exception as exc:
            skipped.append({"path": str(row.path), "error": str(exc)})
            print(f"[{index}/{len(rows)}] {row.path.name} -> SKIP | {exc}")

    return samples, skipped


def _sample_weights(samples: list[LoadedSample]) -> list[float]:
    counts = Counter((sample.slot_id, sample.label) for sample in samples)
    return [1.0 / float(counts[(sample.slot_id, sample.label)]) for sample in samples]


def _balanced_accuracy(labels: np.ndarray, probs: np.ndarray, threshold: float) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    pred = (np.asarray(probs, dtype=np.float32) >= float(threshold)).astype(np.int32)
    screw = labels == 1
    empty = labels == 0
    tpr = float(np.mean(pred[screw] == 1)) if np.any(screw) else 0.0
    tnr = float(np.mean(pred[empty] == 0)) if np.any(empty) else 0.0
    return 0.5 * (tpr + tnr)


def _candidate_thresholds(probs: np.ndarray) -> np.ndarray:
    values = np.sort(np.unique(np.clip(np.asarray(probs, dtype=np.float32), 0.0, 1.0)))
    if values.size == 0:
        return np.asarray([0.5], dtype=np.float32)
    if values.size == 1:
        return np.asarray([0.0, float(values[0]), 1.0], dtype=np.float32)
    mids = (values[:-1] + values[1:]) * 0.5
    return np.unique(np.concatenate([[0.0], values, mids, [1.0]])).astype(np.float32)


def _slot_operating_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    expected: str,
) -> dict[str, float]:
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    pred_screw = probs >= float(threshold)

    screw = labels == 1
    empty = labels == 0
    screw_recall = float(np.mean(pred_screw[screw])) if np.any(screw) else 0.0
    empty_recall = float(np.mean(~pred_screw[empty])) if np.any(empty) else 0.0
    balanced = 0.5 * (screw_recall + empty_recall)

    if expected == "screw":
        normal_pass = screw_recall
        defect_recall = empty_recall
    else:
        normal_pass = empty_recall
        defect_recall = screw_recall

    return {
        "balanced_accuracy": balanced,
        "screw_recall": screw_recall,
        "empty_recall": empty_recall,
        "normal_pass_rate": normal_pass,
        "defect_recall": defect_recall,
    }


def _choose_best(
    candidates: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    expected: str,
    *,
    mode: str,
) -> tuple[float, dict[str, float]]:
    scored: list[tuple[float, dict[str, float]]] = [
        (float(t), _slot_operating_metrics(probs, labels, float(t), expected))
        for t in candidates
    ]

    if mode == "recommended":
        key = lambda item: (
            item[1]["balanced_accuracy"],
            item[1]["defect_recall"],
            item[1]["normal_pass_rate"],
            -abs(item[0] - 0.5),
        )
        return max(scored, key=key)

    if mode == "strict":
        eligible = [item for item in scored if item[1]["defect_recall"] >= 0.98]
        pool = eligible or scored
        key = lambda item: (
            item[1]["defect_recall"],
            item[1]["normal_pass_rate"],
            item[1]["balanced_accuracy"],
            -abs(item[0] - 0.5),
        )
        return max(pool, key=key)

    if mode == "loose":
        eligible = [item for item in scored if item[1]["normal_pass_rate"] >= 0.98]
        pool = eligible or scored
        key = lambda item: (
            item[1]["normal_pass_rate"],
            item[1]["defect_recall"],
            item[1]["balanced_accuracy"],
            -abs(item[0] - 0.5),
        )
        return max(pool, key=key)

    raise ValueError(mode)


def _calibrate_slot(
    probs: np.ndarray,
    labels: np.ndarray,
    expected: str,
) -> dict[str, Any]:
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    empty_probs = probs[labels == 0]
    screw_probs = probs[labels == 1]
    if empty_probs.size == 0 or screw_probs.size == 0:
        raise RuntimeError("Slot calibration requires both empty and screw validation examples.")

    candidates = _candidate_thresholds(probs)
    suggested: dict[str, float] = {}
    metrics: dict[str, dict[str, float]] = {}
    for profile in ("strict", "recommended", "loose"):
        threshold, row = _choose_best(
            candidates,
            probs,
            labels,
            expected,
            mode=profile,
        )
        suggested[profile] = float(threshold)
        metrics[profile] = row

    return {
        "validation_empty": int(empty_probs.size),
        "validation_screw": int(screw_probs.size),
        "empty_probability": {
            "min": float(empty_probs.min()),
            "median": float(np.median(empty_probs)),
            "max": float(empty_probs.max()),
        },
        "screw_probability": {
            "min": float(screw_probs.min()),
            "median": float(np.median(screw_probs)),
            "max": float(screw_probs.max()),
        },
        "separation_gap": float(screw_probs.min() - empty_probs.max()),
        "suggested_thresholds": suggested,
        "profile_metrics": metrics,
    }


@torch.no_grad()
def _predict(
    model: nn.Module,
    samples: list[LoadedSample],
    preprocess_cfg: SemanticPreprocessConfig,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    dataset = PresenceDataset(samples, preprocess_cfg, augment=False, seed=0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    slots: list[str] = []

    model.eval()
    for x, y, slot_ids in loader:
        x = x.to(device)
        logits = model(x).reshape(-1)
        batch_probs = torch.sigmoid(logits).cpu().numpy()
        probs.append(batch_probs.astype(np.float32))
        labels.append(y.numpy().astype(np.int32))
        slots.extend(list(slot_ids))

    return (
        np.concatenate(probs) if probs else np.empty((0,), dtype=np.float32),
        np.concatenate(labels) if labels else np.empty((0,), dtype=np.int32),
        slots,
    )


def _export_onnx(
    model: nn.Module,
    output_path: Path,
    input_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    dummy = torch.zeros(1, 1, input_size, input_size, dtype=torch.float32, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["logit"],
        dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_output = session.run(["logit"], {"input": dummy.detach().cpu().numpy()})[0]
    with torch.no_grad():
        torch_output = model(dummy).detach().cpu().numpy()
    max_abs_diff = float(np.max(np.abs(ort_output - torch_output)))
    if max_abs_diff > 1e-4:
        raise RuntimeError(f"ONNX parity check failed: max_abs_diff={max_abs_diff:.6g}")

    return {
        "input_name": "input",
        "output_name": "logit",
        "opset": 17,
        "max_abs_diff_vs_pytorch": max_abs_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train one shared small CNN for empty/screw semantics, export ONNX, and calibrate "
            "one independent P(screw) threshold for each S01/S02/E01..E09 position."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--center-crop-ratio", type=float, default=0.86)
    parser.add_argument("--ecc-accept", type=float, default=0.70)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.10)
    parser.add_argument("--min-val-per-class", type=int, default=1)
    parser.add_argument(
        "--threshold-profile",
        choices=["strict", "recommended", "loose"],
        default="recommended",
    )
    parser.add_argument(
        "--allow-incomplete-calibration",
        action="store_true",
        help=(
            "Development only. Allows slots without both validation classes to fall back to the "
            "global threshold. Do not use this for final production calibration."
        ),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    manifest_path = Path(args.manifest).resolve()
    reference_path = Path(args.reference).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    reference = read_image(reference_path)
    canonical_h, canonical_w = reference.shape[:2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    slots = enabled_slots(config)
    slot_ids = [str(slot["id"]) for slot in slots]
    expected_by_slot = {str(slot["id"]): str(slot.get("expected", "")).lower() for slot in slots}

    rows = _read_manifest(manifest_path, slot_ids)
    rows = _assign_splits(
        rows,
        slot_ids,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        min_val_per_class=max(1, int(args.min_val_per_class)),
        allow_incomplete=bool(args.allow_incomplete_calibration),
    )

    split_manifest = output / "resolved_manifest.csv"
    with split_manifest.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["image", "source", "split", *slot_ids]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {
                "image": str(row.path),
                "source": row.source,
                "split": row.split,
            }
            for slot_id in slot_ids:
                label = row.labels.get(slot_id)
                out[slot_id] = "" if label is None else ("screw" if label == 1 else "empty")
            writer.writerow(out)

    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "val"]
    test_rows = [row for row in rows if row.split == "test"]

    print("=== Shared semantic S/E CNN v7 ===")
    print(f"Manifest:          {manifest_path}")
    print(f"Canonical:         {canonical_w}x{canonical_h}")
    print(f"Slots:             {slot_ids}")
    print(f"Train images:      {len(train_rows)}")
    print(f"Val images:        {len(val_rows)}")
    print(f"Test images:       {len(test_rows)}")
    print(f"Device:            {device}")
    print("Geometry augment:  X/Y jitter only; NO scale; NO rotation")
    print("Task:              shared P(screw), per-slot probability threshold")
    print()
    print("Full labeled coverage:")
    print(_coverage_text(_coverage(rows, slot_ids)))
    print()

    align_cfg = ProductLocatorConfig(
        ecc_accept_score=float(args.ecc_accept),
        feature_min_inlier_ratio=float(args.min_inlier_ratio),
        canonical_scale=1.0,
    )
    samples, skipped = _load_samples(rows, reference, config, align_cfg)
    if skipped:
        print(f"Alignment skipped: {len(skipped)}")

    train_samples = [sample for sample in samples if sample.split == "train"]
    val_samples = [sample for sample in samples if sample.split == "val"]
    test_samples = [sample for sample in samples if sample.split == "test"]
    if not train_samples or not val_samples:
        raise RuntimeError("No train or validation ROI samples remain after alignment.")

    train_class = Counter(sample.label for sample in train_samples)
    val_class = Counter(sample.label for sample in val_samples)
    if train_class.get(0, 0) == 0 or train_class.get(1, 0) == 0:
        raise RuntimeError("Training split must contain both empty and screw samples.")
    if val_class.get(0, 0) == 0 or val_class.get(1, 0) == 0:
        raise RuntimeError("Validation split must contain both empty and screw samples.")

    preprocess_cfg = SemanticPreprocessConfig(
        input_size=max(32, int(args.input_size)),
        center_crop_ratio=float(args.center_crop_ratio),
    )

    train_dataset = PresenceDataset(
        train_samples,
        preprocess_cfg,
        augment=True,
        seed=int(args.seed),
    )
    weights = _sample_weights(train_samples)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_samples),
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(4, int(args.batch_size)),
        sampler=sampler,
        num_workers=0,
    )
    val_dataset = PresenceDataset(
        val_samples,
        preprocess_cfg,
        augment=False,
        seed=int(args.seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(4, int(args.batch_size)),
        shuffle=False,
        num_workers=0,
    )

    model = TinyPresenceCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = math.inf
    best_epoch = 0
    no_improve = 0
    history: list[dict[str, float]] = []

    print()
    print("=== Train ===")
    for epoch in range(1, max(1, int(args.epochs)) + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x).reshape(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * int(y.numel())
            train_count += int(y.numel())

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_probs_epoch: list[np.ndarray] = []
        val_labels_epoch: list[np.ndarray] = []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x).reshape(-1)
                loss = criterion(logits, y)
                val_loss_sum += float(loss.item()) * int(y.numel())
                val_count += int(y.numel())
                val_probs_epoch.append(torch.sigmoid(logits).cpu().numpy())
                val_labels_epoch.append(y.cpu().numpy())

        train_loss = train_loss_sum / max(1, train_count)
        val_loss = val_loss_sum / max(1, val_count)
        vp = np.concatenate(val_probs_epoch)
        vy = np.concatenate(val_labels_epoch).astype(np.int32)
        val_bal = _balanced_accuracy(vy, vp, 0.5)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_balanced_accuracy_at_0_5": float(val_bal),
            }
        )
        print(
            f"epoch {epoch:03d} | train_loss={train_loss:.5f} "
            f"| val_loss={val_loss:.5f} | val_bal@0.5={val_bal:.4f}"
        )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = float(val_loss)
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= max(1, int(args.patience)):
                print(f"Early stop at epoch {epoch}; best epoch={best_epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    val_probs, val_labels, val_slots = _predict(
        model,
        val_samples,
        preprocess_cfg,
        device,
        max(4, int(args.batch_size)),
    )
    global_calibration = _calibrate_slot(val_probs, val_labels, expected="screw")
    global_thresholds = global_calibration["suggested_thresholds"]

    slot_models: dict[str, dict[str, Any]] = {}
    print()
    print("=== Per-slot probability thresholds ===")
    print("slot  val_empty val_screw   strict   recommended   loose   active   gap")
    for slot_id in slot_ids:
        idx = np.asarray([sid == slot_id for sid in val_slots], dtype=bool)
        probs = val_probs[idx]
        labels = val_labels[idx]
        expected = expected_by_slot.get(slot_id, "")
        empty_count = int(np.sum(labels == 0))
        screw_count = int(np.sum(labels == 1))

        if empty_count > 0 and screw_count > 0:
            calibration = _calibrate_slot(probs, labels, expected=expected)
            source = "slot_validation"
            suggestions = calibration["suggested_thresholds"]
        elif args.allow_incomplete_calibration:
            calibration = {
                "validation_empty": empty_count,
                "validation_screw": screw_count,
                "separation_gap": None,
                "warning": (
                    "Insufficient per-slot validation classes. Threshold copied from global "
                    "validation and is NOT production calibrated."
                ),
            }
            source = "global_fallback"
            suggestions = dict(global_thresholds)
        else:
            raise RuntimeError(
                f"{slot_id} validation lacks both classes: empty={empty_count}, screw={screw_count}"
            )

        active = float(suggestions[str(args.threshold_profile)])
        gap = calibration.get("separation_gap")
        gap_text = "n/a" if gap is None else f"{float(gap):.4f}"
        print(
            f"{slot_id:<5} {empty_count:>9} {screw_count:>9} "
            f"{float(suggestions['strict']):>8.4f} "
            f"{float(suggestions['recommended']):>13.4f} "
            f"{float(suggestions['loose']):>8.4f} "
            f"{active:>8.4f} {gap_text:>7}"
        )
        slot_models[slot_id] = {
            "expected": expected,
            "threshold": active,
            "threshold_profile": str(args.threshold_profile),
            "suggested_thresholds": {
                key: float(value) for key, value in suggestions.items()
            },
            "calibration_source": source,
            "calibration": calibration,
        }

    onnx_path = output / "presence_classifier.onnx"
    onnx_meta = _export_onnx(
        model,
        onnx_path,
        preprocess_cfg.input_size,
        device,
    )
    checkpoint_path = output / "training_checkpoint.pt"
    torch.save(
        {
            "state_dict": best_state,
            "architecture": "TinyPresenceCNN",
            "schema_version": 7,
        },
        checkpoint_path,
    )

    roi_size = config_reference_size(config)
    model_json = {
        "schema_version": 7,
        "task": "shared semantic screw/empty binary classification",
        "canonical_size": [int(canonical_w), int(canonical_h)],
        "roi_coordinate_size": (
            None if roi_size is None else [int(roi_size[0]), int(roi_size[1])]
        ),
        "classifier": {
            "type": "shared_screw_empty_cnn_onnx",
            "architecture": "TinyPresenceCNN",
            "onnx": onnx_path.name,
            "input_name": onnx_meta["input_name"],
            "output_name": onnx_meta["output_name"],
            "output": "single logit; probability_screw = sigmoid(logit)",
            "opset": int(onnx_meta["opset"]),
        },
        "preprocess": {
            "input_size": int(preprocess_cfg.input_size),
            "center_crop_ratio": float(preprocess_cfg.center_crop_ratio),
            "grayscale": True,
            "clahe_clip_limit": float(preprocess_cfg.clahe_clip_limit),
            "clahe_grid_size": int(preprocess_cfg.clahe_grid_size),
            "std_floor": float(preprocess_cfg.std_floor),
            "clip_z": float(preprocess_cfg.clip_z),
            "normalization": (
                "gray/255 -> per-crop (x-mean)/max(std,std_floor) -> "
                "clip[-clip_z,+clip_z]/clip_z"
            ),
            "resize_note": "fixed classifier input resize only; never used for geometric alignment",
        },
        "decision": {
            "score": "P(screw)",
            "S": "PASS when P(screw) >= that S slot threshold; else missing_screw",
            "E": "PASS when P(screw) <= that E slot threshold; else excess_screw",
        },
        "threshold_profile": str(args.threshold_profile),
        "threshold_guidance": {
            "strict": "prioritize defect recall; may increase false NG",
            "recommended": "maximize validation balanced accuracy",
            "loose": "prioritize normal PASS rate; may miss borderline defects",
        },
        "slots": slot_models,
        "training": {
            "manifest": str(manifest_path),
            "resolved_manifest": str(split_manifest),
            "train_images": len(train_rows),
            "val_images": len(val_rows),
            "test_images": len(test_rows),
            "train_roi_samples": len(train_samples),
            "val_roi_samples": len(val_samples),
            "test_roi_samples": len(test_samples),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "alignment_skipped": skipped,
            "augmentation": (
                "photometric gamma/gain/offset/directional shading + small X/Y jitter; "
                "no scale; no rotation"
            ),
            "sampler": "inverse frequency of (slot,label)",
            "test_data_training_policy": (
                "Only rows marked train are optimized. val calibrates thresholds. "
                "Rows marked test are never optimized or used for threshold calibration."
            ),
        },
        "onnx_validation": onnx_meta,
    }
    (output / "model.json").write_text(
        json.dumps(model_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics: dict[str, Any] = {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "validation_global": global_calibration,
        "slot_calibration": {slot_id: row["calibration"] for slot_id, row in slot_models.items()},
    }

    if test_samples:
        test_probs, test_labels, test_slots = _predict(
            model,
            test_samples,
            preprocess_cfg,
            device,
            max(4, int(args.batch_size)),
        )
        test_rows_out: dict[str, Any] = {}
        for slot_id in slot_ids:
            idx = np.asarray([sid == slot_id for sid in test_slots], dtype=bool)
            if not np.any(idx):
                continue
            threshold = float(slot_models[slot_id]["threshold"])
            test_rows_out[slot_id] = {
                "count": int(np.sum(idx)),
                "balanced_accuracy": _balanced_accuracy(
                    test_labels[idx],
                    test_probs[idx],
                    threshold,
                ),
            }
        metrics["test"] = test_rows_out

    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== Build Summary ===")
    print(f"Best epoch:        {best_epoch}")
    print(f"Best val loss:     {best_val_loss:.6f}")
    print(f"Shared ONNX:       {onnx_path}")
    print(f"Model metadata:    {output / 'model.json'}")
    print(f"Metrics:           {output / 'metrics.json'}")
    print(f"Resolved manifest: {split_manifest}")
    print(
        "Production rule: one shared P(screw) CNN + 11 independent probability thresholds."
    )


if __name__ == "__main__":
    main()
