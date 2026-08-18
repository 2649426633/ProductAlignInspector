from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESNET18_WEIGHTS = REPO_ROOT / "weight" / "resnet18-f37072fd.pth"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_device(name: str = "auto") -> torch.device:
    value = str(name).lower()
    if value not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if value == "cuda" or (value == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def _load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"ResNet18 ImageNet weights not found: {path}. "
            "Expected the official resnet18-f37072fd.pth file."
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported weight file: {path}")
    return payload


def build_feature_extractor(
    weights_path: str | Path = DEFAULT_RESNET18_WEIGHTS,
    device: str | torch.device = "auto",
) -> tuple[nn.Module, torch.device]:
    resolved_device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    model = models.resnet18(weights=None)
    model.load_state_dict(_load_state_dict(weights_path), strict=True)
    model.fc = nn.Identity()
    model.to(resolved_device)
    model.eval()
    return model, resolved_device


def preprocess_crop(image_bgr: np.ndarray, input_size: int = 224) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("empty ROI crop")
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    elif image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

    h, w = image_bgr.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    image = cv2.copyMakeBorder(
        image_bgr,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    return np.ascontiguousarray(image, dtype=np.float32)


def extract_features(
    model: nn.Module,
    crops_bgr: Iterable[np.ndarray],
    *,
    device: torch.device,
    input_size: int = 224,
    batch_size: int = 32,
) -> np.ndarray:
    crops = list(crops_bgr)
    if not crops:
        return np.empty((0, 512), dtype=np.float32)
    tensors = [preprocess_crop(crop, input_size=input_size) for crop in crops]
    features: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tensors), max(1, int(batch_size))):
            batch_np = np.stack(tensors[start : start + batch_size], axis=0)
            batch = torch.from_numpy(batch_np).to(device)
            output = model(batch)
            output = torch.nn.functional.normalize(output, p=2, dim=1)
            features.append(output.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(features, axis=0)


def score_features(features: np.ndarray, bank: np.ndarray, k_neighbors: int = 3) -> np.ndarray:
    queries = np.asarray(features, dtype=np.float32)
    memory = np.asarray(bank, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries[None, :]
    if memory.ndim != 2 or memory.shape[0] == 0:
        raise ValueError("normal feature bank is empty")
    k = max(1, min(int(k_neighbors), memory.shape[0]))
    similarities = queries @ memory.T
    top = np.partition(similarities, kth=memory.shape[0] - k, axis=1)[:, -k:]
    distances = 1.0 - top
    return np.mean(distances, axis=1).astype(np.float32)


def leave_one_out_scores(bank: np.ndarray, k_neighbors: int = 3) -> np.ndarray:
    memory = np.asarray(bank, dtype=np.float32)
    if memory.ndim != 2 or memory.shape[0] < 2:
        raise ValueError("at least two GOOD features are required for calibration")
    k = max(1, min(int(k_neighbors), memory.shape[0] - 1))
    similarities = memory @ memory.T
    np.fill_diagonal(similarities, -np.inf)
    top = np.partition(similarities, kth=memory.shape[0] - k, axis=1)[:, -k:]
    distances = 1.0 - top
    return np.mean(distances, axis=1).astype(np.float32)


def calibrate_threshold(
    bank: np.ndarray,
    *,
    k_neighbors: int = 3,
    quantile: float = 0.98,
    margin: float = 1.10,
) -> tuple[float, dict[str, float]]:
    if not 0.5 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0.5 and 1.0")
    if margin < 1.0:
        raise ValueError("margin must be >= 1.0")
    scores = leave_one_out_scores(bank, k_neighbors=k_neighbors)
    q_value = float(np.quantile(scores, quantile))
    max_value = float(np.max(scores))
    threshold = max(q_value * float(margin), max_value * 1.02, 1e-6)
    stats = {
        "count": float(scores.size),
        "min": float(np.min(scores)),
        "p50": float(np.quantile(scores, 0.50)),
        "p90": float(np.quantile(scores, 0.90)),
        "p95": float(np.quantile(scores, 0.95)),
        "p98": float(np.quantile(scores, 0.98)),
        "p99": float(np.quantile(scores, 0.99)),
        "max": max_value,
        "calibration_quantile": float(quantile),
        "calibration_quantile_value": q_value,
        "margin": float(margin),
        "threshold": float(threshold),
    }
    return float(threshold), stats
