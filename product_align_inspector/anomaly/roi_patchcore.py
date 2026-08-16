from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ROIAnomalyRegion:
    id: str
    roi: tuple[int, int, int, int]
    source_group: str


@dataclass
class ROIModel:
    roi_id: str
    roi: tuple[int, int, int, int]
    source_group: str
    memory: np.ndarray
    threshold: float | None
    calibration_scores: list[float]
    score_top_fraction: float
    patch_grid: int
    feature_dim: int


def collect_anomaly_regions(config: dict[str, Any]) -> list[ROIAnomalyRegion]:
    """Collect all fixed regions that may be used for one-class anomaly detection.

    The existing screw/spring ROI configuration is reused. A future/general
    surface-defect ROI can be added under ``anomaly_regions`` without changing
    this runtime.
    """
    regions: list[ROIAnomalyRegion] = []
    seen: set[str] = set()

    for group in ("anomaly_regions", "screw_slots", "spring_regions"):
        for item in config.get(group, []):
            if not bool(item.get("enabled", True)):
                continue
            roi_id = str(item.get("id", "")).strip()
            roi = item.get("roi")
            if not roi_id or roi is None:
                continue
            if roi_id in seen:
                raise ValueError(f"Duplicate ROI id in config: {roi_id}")
            if len(roi) != 4:
                raise ValueError(f"ROI {roi_id} must have [x,y,w,h], got {roi}")
            x, y, w, h = map(int, roi)
            if w <= 0 or h <= 0:
                raise ValueError(f"ROI {roi_id} has invalid size: {roi}")
            regions.append(ROIAnomalyRegion(roi_id, (x, y, w, h), group))
            seen.add(roi_id)
    return regions


def select_regions(config: dict[str, Any], roi_ids: list[str] | None) -> list[ROIAnomalyRegion]:
    regions = collect_anomaly_regions(config)
    if not regions:
        raise ValueError("No enabled anomaly_regions/screw_slots/spring_regions found in config")
    if not roi_ids:
        return regions

    requested = list(dict.fromkeys(str(v) for v in roi_ids))
    by_id = {region.id: region for region in regions}
    missing = [roi_id for roi_id in requested if roi_id not in by_id]
    if missing:
        raise ValueError(
            f"ROI id(s) not found in config: {', '.join(missing)}. "
            f"Available: {', '.join(sorted(by_id))}"
        )
    return [by_id[roi_id] for roi_id in requested]


def nearest_cosine_distances(
    query_tokens: np.ndarray,
    memory: np.ndarray,
    *,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Distance from every normalized query token to its nearest memory token."""
    query = np.asarray(query_tokens, dtype=np.float32)
    bank = np.asarray(memory, dtype=np.float32)
    if query.ndim != 2 or bank.ndim != 2 or query.shape[1] != bank.shape[1]:
        raise ValueError(f"Feature shape mismatch: query={query.shape}, memory={bank.shape}")
    if len(bank) == 0:
        raise ValueError("Empty memory bank")

    result = np.empty((len(query),), dtype=np.float32)
    for start in range(0, len(query), max(1, int(chunk_size))):
        stop = min(len(query), start + max(1, int(chunk_size)))
        similarities = query[start:stop] @ bank.T
        max_similarity = np.max(similarities, axis=1)
        result[start:stop] = np.clip(1.0 - max_similarity, 0.0, 2.0)
    return result


def score_patch_tokens(
    query_tokens: np.ndarray,
    memory: np.ndarray,
    *,
    patch_grid: int,
    top_fraction: float = 0.05,
) -> tuple[float, np.ndarray, dict[str, float]]:
    distances = nearest_cosine_distances(query_tokens, memory)
    expected = int(patch_grid) * int(patch_grid)
    if len(distances) != expected:
        raise ValueError(f"Expected {expected} patch scores, got {len(distances)}")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0,1]")

    k = max(1, int(np.ceil(len(distances) * float(top_fraction))))
    top_values = np.partition(distances, len(distances) - k)[-k:]
    score = float(np.mean(top_values))
    stats = {
        "score": score,
        "max": float(np.max(distances)),
        "mean": float(np.mean(distances)),
        "p95": float(np.quantile(distances, 0.95)),
        "top_k": int(k),
    }
    return score, distances.reshape(patch_grid, patch_grid), stats


def save_roi_model(model_dir: str | Path, model: ROIModel) -> Path:
    model_dir = Path(model_dir)
    banks_dir = model_dir / "banks"
    banks_dir.mkdir(parents=True, exist_ok=True)
    path = banks_dir / f"{model.roi_id}.npz"
    np.savez_compressed(
        path,
        memory=np.asarray(model.memory, dtype=np.float32),
        roi=np.asarray(model.roi, dtype=np.int32),
        threshold=np.asarray([np.nan if model.threshold is None else model.threshold], dtype=np.float32),
        calibration_scores=np.asarray(model.calibration_scores, dtype=np.float32),
        score_top_fraction=np.asarray([model.score_top_fraction], dtype=np.float32),
        patch_grid=np.asarray([model.patch_grid], dtype=np.int32),
        feature_dim=np.asarray([model.feature_dim], dtype=np.int32),
        source_group=np.asarray([model.source_group]),
    )
    return path


def load_roi_model(model_dir: str | Path, roi_id: str) -> ROIModel:
    path = Path(model_dir) / "banks" / f"{roi_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"ROI memory bank not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        threshold_value = float(data["threshold"][0])
        return ROIModel(
            roi_id=roi_id,
            roi=tuple(int(v) for v in data["roi"].tolist()),
            source_group=str(data["source_group"][0]),
            memory=np.asarray(data["memory"], dtype=np.float32),
            threshold=None if np.isnan(threshold_value) else threshold_value,
            calibration_scores=[float(v) for v in data["calibration_scores"].tolist()],
            score_top_fraction=float(data["score_top_fraction"][0]),
            patch_grid=int(data["patch_grid"][0]),
            feature_dim=int(data["feature_dim"][0]),
        )


def write_model_manifest(model_dir: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(model_dir) / "model.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_model_manifest(model_dir: str | Path) -> dict[str, Any]:
    path = Path(model_dir) / "model.json"
    if not path.exists():
        raise FileNotFoundError(f"ROI DINO PatchCore manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
