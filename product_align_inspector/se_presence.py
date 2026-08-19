from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .inspection_pipeline import DetectionResult
from .roi import crop_roi, enabled_slots, roi_for_image


@dataclass
class PresenceDescriptorConfig:
    size: int = 40
    gray_weight: float = 0.35
    gradient_weight: float = 0.65


def _gray_uint8(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def appearance_descriptor(
    crop: np.ndarray,
    cfg: PresenceDescriptorConfig | None = None,
) -> np.ndarray:
    """Lighting-resistant handcrafted descriptor for screw/empty appearance.

    It intentionally has no deep-learning/runtime dependency so it is easy to port
    to OpenCvSharp later. S and E still use separate GOOD banks/models.
    """
    cfg = cfg or PresenceDescriptorConfig()
    size = max(16, int(cfg.size))

    gray = _gray_uint8(crop)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    gray = gray - float(gray.mean())
    gray_std = float(gray.std())
    if gray_std > 1e-6:
        gray /= gray_std

    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    grad = cv2.magnitude(gx, gy)
    grad_mean = float(grad.mean())
    grad_std = float(grad.std())
    if grad_std > 1e-6:
        grad = (grad - grad_mean) / grad_std
    else:
        grad = grad - grad_mean

    vector = np.concatenate(
        [
            gray.reshape(-1) * float(cfg.gray_weight),
            grad.reshape(-1) * float(cfg.gradient_weight),
        ]
    ).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-12:
        vector /= norm
    return vector


def nearest_cosine_distance(vector: np.ndarray, bank: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    bank = np.asarray(bank, dtype=np.float32)
    if bank.ndim != 2 or bank.shape[1] != vector.shape[1] or bank.shape[0] == 0:
        raise ValueError(f"Invalid bank shape {bank.shape} for descriptor {vector.shape}")
    # Both descriptors and saved banks are L2 normalized.
    similarity = bank @ vector[0]
    return float(1.0 - float(np.max(similarity)))


def leave_one_out_distances(bank: np.ndarray) -> np.ndarray:
    bank = np.asarray(bank, dtype=np.float32)
    if bank.ndim != 2 or bank.shape[0] < 2:
        return np.zeros((bank.shape[0],), dtype=np.float32)
    similarity = bank @ bank.T
    np.fill_diagonal(similarity, -np.inf)
    return (1.0 - np.max(similarity, axis=1)).astype(np.float32)


def threshold_from_good(
    distances: np.ndarray,
    *,
    margin_factor: float = 1.25,
    margin_abs: float = 0.002,
) -> float:
    values = np.asarray(distances, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.05
    # GOOD-only threshold: cover the observed normal range with a modest margin.
    base = max(float(np.max(values)), float(values.mean() + 4.0 * values.std()))
    return max(0.005, base * float(margin_factor) + float(margin_abs))


class _BasePresenceDetector:
    expected: str = ""
    defect_reason: str = ""
    name: str = ""

    def __init__(self, model_dir: str | Path, config: dict[str, Any]) -> None:
        self.model_dir = Path(model_dir)
        self.config = config
        model_path = self.model_dir / "model.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Presence model not found: {model_path}")
        self.model = json.loads(model_path.read_text(encoding="utf-8"))
        if str(self.model.get("expected", "")).lower() != self.expected:
            raise ValueError(
                f"Wrong model type: expected={self.expected}, model={self.model.get('expected')}"
            )
        descriptor_cfg = self.model.get("descriptor") or {}
        self.descriptor_cfg = PresenceDescriptorConfig(
            size=int(descriptor_cfg.get("size", 40)),
            gray_weight=float(descriptor_cfg.get("gray_weight", 0.35)),
            gradient_weight=float(descriptor_cfg.get("gradient_weight", 0.65)),
        )
        self.thresholds = {str(k): float(v) for k, v in (self.model.get("thresholds") or {}).items()}
        self.banks: dict[str, np.ndarray] = {}
        for slot_id in self.thresholds:
            path = self.model_dir / "banks" / f"{slot_id}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Presence bank not found: {path}")
            self.banks[slot_id] = np.load(path).astype(np.float32)

    def inspect(self, canonical_bgr: np.ndarray, context: dict[str, Any]) -> DetectionResult:
        h, w = canonical_bgr.shape[:2]
        slot_rows: list[dict[str, Any]] = []
        any_ng = False

        for slot in enabled_slots(self.config, expected=self.expected):
            slot_id = str(slot.get("id", ""))
            if slot_id not in self.banks or slot_id not in self.thresholds:
                raise RuntimeError(f"{self.name}: model is missing slot {slot_id}")

            roi = roi_for_image(slot["roi"], self.config, w, h)
            crop = crop_roi(canonical_bgr, roi)
            descriptor = appearance_descriptor(crop, self.descriptor_cfg)
            score = nearest_cosine_distance(descriptor, self.banks[slot_id])
            threshold = float(self.thresholds[slot_id])
            status = "NG" if score > threshold else "PASS"
            any_ng = any_ng or status == "NG"
            slot_rows.append(
                {
                    "id": slot_id,
                    "expected": self.expected,
                    "status": status,
                    "reason": self.defect_reason if status == "NG" else "",
                    "score": score,
                    "threshold": threshold,
                    "roi_canonical": roi,
                }
            )

        return DetectionResult(
            detector=self.name,
            status="NG" if any_ng else "PASS",
            reason=self.defect_reason if any_ng else "",
            details={"slots": slot_rows},
        )


class SPresenceDetector(_BasePresenceDetector):
    """S01/S02: normal state contains a screw; NG means missing/abnormal screw appearance."""

    expected = "screw"
    defect_reason = "missing_screw"
    name = "S_presence"


class EEmptyDetector(_BasePresenceDetector):
    """E01..E09: normal state is empty; NG means an extra screw/object appeared."""

    expected = "empty"
    defect_reason = "excess_screw"
    name = "E_empty"
