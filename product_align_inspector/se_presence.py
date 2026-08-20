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
    """Descriptor tuned for circular screw/empty-hole ROIs.

    The old v1 descriptor flattened the ROI pixels, so a physically rotated part
    could change the lighting direction inside the ROI and produce a very large
    distance even after geometric alignment. v2 deliberately removes angular
    information and keeps radial/structural statistics instead.
    """

    size: int = 64
    center_crop_ratio: float = 0.78
    radial_bins: int = 10
    hist_bins: int = 16


def _gray_uint8(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _center_square(gray: np.ndarray, ratio: float) -> np.ndarray:
    h, w = gray.shape[:2]
    side = max(12, int(round(min(h, w) * float(np.clip(ratio, 0.45, 1.0)))))
    cx = w // 2
    cy = h // 2
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(w, x0 + side)
    y1 = min(h, y0 + side)
    x0 = max(0, x1 - side)
    y0 = max(0, y1 - side)
    return gray[y0:y1, x0:x1].copy()


def _normalize_01(values: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    scale = float(np.percentile(np.abs(arr), percentile))
    if scale <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / scale, 0.0, 1.0).astype(np.float32)


def _hist01(values: np.ndarray, bins: int) -> np.ndarray:
    hist, _ = np.histogram(
        np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0),
        bins=max(4, int(bins)),
        range=(0.0, 1.0),
    )
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return hist


def appearance_descriptor(
    crop: np.ndarray,
    cfg: PresenceDescriptorConfig | None = None,
) -> np.ndarray:
    """Return a local lighting/rotation-resistant S/E descriptor.

    Main ideas:
      * keep only the center of each ROI, where the screw/hole actually sits;
      * CLAHE normalizes exposure differences;
      * remove slowly varying illumination with a Gaussian high-pass;
      * pool intensity/edge/texture statistics by radius instead of angle.

    This makes a 180-degree physically rotated GOOD part much less likely to be
    rejected merely because the light/shadow direction around a circular hole or
    screw head changed.
    """

    cfg = cfg or PresenceDescriptorConfig()
    size = max(32, int(cfg.size))
    radial_bins = max(4, int(cfg.radial_bins))
    hist_bins = max(8, int(cfg.hist_bins))

    gray = _gray_uint8(crop)
    gray = _center_square(gray, cfg.center_crop_ratio)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    # Remove low-frequency directional lighting while keeping the screw cross,
    # hole rim, central cavity and other local structure.
    blur_k = max(5, int(round(size * 0.23)))
    if blur_k % 2 == 0:
        blur_k += 1
    low = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    high = gray - low
    high_abs = _normalize_01(np.abs(high), 95.0)

    gx = cv2.Scharr(high, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(high, cv2.CV_32F, 0, 1)
    grad = _normalize_01(cv2.magnitude(gx, gy), 95.0)
    lap = _normalize_01(np.abs(cv2.Laplacian(high, cv2.CV_32F, ksize=3)), 95.0)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = (size - 1) * 0.5
    cy = (size - 1) * 0.5
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radius /= max(1e-6, size * 0.5)
    disk = radius <= 1.0

    features: list[float] = []
    # Angularly pooled radial statistics: deliberately rotation invariant.
    for i in range(radial_bins):
        r0 = i / radial_bins
        r1 = (i + 1) / radial_bins
        mask = disk & (radius >= r0) & (radius < r1)
        if not np.any(mask):
            features.extend([0.0] * 5)
            continue
        features.extend(
            [
                float(gray[mask].mean()),
                float(gray[mask].std()),
                float(high_abs[mask].mean()),
                float(grad[mask].mean()),
                float(lap[mask].mean()),
            ]
        )

    # Global distributions inside the central disk. These retain the difference
    # between a simple empty cavity and the richer texture of a screw head.
    disk_gray = gray[disk]
    disk_grad = grad[disk]
    disk_lap = lap[disk]
    features.extend(_hist01(disk_gray, hist_bins).tolist())
    features.extend(_hist01(disk_grad, hist_bins).tolist())
    features.extend(_hist01(disk_lap, hist_bins).tolist())

    # A few center-vs-ring contrasts are useful for distinguishing an empty dark
    # hole from a metallic screw head, without relying on any particular angle.
    inner = disk & (radius <= 0.30)
    middle = disk & (radius >= 0.38) & (radius <= 0.68)
    outer = disk & (radius >= 0.68) & (radius <= 0.95)
    for channel in (gray, high_abs, grad, lap):
        inner_mean = float(channel[inner].mean()) if np.any(inner) else 0.0
        middle_mean = float(channel[middle].mean()) if np.any(middle) else 0.0
        outer_mean = float(channel[outer].mean()) if np.any(outer) else 0.0
        features.extend([inner_mean, middle_mean, outer_mean, inner_mean - middle_mean])

    vector = np.asarray(features, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-12:
        vector /= norm
    return vector


def nearest_cosine_distance(vector: np.ndarray, bank: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    bank = np.asarray(bank, dtype=np.float32)
    if bank.ndim != 2 or bank.shape[1] != vector.shape[1] or bank.shape[0] == 0:
        raise ValueError(f"Invalid bank shape {bank.shape} for descriptor {vector.shape}")
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
        descriptor_type = str(descriptor_cfg.get("type", ""))
        if descriptor_type != "radial_presence_v2":
            raise RuntimeError(
                f"{self.name}: old/incompatible S/E model. Rebuild models with "
                "tools/build_se_presence_models.py."
            )

        canonical_size = self.model.get("canonical_size")
        self.canonical_size = None if canonical_size is None else (int(canonical_size[0]), int(canonical_size[1]))
        self.descriptor_cfg = PresenceDescriptorConfig(
            size=int(descriptor_cfg.get("size", 64)),
            center_crop_ratio=float(descriptor_cfg.get("center_crop_ratio", 0.78)),
            radial_bins=int(descriptor_cfg.get("radial_bins", 10)),
            hist_bins=int(descriptor_cfg.get("hist_bins", 16)),
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
        if self.canonical_size is not None and self.canonical_size != (w, h):
            raise RuntimeError(
                f"{self.name}: model canonical size={self.canonical_size[0]}x{self.canonical_size[1]}, "
                f"runtime={w}x{h}"
            )

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
    expected = "screw"
    defect_reason = "missing_screw"
    name = "S_presence"


class EEmptyDetector(_BasePresenceDetector):
    expected = "empty"
    defect_reason = "excess_screw"
    name = "E_empty"
