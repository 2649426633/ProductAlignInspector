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
    """Local structure descriptor configuration for S/E presence inspection.

    v6 intentionally suppresses absolute brightness and slow illumination changes.
    The descriptor is still deterministic and lightweight enough to port to C#.
    """

    size: int = 64
    center_crop_ratio: float = 0.72
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
    cx, cy = w // 2, h // 2
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(w, x0 + side)
    y1 = min(h, y0 + side)
    x0 = max(0, x1 - side)
    y0 = max(0, y1 - side)
    return gray[y0:y1, x0:x1].copy()


def _normalize_01(values: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    arr = np.abs(np.asarray(values, dtype=np.float32))
    scale = float(np.percentile(arr, percentile))
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


def _rotation_invariant_lbp_stats(gray: np.ndarray) -> np.ndarray:
    """Return compact LBP statistics robust to monotonic illumination changes.

    Instead of a 256-bin raw LBP code histogram, v6 stores the number of bright
    neighbours and the number of binary transitions around the centre pixel.
    Both statistics are invariant to circular rotation of the local pattern.
    """

    padded = np.pad(gray, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    neighbors = [
        padded[:-2, :-2],
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, 2:],
        padded[2:, 2:],
        padded[2:, 1:-1],
        padded[2:, :-2],
        padded[1:-1, :-2],
    ]
    bits = np.stack([(n >= center).astype(np.uint8) for n in neighbors], axis=0)
    ones = bits.sum(axis=0)
    transitions = np.sum(bits != np.roll(bits, -1, axis=0), axis=0)

    hist_ones = np.bincount(ones.reshape(-1), minlength=9).astype(np.float32)
    hist_transitions = np.bincount(transitions.reshape(-1), minlength=9).astype(np.float32)
    hist_ones /= max(1.0, float(hist_ones.sum()))
    hist_transitions /= max(1.0, float(hist_transitions.sum()))
    return np.concatenate([hist_ones, hist_transitions]).astype(np.float32)


def appearance_descriptor(
    crop: np.ndarray,
    cfg: PresenceDescriptorConfig | None = None,
) -> np.ndarray:
    """Return an illumination-resistant structure descriptor for one S/E slot.

    Important v6 change: absolute gray level is not used as a feature. We first
    remove slow illumination with local contrast normalization, then describe only
    residual structure, edges, Laplacian texture and rotation-invariant LBP.
    This targets the observed case where a geometrically correct GOOD image has a
    very different metal reflection pattern from the training images.
    """

    cfg = cfg or PresenceDescriptorConfig()
    size = max(32, int(cfg.size))
    radial_bins = max(4, int(cfg.radial_bins))
    hist_bins = max(8, int(cfg.hist_bins))

    gray_u8 = _gray_uint8(crop)
    gray_u8 = _center_square(gray_u8, cfg.center_crop_ratio)
    gray_u8 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_u8)
    gray = cv2.resize(gray_u8, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    # Local contrast normalization (Retinex-like). Slow brightness/shading is
    # removed while screw slots, hole rims and local metal structure remain.
    sigma = max(2.0, size * 0.10)
    low = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    residual = gray - low
    energy = cv2.GaussianBlur(residual * residual, (0, 0), sigmaX=sigma, sigmaY=sigma)
    local_std = np.sqrt(np.maximum(energy, 1e-5))
    normalized = np.clip(residual / (local_std + 1e-3), -3.0, 3.0)
    structure = np.clip(np.abs(normalized) / 3.0, 0.0, 1.0).astype(np.float32)

    gx = cv2.Scharr(normalized, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(normalized, cv2.CV_32F, 0, 1)
    grad = _normalize_01(cv2.magnitude(gx, gy), 95.0)
    lap = _normalize_01(cv2.Laplacian(normalized, cv2.CV_32F, ksize=3), 95.0)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = (size - 1) * 0.5
    cy = (size - 1) * 0.5
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radius /= max(1e-6, size * 0.5)
    disk = radius <= 1.0

    features: list[float] = []

    # Radial pooling removes angular lighting direction while retaining the
    # centre/rim structure that distinguishes a stable screw/hole appearance.
    for i in range(radial_bins):
        r0 = i / radial_bins
        r1 = (i + 1) / radial_bins
        mask = disk & (radius >= r0) & (radius < r1)
        if not np.any(mask):
            features.extend([0.0] * 6)
            continue
        features.extend(
            [
                float(structure[mask].mean()),
                float(structure[mask].std()),
                float(grad[mask].mean()),
                float(grad[mask].std()),
                float(lap[mask].mean()),
                float(lap[mask].std()),
            ]
        )

    features.extend(_hist01(structure[disk], hist_bins).tolist())
    features.extend(_hist01(grad[disk], hist_bins).tolist())
    features.extend(_hist01(lap[disk], hist_bins).tolist())

    inner = disk & (radius <= 0.30)
    middle = disk & (radius >= 0.38) & (radius <= 0.68)
    outer = disk & (radius >= 0.68) & (radius <= 0.95)
    for channel in (structure, grad, lap):
        a = float(channel[inner].mean()) if np.any(inner) else 0.0
        b = float(channel[middle].mean()) if np.any(middle) else 0.0
        c = float(channel[outer].mean()) if np.any(outer) else 0.0
        features.extend([a, b, c, a - b, b - c])

    # LBP uses only local ordering, not absolute brightness.
    lbp_gray = cv2.resize(gray_u8, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    features.extend(_rotation_invariant_lbp_stats(lbp_gray).tolist())

    vector = np.asarray(features, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-12:
        vector /= norm
    return vector


def topk_cosine_distance(vector: np.ndarray, bank: np.ndarray, top_k: int = 5) -> float:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    bank = np.asarray(bank, dtype=np.float32)
    if bank.ndim != 2 or bank.shape[0] == 0 or bank.shape[1] != vector.shape[0]:
        raise ValueError(f"Invalid bank shape {bank.shape} for descriptor {vector.shape}")
    distances = 1.0 - bank @ vector
    k = max(1, min(int(top_k), distances.size))
    return float(np.mean(np.partition(distances, k - 1)[:k]))


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
        if int(self.model.get("schema_version", 0)) != 6:
            raise RuntimeError(
                f"{self.name}: old/incompatible S/E model. Rebuild with tools/build_se_presence_models.py."
            )
        if str(self.model.get("expected", "")).lower() != self.expected:
            raise ValueError(f"Wrong model type: expected={self.expected}, model={self.model.get('expected')}")

        descriptor_cfg = self.model.get("descriptor") or {}
        if str(descriptor_cfg.get("type", "")) != "slotwise_structure_v6":
            raise RuntimeError(f"{self.name}: unsupported descriptor type {descriptor_cfg.get('type')}")

        canonical_size = self.model.get("canonical_size")
        self.canonical_size = None if canonical_size is None else (int(canonical_size[0]), int(canonical_size[1]))
        self.descriptor_cfg = PresenceDescriptorConfig(
            size=int(descriptor_cfg.get("size", 64)),
            center_crop_ratio=float(descriptor_cfg.get("center_crop_ratio", 0.72)),
            radial_bins=int(descriptor_cfg.get("radial_bins", 10)),
            hist_bins=int(descriptor_cfg.get("hist_bins", 16)),
        )
        classifier = self.model.get("classifier") or {}
        self.default_top_k = int(classifier.get("top_k", 5))
        self.threshold_profile = str(classifier.get("threshold_profile", "recommended"))

        slot_models = self.model.get("slots") or {}
        if not slot_models:
            raise RuntimeError(f"{self.name}: model has no slot definitions")

        self.slot_models: dict[str, dict[str, Any]] = {}
        for slot_id, row in slot_models.items():
            bank_rel = str(row.get("bank", f"banks/{slot_id}.npy"))
            bank_path = self.model_dir / bank_rel
            if not bank_path.exists():
                raise FileNotFoundError(f"{self.name}: bank not found for {slot_id}: {bank_path}")
            self.slot_models[str(slot_id)] = {
                "bank": np.load(bank_path).astype(np.float32),
                "threshold": float(row["threshold"]),
                "top_k": int(row.get("top_k", self.default_top_k)),
                "suggested_thresholds": row.get("suggested_thresholds") or {},
                "calibration": row.get("calibration") or {},
            }

    def inspect(self, canonical_bgr: np.ndarray, context: dict[str, Any]) -> DetectionResult:
        h, w = canonical_bgr.shape[:2]
        if self.canonical_size is not None and self.canonical_size != (w, h):
            raise RuntimeError(
                f"{self.name}: model canonical size={self.canonical_size[0]}x{self.canonical_size[1]}, runtime={w}x{h}"
            )

        rows: list[dict[str, Any]] = []
        any_ng = False
        for slot in enabled_slots(self.config, expected=self.expected):
            slot_id = str(slot.get("id", ""))
            model = self.slot_models.get(slot_id)
            if model is None:
                raise RuntimeError(f"{self.name}: model is missing slot {slot_id}")

            roi = roi_for_image(slot["roi"], self.config, w, h)
            descriptor = appearance_descriptor(crop_roi(canonical_bgr, roi), self.descriptor_cfg)
            score = topk_cosine_distance(descriptor, model["bank"], model["top_k"])
            threshold = float(model["threshold"])
            passed = score <= threshold
            status = "PASS" if passed else "NG"
            any_ng = any_ng or not passed

            rows.append(
                {
                    "id": slot_id,
                    "expected": self.expected,
                    "status": status,
                    "reason": "" if passed else self.defect_reason,
                    "score": float(score),
                    "threshold": threshold,
                    "threshold_profile": self.threshold_profile,
                    "suggested_thresholds": model["suggested_thresholds"],
                    "decision": "illumination-invariant score <= slot_threshold",
                    "roi_canonical": roi,
                }
            )

        return DetectionResult(
            detector=self.name,
            status="NG" if any_ng else "PASS",
            reason=self.defect_reason if any_ng else "",
            details={"slots": rows},
        )


class SPresenceDetector(_BasePresenceDetector):
    expected = "screw"
    defect_reason = "missing_screw"
    name = "S_presence"


class EEmptyDetector(_BasePresenceDetector):
    expected = "empty"
    defect_reason = "excess_screw"
    name = "E_empty"
