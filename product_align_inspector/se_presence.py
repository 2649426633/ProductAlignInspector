from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .inspection_pipeline import DetectionResult
from .roi import crop_roi, enabled_slots, roi_for_image


@dataclass(frozen=True)
class SemanticPreprocessConfig:
    """Preprocessing shared by Python training/runtime and the future C# runtime."""

    input_size: int = 96
    center_crop_ratio: float = 0.86
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    std_floor: float = 0.08
    clip_z: float = 3.0


def _gray_uint8(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _center_square(gray: np.ndarray, ratio: float) -> np.ndarray:
    h, w = gray.shape[:2]
    side = max(12, int(round(min(h, w) * float(np.clip(ratio, 0.55, 1.0)))))
    cx, cy = w // 2, h // 2
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(w, x0 + side)
    y1 = min(h, y0 + side)
    x0 = max(0, x1 - side)
    y0 = max(0, y1 - side)
    return gray[y0:y1, x0:x1].copy()


def preprocess_presence_crop(
    crop: np.ndarray,
    cfg: SemanticPreprocessConfig | None = None,
) -> np.ndarray:
    """Convert one canonical ROI to ONNX input [1, 1, H, W] float32.

    This is classifier input resizing only; it is not geometric alignment. Product
    geometry is already fixed by the rigid canonical alignment stage.
    """

    cfg = cfg or SemanticPreprocessConfig()
    size = max(32, int(cfg.input_size))
    gray = _gray_uint8(crop)
    gray = _center_square(gray, cfg.center_crop_ratio)

    grid = max(2, int(cfg.clahe_grid_size))
    gray = cv2.createCLAHE(
        clipLimit=float(cfg.clahe_clip_limit),
        tileGridSize=(grid, grid),
    ).apply(gray)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

    x = gray.astype(np.float32) / 255.0
    mean = float(x.mean())
    std = max(float(x.std()), float(cfg.std_floor))
    x = (x - mean) / std
    clip_z = max(1.0, float(cfg.clip_z))
    x = np.clip(x, -clip_z, clip_z) / clip_z
    return x[None, None, :, :].astype(np.float32)


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-value)))


class SharedSemanticPresenceModel:
    """Shared ONNX screw/empty classifier plus 11 independent slot thresholds."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        model_json = self.model_dir / "model.json"
        if not model_json.exists():
            raise FileNotFoundError(f"Semantic S/E model metadata not found: {model_json}")

        self.model = json.loads(model_json.read_text(encoding="utf-8"))
        if int(self.model.get("schema_version", 0)) != 7:
            raise RuntimeError(
                "Old/incompatible S/E model. Build the shared semantic CNN with "
                "tools/build_se_presence_models.py."
            )
        classifier = self.model.get("classifier") or {}
        if str(classifier.get("type", "")) != "shared_screw_empty_cnn_onnx":
            raise RuntimeError(f"Unsupported S/E classifier type: {classifier.get('type')}")

        onnx_rel = str(classifier.get("onnx", "presence_classifier.onnx"))
        self.onnx_path = self.model_dir / onnx_rel
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"Shared S/E ONNX file not found: {self.onnx_path}")

        pre = self.model.get("preprocess") or {}
        self.preprocess_cfg = SemanticPreprocessConfig(
            input_size=int(pre.get("input_size", 96)),
            center_crop_ratio=float(pre.get("center_crop_ratio", 0.86)),
            clahe_clip_limit=float(pre.get("clahe_clip_limit", 2.0)),
            clahe_grid_size=int(pre.get("clahe_grid_size", 8)),
            std_floor=float(pre.get("std_floor", 0.08)),
            clip_z=float(pre.get("clip_z", 3.0)),
        )

        canonical_size = self.model.get("canonical_size")
        self.canonical_size = (
            None
            if canonical_size is None
            else (int(canonical_size[0]), int(canonical_size[1]))
        )

        slots = self.model.get("slots") or {}
        if not slots:
            raise RuntimeError("Shared S/E model has no per-slot thresholds.")
        self.slots: dict[str, dict[str, Any]] = {}
        for slot_id, row in slots.items():
            if "threshold" not in row:
                raise RuntimeError(f"Shared S/E model is missing threshold for {slot_id}")
            self.slots[str(slot_id)] = {
                "threshold": float(row["threshold"]),
                "threshold_profile": str(row.get("threshold_profile", "recommended")),
                "suggested_thresholds": row.get("suggested_thresholds") or {},
                "calibration": row.get("calibration") or {},
            }

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for S/E semantic inference. "
                "Install with: pip install -r requirements.txt"
            ) from exc

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) < 1:
            raise RuntimeError(
                f"Unexpected ONNX interface: inputs={len(inputs)}, outputs={len(outputs)}"
            )
        self.input_name = str(classifier.get("input_name") or inputs[0].name)
        self.output_name = str(classifier.get("output_name") or outputs[0].name)

    def screw_probability(self, crop: np.ndarray) -> float:
        tensor = preprocess_presence_crop(crop, self.preprocess_cfg)
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        logit = float(np.asarray(output, dtype=np.float32).reshape(-1)[0])
        return _sigmoid(logit)


_MODEL_CACHE: dict[str, SharedSemanticPresenceModel] = {}


def _load_shared_model(model_dir: str | Path) -> SharedSemanticPresenceModel:
    key = str(Path(model_dir).resolve())
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = SharedSemanticPresenceModel(key)
        _MODEL_CACHE[key] = model
    return model


class _BasePresenceDetector:
    expected: str = ""
    defect_reason: str = ""
    name: str = ""

    def __init__(
        self,
        model_dir: str | Path,
        config: dict[str, Any],
        *,
        shared_model: SharedSemanticPresenceModel | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.config = config
        self.shared = shared_model or _load_shared_model(self.model_dir)

    def inspect(self, canonical_bgr: np.ndarray, context: dict[str, Any]) -> DetectionResult:
        h, w = canonical_bgr.shape[:2]
        if self.shared.canonical_size is not None and self.shared.canonical_size != (w, h):
            raise RuntimeError(
                f"{self.name}: model canonical size="
                f"{self.shared.canonical_size[0]}x{self.shared.canonical_size[1]}, "
                f"runtime={w}x{h}"
            )

        rows: list[dict[str, Any]] = []
        any_ng = False
        for slot in enabled_slots(self.config, expected=self.expected):
            slot_id = str(slot.get("id", ""))
            slot_model = self.shared.slots.get(slot_id)
            if slot_model is None:
                raise RuntimeError(f"{self.name}: shared model is missing slot {slot_id}")

            roi = roi_for_image(slot["roi"], self.config, w, h)
            crop = crop_roi(canonical_bgr, roi)
            probability_screw = self.shared.screw_probability(crop)
            threshold = float(slot_model["threshold"])

            if self.expected == "screw":
                passed = probability_screw >= threshold
                decision = "P(screw) >= slot_threshold"
            else:
                passed = probability_screw <= threshold
                decision = "P(screw) <= slot_threshold"

            status = "PASS" if passed else "NG"
            any_ng = any_ng or not passed
            rows.append(
                {
                    "id": slot_id,
                    "expected": self.expected,
                    "status": status,
                    "reason": "" if passed else self.defect_reason,
                    "score": float(probability_screw),
                    "probability_screw": float(probability_screw),
                    "threshold": threshold,
                    "threshold_profile": slot_model["threshold_profile"],
                    "suggested_thresholds": slot_model["suggested_thresholds"],
                    "calibration": slot_model["calibration"],
                    "decision": decision,
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
