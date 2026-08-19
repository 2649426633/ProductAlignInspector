from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _as_affine(matrix: Any) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.shape != (2, 3):
        raise ValueError(f"Expected 2x3 affine matrix, got {arr.shape}")
    return arr


def compose_affine(after: np.ndarray, before: np.ndarray) -> np.ndarray:
    """Return the affine transform equivalent to after(before(x))."""
    a = np.vstack([_as_affine(after).astype(np.float64), [0.0, 0.0, 1.0]])
    b = np.vstack([_as_affine(before).astype(np.float64), [0.0, 0.0, 1.0]])
    return (a @ b)[:2].astype(np.float32)


def invert_affine(matrix: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(_as_affine(matrix)).astype(np.float32)


def transform_points(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.transform(points, _as_affine(matrix)).reshape(-1, 2)


def roi_to_polygon(roi: list[int] | tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = map(float, roi)
    return np.asarray(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        dtype=np.float32,
    )


@dataclass
class CanonicalCalibration:
    raw_reference_size: tuple[int, int]
    canonical_size: tuple[int, int]
    raw_reference_to_canonical: np.ndarray
    canonical_to_raw_reference: np.ndarray
    raw_reference: str = ""
    canonical_reference: str = ""
    feature_matches: int = 0
    feature_inliers: int = 0
    feature_inlier_ratio: float = 0.0

    @classmethod
    def load(cls, path: str | Path) -> "CanonicalCalibration":
        import json

        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        raw_size = tuple(map(int, data["raw_reference_size"]))
        canonical_size = tuple(map(int, data["canonical_size"]))
        return cls(
            raw_reference_size=(raw_size[0], raw_size[1]),
            canonical_size=(canonical_size[0], canonical_size[1]),
            raw_reference_to_canonical=_as_affine(data["raw_reference_to_canonical"]),
            canonical_to_raw_reference=_as_affine(data["canonical_to_raw_reference"]),
            raw_reference=str(data.get("raw_reference", "")),
            canonical_reference=str(data.get("canonical_reference", "")),
            feature_matches=int(data.get("feature_matches", 0)),
            feature_inliers=int(data.get("feature_inliers", 0)),
            feature_inlier_ratio=float(data.get("feature_inlier_ratio", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "raw_reference": self.raw_reference,
            "canonical_reference": self.canonical_reference,
            "raw_reference_size": [int(self.raw_reference_size[0]), int(self.raw_reference_size[1])],
            "canonical_size": [int(self.canonical_size[0]), int(self.canonical_size[1])],
            "raw_reference_to_canonical": self.raw_reference_to_canonical.tolist(),
            "canonical_to_raw_reference": self.canonical_to_raw_reference.tolist(),
            "feature_matches": int(self.feature_matches),
            "feature_inliers": int(self.feature_inliers),
            "feature_inlier_ratio": float(self.feature_inlier_ratio),
            "note": (
                "Per-frame motion is rigid in RAW coordinates. This fixed matrix only "
                "converts the RAW reference coordinate system into the canonical preview coordinate system."
            ),
        }


class CanonicalFrameMapper:
    """Map each rigidly aligned RAW frame into the fixed preview/canonical frame.

    Runtime geometry:
        input RAW --(rigid, scale=1)--> RAW reference
                  --(fixed calibration)--> canonical preview

    The fixed calibration is computed once. It is never re-estimated per frame.
    """

    def __init__(self, calibration: CanonicalCalibration):
        self.calibration = calibration

    @classmethod
    def from_json(cls, path: str | Path) -> "CanonicalFrameMapper":
        return cls(CanonicalCalibration.load(path))

    def raw_input_to_canonical(self, input_to_raw_reference: np.ndarray) -> np.ndarray:
        return compose_affine(
            self.calibration.raw_reference_to_canonical,
            _as_affine(input_to_raw_reference),
        )

    def canonical_to_raw_input(self, input_to_raw_reference: np.ndarray) -> np.ndarray:
        return invert_affine(self.raw_input_to_canonical(input_to_raw_reference))

    def warp_to_canonical(
        self,
        raw_input: np.ndarray,
        input_to_raw_reference: np.ndarray,
        *,
        border_value: int = 255,
    ) -> tuple[np.ndarray, np.ndarray]:
        matrix = self.raw_input_to_canonical(input_to_raw_reference)
        width, height = self.calibration.canonical_size
        border = (border_value, border_value, border_value) if raw_input.ndim == 3 else border_value
        canonical = cv2.warpAffine(
            raw_input,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )
        return canonical, matrix

    def canonical_roi_on_raw(
        self,
        roi: list[int] | tuple[int, int, int, int],
        input_to_raw_reference: np.ndarray,
    ) -> np.ndarray:
        return transform_points(
            roi_to_polygon(roi),
            self.canonical_to_raw_input(input_to_raw_reference),
        )

    def draw_roi_on_raw(
        self,
        raw: np.ndarray,
        roi: list[int] | tuple[int, int, int, int],
        input_to_raw_reference: np.ndarray,
        *,
        label: str = "",
        color: tuple[int, int, int] = (0, 200, 0),
        thickness: int = 3,
    ) -> np.ndarray:
        out = raw.copy()
        polygon = np.round(
            self.canonical_roi_on_raw(roi, input_to_raw_reference)
        ).astype(np.int32)
        cv2.polylines(out, [polygon], True, color, thickness, cv2.LINE_AA)
        if label:
            x, y = polygon[0].tolist()
            cv2.putText(
                out,
                label,
                (int(x), max(25, int(y) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
        return out
