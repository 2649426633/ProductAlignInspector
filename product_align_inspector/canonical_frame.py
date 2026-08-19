from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


def as_affine(matrix: Any) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.shape != (2, 3):
        raise ValueError(f"Expected 2x3 affine matrix, got {arr.shape}")
    return arr


def compose_affine(after: np.ndarray, before: np.ndarray) -> np.ndarray:
    """Return after(before(x))."""
    a = np.vstack([as_affine(after).astype(np.float64), [0.0, 0.0, 1.0]])
    b = np.vstack([as_affine(before).astype(np.float64), [0.0, 0.0, 1.0]])
    return (a @ b)[:2].astype(np.float32)


def invert_affine(matrix: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(as_affine(matrix)).astype(np.float32)


def transform_points(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.transform(points, as_affine(matrix)).reshape(-1, 2)


def roi_to_polygon(roi: list[int] | tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = map(float, roi)
    return np.asarray(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class CanonicalTransform:
    """One frame's complete RAW <-> canonical coordinate transform."""

    canonical_size: tuple[int, int]
    input_to_canonical: np.ndarray
    canonical_to_input: np.ndarray

    @classmethod
    def from_input_to_canonical(
        cls,
        matrix: np.ndarray,
        canonical_size: tuple[int, int],
    ) -> "CanonicalTransform":
        forward = as_affine(matrix).copy()
        return cls(
            canonical_size=(int(canonical_size[0]), int(canonical_size[1])),
            input_to_canonical=forward,
            canonical_to_input=invert_affine(forward),
        )

    def warp_raw_to_canonical(
        self,
        raw: np.ndarray,
        *,
        border_value: int = 255,
    ) -> np.ndarray:
        width, height = self.canonical_size
        border = (border_value, border_value, border_value) if raw.ndim == 3 else border_value
        return cv2.warpAffine(
            raw,
            self.input_to_canonical,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )

    def canonical_points_on_raw(self, points_xy: np.ndarray) -> np.ndarray:
        return transform_points(points_xy, self.canonical_to_input)

    def canonical_roi_on_raw(
        self,
        roi: list[int] | tuple[int, int, int, int],
    ) -> np.ndarray:
        return self.canonical_points_on_raw(roi_to_polygon(roi))
