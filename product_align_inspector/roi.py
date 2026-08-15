from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_product_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def crop_roi(image: np.ndarray, roi_xywh: list[int] | tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = map(int, roi_xywh)
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid ROI size: {roi_xywh}")
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(image.shape[1], x + w)
    y1 = min(image.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI is outside image: {roi_xywh}, image={image.shape}")
    return image[y0:y1, x0:x1].copy()


def validate_roi(roi_xywh: list[int] | tuple[int, int, int, int], image_width: int, image_height: int) -> bool:
    x, y, w, h = map(int, roi_xywh)
    return x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= image_width and y + h <= image_height
