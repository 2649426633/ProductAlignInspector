from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def load_product_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config_reference_size(config: dict[str, Any]) -> tuple[int, int] | None:
    """Return the coordinate-system size used when the ROIs were annotated.

    Supports both the old top-level fields and the current annotate_rois.py schema:
      coordinate_system.image_width / image_height
    """
    rw = config.get("reference_width")
    rh = config.get("reference_height")
    if rw is not None and rh is not None:
        return int(rw), int(rh)

    coordinate_system = config.get("coordinate_system") or {}
    rw = coordinate_system.get("image_width")
    rh = coordinate_system.get("image_height")
    if rw is not None and rh is not None:
        return int(rw), int(rh)
    return None


def scale_roi(
    roi_xywh: list[int] | tuple[int, int, int, int],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[int]:
    """Map one ROI through a fixed deterministic coordinate-system resize.

    This is NOT per-frame geometry estimation. It only converts coordinates between
    the saved ROI annotation image and the current canonical/reference image.
    """
    x, y, w, h = map(float, roi_xywh)
    sw, sh = map(float, source_size)
    tw, th = map(float, target_size)
    if sw <= 0 or sh <= 0 or tw <= 0 or th <= 0:
        raise ValueError(f"Invalid coordinate sizes: source={source_size}, target={target_size}")

    sx = tw / sw
    sy = th / sh
    x0 = int(round(x * sx))
    y0 = int(round(y * sy))
    x1 = int(round((x + w) * sx))
    y1 = int(round((y + h) * sy))
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def roi_for_image(
    roi_xywh: list[int] | tuple[int, int, int, int],
    config: dict[str, Any] | None,
    image_width: int,
    image_height: int,
) -> list[int]:
    if not config:
        return list(map(int, roi_xywh))
    source_size = config_reference_size(config)
    if source_size is None or source_size == (int(image_width), int(image_height)):
        return list(map(int, roi_xywh))
    return scale_roi(roi_xywh, source_size, (int(image_width), int(image_height)))


def enabled_slots(
    config: dict[str, Any] | None,
    *,
    expected: str | None = None,
) -> list[dict[str, Any]]:
    if not config:
        return []
    rows: list[dict[str, Any]] = []
    for slot in config.get("screw_slots", []):
        if not bool(slot.get("enabled", True)):
            continue
        if slot.get("roi") is None:
            continue
        if expected is not None and str(slot.get("expected", "")).lower() != expected.lower():
            continue
        rows.append(slot)
    return rows


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


def validate_roi(
    roi_xywh: list[int] | tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> bool:
    x, y, w, h = map(int, roi_xywh)
    return x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= image_width and y + h <= image_height
