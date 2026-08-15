from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read image from Windows paths containing Chinese/non-ASCII characters."""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise FileNotFoundError(f"Cannot decode image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Write image safely to Windows paths containing Chinese/non-ASCII characters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
