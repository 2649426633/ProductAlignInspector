from __future__ import annotations

import json
from pathlib import Path


def load_decision_multipliers(path: str | Path | None) -> tuple[float, dict[str, float]]:
    """Load per-ROI decision threshold multipliers.

    Effective threshold = calibrated threshold * global threshold scale * ROI multiplier.
    Missing file/path means multiplier 1.0 for every ROI.
    """
    if path is None:
        return 1.0, {}

    p = Path(path)
    if not p.is_file():
        return 1.0, {}

    data = json.loads(p.read_text(encoding="utf-8"))
    default_multiplier = float(data.get("default_multiplier", 1.0))
    if default_multiplier <= 0:
        raise ValueError("default_multiplier must be > 0")

    raw = data.get("roi_multipliers", {})
    if not isinstance(raw, dict):
        raise ValueError("roi_multipliers must be an object")

    multipliers: dict[str, float] = {}
    for key, value in raw.items():
        multiplier = float(value)
        if multiplier <= 0:
            raise ValueError(f"ROI multiplier must be > 0: {key}={value}")
        multipliers[str(key)] = multiplier
    return default_multiplier, multipliers


def roi_multiplier(roi_id: str, default_multiplier: float, multipliers: dict[str, float]) -> float:
    return float(multipliers.get(roi_id, default_multiplier))
