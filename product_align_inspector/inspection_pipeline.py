from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import cv2
import numpy as np

from .alignment import ProductLocatorConfig, align_to_reference, make_overlay
from .canonical_frame import roi_to_polygon, transform_points
from .io_utils import read_image, write_image, write_json


class Detector(Protocol):
    """Detector contract. Every detector receives the canonical/reference-sized image."""

    name: str

    def inspect(self, canonical_bgr: np.ndarray, context: dict[str, Any]) -> "DetectionResult": ...


@dataclass
class DetectionResult:
    detector: str
    status: str  # PASS / NG / ERROR
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class InspectionRecord:
    timestamp: str
    input: str
    relative_path: str
    final_status: str
    detection_run: bool
    alignment_status: str
    alignment_method: str = ""
    alignment_ecc: float | None = None
    feature_matches: int = 0
    feature_inliers: int = 0
    feature_inlier_ratio: float = 0.0
    canonical_scale: float | None = None
    rotation_deg: float | None = None
    tx: float | None = None
    ty: float | None = None
    alignment_time_sec: float = 0.0
    total_time_sec: float = 0.0
    canonical_path: str = ""
    aligned_path: str = ""
    overlay_path: str = ""
    restored_path: str = ""
    input_to_canonical: list[list[float]] | None = None
    canonical_to_input: list[list[float]] | None = None
    error: str = ""
    detections: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _enabled_rois(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not config:
        return []
    rows: list[dict[str, Any]] = []
    for slot in config.get("screw_slots", []):
        if bool(slot.get("enabled", True)) and slot.get("roi") is not None:
            rows.append(slot)
    return rows


def _draw_canonical_rois(image: np.ndarray, config: dict[str, Any] | None) -> np.ndarray:
    out = image.copy()
    for slot in _enabled_rois(config):
        x, y, w, h = map(int, slot["roi"])
        expected = str(slot.get("expected", ""))
        color = (0, 190, 0) if expected == "screw" else (0, 150, 220)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            out,
            f"{slot.get('id', '?')}:{expected}",
            (x, max(24, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def _draw_rois_back_on_raw(
    raw: np.ndarray,
    config: dict[str, Any] | None,
    canonical_to_input: np.ndarray,
) -> np.ndarray:
    out = raw.copy()
    thickness = max(2, int(round(min(raw.shape[:2]) / 1200)))
    for slot in _enabled_rois(config):
        expected = str(slot.get("expected", ""))
        color = (0, 190, 0) if expected == "screw" else (0, 150, 220)
        polygon = transform_points(roi_to_polygon(slot["roi"]), canonical_to_input)
        polygon_i = np.round(polygon).astype(np.int32)
        cv2.polylines(out, [polygon_i], True, color, thickness, cv2.LINE_AA)
        x, y = polygon_i[0].tolist()
        cv2.putText(
            out,
            f"{slot.get('id', '?')}:{expected}",
            (int(x), max(30, int(y) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            max(2, thickness - 1),
            cv2.LINE_AA,
        )
    return out


class InspectionPipeline:
    """Production inspection skeleton.

    RAW image -> canonical reference -> ROI/detectors -> map results back to RAW.

    Alignment failure is not product NG. It becomes SKIP_ALIGNMENT, is logged,
    and no detector is executed for that frame.
    """

    def __init__(
        self,
        *,
        reference: np.ndarray,
        output_dir: str | Path,
        align_cfg: ProductLocatorConfig | None = None,
        detectors: list[Detector] | None = None,
        config: dict[str, Any] | None = None,
        save_aligned: bool = True,
        save_overlay: bool = True,
        save_restored: bool = True,
    ) -> None:
        self.reference = reference
        self.output_dir = Path(output_dir)
        self.align_cfg = align_cfg or ProductLocatorConfig()
        self.detectors = list(detectors or [])
        self.config = config
        self.save_aligned = bool(save_aligned)
        self.save_overlay = bool(save_overlay)
        self.save_restored = bool(save_restored)

    @staticmethod
    def _now() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    @staticmethod
    def _safe_relative(relative_path: str | Path) -> Path:
        rel = Path(relative_path)
        parts = [p for p in rel.parts if p not in ("", ".", "..")]
        return Path(*parts) if parts else Path("image")

    def process(
        self,
        image_path: str | Path,
        *,
        relative_path: str | Path | None = None,
        context: dict[str, Any] | None = None,
    ) -> InspectionRecord:
        started = perf_counter()
        image_path = Path(image_path)
        rel = self._safe_relative(relative_path or image_path.name)
        ctx = dict(context or {})
        ctx.update({"input": str(image_path), "relative_path": rel.as_posix()})

        try:
            raw = read_image(image_path)
        except Exception as exc:
            return InspectionRecord(
                timestamp=self._now(),
                input=str(image_path),
                relative_path=rel.as_posix(),
                final_status="ERROR",
                detection_run=False,
                alignment_status="NOT_RUN",
                total_time_sec=perf_counter() - started,
                error=f"read_failed: {exc}",
            )

        alignment_started = perf_counter()
        try:
            alignment = align_to_reference(raw, self.reference, self.align_cfg)
        except Exception as exc:
            return InspectionRecord(
                timestamp=self._now(),
                input=str(image_path),
                relative_path=rel.as_posix(),
                final_status="SKIP_ALIGNMENT",
                detection_run=False,
                alignment_status="FAILED",
                alignment_time_sec=perf_counter() - alignment_started,
                total_time_sec=perf_counter() - started,
                error=str(exc),
            )

        alignment_time = perf_counter() - alignment_started
        canonical = alignment.aligned
        input_to_canonical = alignment.input_to_reference
        if input_to_canonical is None:
            input_to_canonical = alignment.feature_matrix
        canonical_to_input = alignment.reference_to_input
        if input_to_canonical is None or canonical_to_input is None:
            return InspectionRecord(
                timestamp=self._now(),
                input=str(image_path),
                relative_path=rel.as_posix(),
                final_status="SKIP_ALIGNMENT",
                detection_run=False,
                alignment_status="FAILED",
                alignment_time_sec=alignment_time,
                total_time_sec=perf_counter() - started,
                error="Alignment matrix missing; frame skipped.",
            )

        canonical_path = ""
        overlay_path = ""
        restored_path = ""

        if self.save_aligned:
            target = self.output_dir / "canonical" / rel.with_suffix(".png")
            target.parent.mkdir(parents=True, exist_ok=True)
            write_image(target, canonical)
            canonical_path = str(target)

        if self.save_overlay:
            target = self.output_dir / "overlays" / rel.with_suffix(".jpg")
            target.parent.mkdir(parents=True, exist_ok=True)
            overlay = make_overlay(self.reference, canonical)
            overlay = _draw_canonical_rois(overlay, self.config)
            write_image(target, overlay)
            overlay_path = str(target)

        if self.save_restored:
            target = self.output_dir / "restored" / rel.with_suffix(".jpg")
            target.parent.mkdir(parents=True, exist_ok=True)
            restored = _draw_rois_back_on_raw(raw, self.config, canonical_to_input)
            write_image(target, restored)
            restored_path = str(target)

        tx: float | None = None
        ty: float | None = None
        if alignment.rigid_translation_xy is not None:
            tx = float(alignment.rigid_translation_xy[0])
            ty = float(alignment.rigid_translation_xy[1])

        common = dict(
            timestamp=self._now(),
            input=str(image_path),
            relative_path=rel.as_posix(),
            alignment_status="OK",
            alignment_method=alignment.method,
            alignment_ecc=alignment.ecc_score,
            feature_matches=alignment.feature_matches,
            feature_inliers=alignment.feature_inliers,
            feature_inlier_ratio=alignment.feature_inlier_ratio,
            canonical_scale=alignment.canonical_scale,
            rotation_deg=alignment.rigid_rotation_deg,
            tx=tx,
            ty=ty,
            alignment_time_sec=alignment_time,
            canonical_path=canonical_path,
            aligned_path=canonical_path,
            overlay_path=overlay_path,
            restored_path=restored_path,
            input_to_canonical=np.asarray(input_to_canonical).tolist(),
            canonical_to_input=np.asarray(canonical_to_input).tolist(),
        )

        if not self.detectors:
            return InspectionRecord(
                **common,
                final_status="READY_FOR_DETECTION",
                detection_run=False,
                total_time_sec=perf_counter() - started,
            )

        ctx.update(
            {
                "raw": raw,
                "canonical_reference": self.reference,
                "config": self.config,
                "alignment": alignment.to_dict(),
                "input_to_canonical": input_to_canonical,
                "canonical_to_input": canonical_to_input,
            }
        )

        detection_rows: list[dict[str, Any]] = []
        any_ng = False
        any_error = False
        for detector in self.detectors:
            try:
                result = detector.inspect(canonical, ctx)
            except Exception as exc:
                result = DetectionResult(
                    detector=getattr(detector, "name", detector.__class__.__name__),
                    status="ERROR",
                    reason=str(exc),
                )
            detection_rows.append(asdict(result))
            status = str(result.status).upper()
            any_ng = any_ng or status == "NG"
            any_error = any_error or status == "ERROR"

        if any_error:
            final_status = "ERROR"
        elif any_ng:
            final_status = "NG"
        else:
            final_status = "PASS"

        return InspectionRecord(
            **common,
            final_status=final_status,
            detection_run=True,
            total_time_sec=perf_counter() - started,
            detections=detection_rows,
        )


def write_record_json(record: InspectionRecord, path: str | Path) -> None:
    write_json(Path(path), record.to_dict())
