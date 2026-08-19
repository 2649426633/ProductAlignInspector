from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from .alignment import ProductLocatorConfig, align_to_reference
from .io_utils import read_image, write_image, write_json


class Detector(Protocol):
    """Minimal detector contract for later S/E/spring/surface modules."""

    name: str

    def inspect(self, aligned_bgr: np.ndarray, context: dict[str, Any]) -> "DetectionResult": ...


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
    rotation_deg: float | None = None
    tx: float | None = None
    ty: float | None = None
    alignment_time_sec: float = 0.0
    total_time_sec: float = 0.0
    aligned_path: str = ""
    error: str = ""
    detections: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InspectionPipeline:
    """Production-oriented inspection skeleton.

    Stage order is fixed:
      raw image -> rigid alignment -> detectors -> final decision/log

    The important safety rule is that an alignment failure NEVER becomes product NG
    and NEVER enters any detector. It is returned as SKIP_ALIGNMENT and can be
    reviewed/retried separately.
    """

    def __init__(
        self,
        *,
        reference: np.ndarray,
        output_dir: str | Path,
        align_cfg: ProductLocatorConfig | None = None,
        detectors: list[Detector] | None = None,
        save_aligned: bool = True,
    ) -> None:
        self.reference = reference
        self.output_dir = Path(output_dir)
        self.align_cfg = align_cfg or ProductLocatorConfig()
        self.detectors = list(detectors or [])
        self.save_aligned = bool(save_aligned)

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
            # Production rule: alignment failure is skipped, logged, and never
            # passed to S/E/spring/anomaly detectors.
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
        aligned = alignment.aligned
        aligned_path = ""
        if self.save_aligned:
            target = self.output_dir / "aligned" / rel.with_suffix(".png")
            target.parent.mkdir(parents=True, exist_ok=True)
            write_image(target, aligned)
            aligned_path = str(target)

        tx: float | None = None
        ty: float | None = None
        if alignment.rigid_translation_xy is not None:
            tx = float(alignment.rigid_translation_xy[0])
            ty = float(alignment.rigid_translation_xy[1])

        # Current framework can run before any detector is ready. An aligned image
        # is then READY_FOR_DETECTION rather than PASS/NG.
        if not self.detectors:
            return InspectionRecord(
                timestamp=self._now(),
                input=str(image_path),
                relative_path=rel.as_posix(),
                final_status="READY_FOR_DETECTION",
                detection_run=False,
                alignment_status="OK",
                alignment_method=alignment.method,
                alignment_ecc=alignment.ecc_score,
                rotation_deg=alignment.rigid_rotation_deg,
                tx=tx,
                ty=ty,
                alignment_time_sec=alignment_time,
                total_time_sec=perf_counter() - started,
                aligned_path=aligned_path,
            )

        ctx["alignment"] = alignment.to_dict()
        detection_rows: list[dict[str, Any]] = []
        any_ng = False
        any_error = False
        for detector in self.detectors:
            try:
                result = detector.inspect(aligned, ctx)
            except Exception as exc:
                result = DetectionResult(
                    detector=getattr(detector, "name", detector.__class__.__name__),
                    status="ERROR",
                    reason=str(exc),
                )

            row = asdict(result)
            detection_rows.append(row)
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
            timestamp=self._now(),
            input=str(image_path),
            relative_path=rel.as_posix(),
            final_status=final_status,
            detection_run=True,
            alignment_status="OK",
            alignment_method=alignment.method,
            alignment_ecc=alignment.ecc_score,
            rotation_deg=alignment.rigid_rotation_deg,
            tx=tx,
            ty=ty,
            alignment_time_sec=alignment_time,
            total_time_sec=perf_counter() - started,
            aligned_path=aligned_path,
            detections=detection_rows,
        )


def write_record_json(record: InspectionRecord, path: str | Path) -> None:
    write_json(Path(path), record.to_dict())
