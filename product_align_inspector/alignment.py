from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class ProductLocatorConfig:
    foreground_threshold: int = 238
    border_margin_ratio: float = 0.004
    min_component_area_ratio: float = 0.00002
    close_kernel_ratio: float = 0.006
    crop_padding_ratio: float = 0.08


@dataclass
class ProductLocation:
    bbox_xywh: tuple[int, int, int, int]
    center_xy: tuple[float, float]
    angle_deg: float
    area_px: float
    mask: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("mask")
        return data


@dataclass
class AlignmentResult:
    aligned: np.ndarray
    coarse: np.ndarray
    foreground_mask: np.ndarray
    location: ProductLocation
    ecc_score: float | None
    ecc_matrix: np.ndarray | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location.to_dict(),
            "ecc_score": None if self.ecc_score is None else float(self.ecc_score),
            "ecc_matrix": None if self.ecc_matrix is None else self.ecc_matrix.tolist(),
            "aligned_shape": list(self.aligned.shape),
            "coarse_shape": list(self.coarse.shape),
        }


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def _remove_border_components(mask: np.ndarray, cfg: ProductLocatorConfig) -> np.ndarray:
    h, w = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    border_margin = max(1, int(min(h, w) * cfg.border_margin_ratio))
    min_area = max(25, int(h * w * cfg.min_component_area_ratio))

    for label in range(1, n):
        x, y, cw, ch, area = stats[label]
        if area < min_area:
            continue
        touches_border = (
            x <= border_margin
            or y <= border_margin
            or x + cw >= w - border_margin
            or y + ch >= h - border_margin
        )
        if touches_border:
            continue
        out[labels == label] = 255
    return out


def build_foreground_mask(image: np.ndarray, cfg: ProductLocatorConfig | None = None) -> np.ndarray:
    """Build a product foreground mask while suppressing dark vignette connected to image borders."""
    cfg = cfg or ProductLocatorConfig()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # The product is darker than the bright background. Border-connected dark regions
    # are removed afterwards, which handles lens/light dark corners.
    mask = np.where(gray < cfg.foreground_threshold, 255, 0).astype(np.uint8)
    mask = _remove_border_components(mask, cfg)

    k = _odd(min(gray.shape) * cfg.close_kernel_ratio)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Merge product sub-parts that are separated by small assembly gaps.
    dilate_k = _odd(max(3, k // 2))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k))
    merged = cv2.dilate(mask, dilate_kernel, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Product foreground not found. Adjust foreground_threshold or lighting.")

    contour = max(contours, key=cv2.contourArea)
    final_mask = np.zeros_like(mask)
    cv2.drawContours(final_mask, [contour], -1, 255, thickness=cv2.FILLED)
    return final_mask


def _principal_angle_deg(contour: np.ndarray) -> float:
    pts = contour.reshape(-1, 2).astype(np.float64)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    axis = vectors[:, int(np.argmax(values))]
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    # Long-axis orientation is 180-degree periodic; normalize around horizontal.
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return angle


def locate_product(image: np.ndarray, cfg: ProductLocatorConfig | None = None) -> ProductLocation:
    cfg = cfg or ProductLocatorConfig()
    mask = build_foreground_mask(image, cfg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("Product contour not found.")

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    m = cv2.moments(contour)
    if abs(m["m00"]) > 1e-6:
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
    else:
        cx = x + w / 2.0
        cy = y + h / 2.0

    return ProductLocation(
        bbox_xywh=(int(x), int(y), int(w), int(h)),
        center_xy=(float(cx), float(cy)),
        angle_deg=_principal_angle_deg(contour),
        area_px=float(cv2.contourArea(contour)),
        mask=mask,
    )


def _rotate_about(image: np.ndarray, center: tuple[float, float], angle_deg: float, border_value: int = 255) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    h, w = image.shape[:2]
    if image.ndim == 3:
        border = (border_value, border_value, border_value)
    else:
        border = border_value
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def _crop_with_padding(image: np.ndarray, bbox: tuple[int, int, int, int], padding_ratio: float) -> np.ndarray:
    x, y, w, h = bbox
    pad_x = int(round(w * padding_ratio))
    pad_y = int(round(h * padding_ratio))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(image.shape[1], x + w + pad_x)
    y1 = min(image.shape[0], y + h + pad_y)
    return image[y0:y1, x0:x1].copy()


def coarse_align(
    image: np.ndarray,
    cfg: ProductLocatorConfig | None = None,
    target_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, ProductLocation, np.ndarray]:
    """Locate product, rotate its long axis to horizontal and crop around it.

    target_size is (width, height). If supplied, the cropped product is resized to it.
    """
    cfg = cfg or ProductLocatorConfig()
    initial = locate_product(image, cfg)

    # Rotate the detected long axis back to horizontal.
    rotated = _rotate_about(image, initial.center_xy, -initial.angle_deg)
    rotated_location = locate_product(rotated, cfg)
    crop = _crop_with_padding(rotated, rotated_location.bbox_xywh, cfg.crop_padding_ratio)

    if target_size is not None:
        crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA if crop.shape[1] > target_size[0] else cv2.INTER_CUBIC)

    return crop, initial, initial.mask


def _ecc_ready(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.normalize(gray, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return gray


def align_to_reference(
    image: np.ndarray,
    reference: np.ndarray,
    cfg: ProductLocatorConfig | None = None,
    ecc_iterations: int = 200,
    ecc_epsilon: float = 1e-6,
    motion: int = cv2.MOTION_AFFINE,
) -> AlignmentResult:
    """Coarse-align an image and refine it against a reference with ECC."""
    cfg = cfg or ProductLocatorConfig()
    target_size = (reference.shape[1], reference.shape[0])
    coarse, location, original_mask = coarse_align(image, cfg, target_size=target_size)

    template = _ecc_ready(reference)
    moving = _ecc_ready(coarse)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_iterations, ecc_epsilon)

    ecc_score: float | None = None
    try:
        ecc_score, warp = cv2.findTransformECC(template, moving, warp, motion, criteria, None, 5)
        aligned = cv2.warpAffine(
            coarse,
            warp,
            target_size,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    except cv2.error:
        # Coarse alignment is still a usable fallback. Keep diagnostics so batch jobs do not die.
        aligned = coarse.copy()
        warp = None

    return AlignmentResult(
        aligned=aligned,
        coarse=coarse,
        foreground_mask=original_mask,
        location=location,
        ecc_score=ecc_score,
        ecc_matrix=warp,
    )


def make_overlay(reference: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    if reference.shape[:2] != aligned.shape[:2]:
        aligned = cv2.resize(aligned, (reference.shape[1], reference.shape[0]))
    return cv2.addWeighted(reference, 0.5, aligned, 0.5, 0.0)
