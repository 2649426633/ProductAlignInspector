from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class ProductLocatorConfig:
    foreground_threshold: int = 238
    background_margin: int = 8
    border_margin_ratio: float = 0.004
    min_component_area_ratio: float = 0.00002
    close_kernel_ratio: float = 0.006

    # SIFT is used only to obtain robust correspondences / a RANSAC inlier set.
    # The FINAL geometry is rigid: rotation + X/Y translation, scale exactly 1.
    feature_max_dim: int = 1800
    feature_nfeatures: int = 6000
    feature_ratio_test: float = 0.72
    feature_min_matches: int = 16
    feature_min_inliers: int = 10
    feature_min_inlier_ratio: float = 0.10
    feature_ransac_threshold_px: float = 5.0

    # Fixed-camera production invariant. Kept as a field for CLI/API compatibility.
    # Only 1.0 is accepted in the rigid path.
    canonical_scale: float | None = 1.0
    canonical_scale_tolerance: float = 0.02

    # Fine refinement is also rigid (cv2.MOTION_EUCLIDEAN).
    ecc_max_dim: int = 1600
    ecc_iterations: int = 120
    ecc_epsilon: float = 1e-5
    ecc_accept_score: float = 0.70

    # Compatibility fields retained for older tools/configuration code.
    geometry_max_dim: int = 1200
    candidate_max_dim: int = 700
    candidate_ambiguity_margin: float = 0.03
    crop_padding_ratio: float = 0.08
    max_abs_coarse_rotation_deg: float = 180.0
    fallback_quadrant_search: bool = False
    fallback_preview_max_dim: int = 900
    fallback_preview_ecc_iterations: int = 80
    fallback_full_ecc_iterations: int = 350
    feature_min_scale: float = 1.0
    feature_max_scale: float = 1.0


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
    # aligned is always in the canonical/reference coordinate system.
    aligned: np.ndarray
    coarse: np.ndarray
    foreground_mask: np.ndarray
    location: ProductLocation
    ecc_score: float | None
    ecc_matrix: np.ndarray | None
    method: str = "sift_ransac_rigid+euclidean_ecc"
    feature_matches: int = 0
    feature_inliers: int = 0
    feature_inlier_ratio: float = 0.0

    # Compatibility: feature_matrix is the final RAW-input -> canonical matrix.
    feature_matrix: np.ndarray | None = None
    input_to_reference: np.ndarray | None = None
    reference_to_input: np.ndarray | None = None

    fallback_rotation_deg: float | None = None
    rigid_rotation_deg: float | None = None
    rigid_translation_xy: tuple[float, float] | None = None
    candidate_score: float | None = None
    canonical_scale: float | None = 1.0

    def to_dict(self) -> dict[str, Any]:
        input_to_reference = self.input_to_reference
        if input_to_reference is None:
            input_to_reference = self.feature_matrix
        return {
            "method": self.method,
            "location": self.location.to_dict(),
            "feature_matches": int(self.feature_matches),
            "feature_inliers": int(self.feature_inliers),
            "feature_inlier_ratio": float(self.feature_inlier_ratio),
            "feature_matrix": None if self.feature_matrix is None else self.feature_matrix.tolist(),
            "input_to_reference": None if input_to_reference is None else input_to_reference.tolist(),
            "reference_to_input": None if self.reference_to_input is None else self.reference_to_input.tolist(),
            "fallback_rotation_deg": None
            if self.fallback_rotation_deg is None
            else float(self.fallback_rotation_deg),
            "ecc_score": None if self.ecc_score is None else float(self.ecc_score),
            "ecc_matrix": None if self.ecc_matrix is None else self.ecc_matrix.tolist(),
            "rigid_rotation_deg": None
            if self.rigid_rotation_deg is None
            else float(self.rigid_rotation_deg),
            "rigid_translation_xy": None
            if self.rigid_translation_xy is None
            else [float(self.rigid_translation_xy[0]), float(self.rigid_translation_xy[1])],
            "candidate_score": None if self.candidate_score is None else float(self.candidate_score),
            "canonical_scale": 1.0,
            "aligned_shape": list(self.aligned.shape),
            "coarse_shape": list(self.coarse.shape),
        }


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def _resize_max(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    if max_dim <= 0:
        return image.copy(), 1.0
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale >= 0.999999:
        return image.copy(), 1.0
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def _gray_uint8(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _feature_image(image: np.ndarray) -> np.ndarray:
    gray = _gray_uint8(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _registration_image(image: np.ndarray) -> np.ndarray:
    gray = _feature_image(image)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    magnitude = cv2.GaussianBlur(magnitude, (5, 5), 0)
    return cv2.normalize(
        magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )


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
        if (
            x <= border_margin
            or y <= border_margin
            or x + cw >= w - border_margin
            or y + ch >= h - border_margin
        ):
            continue
        out[labels == label] = 255
    return out


def build_foreground_mask(
    image: np.ndarray,
    cfg: ProductLocatorConfig | None = None,
) -> np.ndarray:
    cfg = cfg or ProductLocatorConfig()
    gray = cv2.GaussianBlur(_gray_uint8(image), (5, 5), 0)
    background_level = float(np.percentile(gray, 95.0))
    dynamic_threshold = int(
        np.clip(background_level - float(cfg.background_margin), 40.0, 250.0)
    )
    threshold = min(int(cfg.foreground_threshold), dynamic_threshold)
    mask = np.where(gray < threshold, 255, 0).astype(np.uint8)
    mask = _remove_border_components(mask, cfg)
    k = _odd(int(round(min(gray.shape) * cfg.close_kernel_ratio)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Product foreground not found.")
    contour = max(contours, key=cv2.contourArea)
    final_mask = np.zeros_like(mask)
    cv2.drawContours(final_mask, [contour], -1, 255, cv2.FILLED)
    return final_mask


def _principal_angle_deg(contour: np.ndarray) -> float:
    pts = contour.reshape(-1, 2).astype(np.float64)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    axis = vectors[:, int(np.argmax(values))]
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return angle


def _location_from_mask(mask: np.ndarray) -> ProductLocation:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("Product contour not found.")
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-6:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
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


def locate_product(
    image: np.ndarray,
    cfg: ProductLocatorConfig | None = None,
) -> ProductLocation:
    cfg = cfg or ProductLocatorConfig()
    return _location_from_mask(build_foreground_mask(image, cfg))


def _diagnostic_location(image: np.ndarray, cfg: ProductLocatorConfig) -> ProductLocation:
    try:
        return locate_product(image, cfg)
    except Exception:
        h, w = image.shape[:2]
        return ProductLocation(
            bbox_xywh=(0, 0, int(w), int(h)),
            center_xy=(w / 2.0, h / 2.0),
            angle_deg=0.0,
            area_px=float(w * h),
            mask=np.zeros((h, w), dtype=np.uint8),
        )


def _compose_affine(after: np.ndarray, before: np.ndarray) -> np.ndarray:
    a = np.vstack([np.asarray(after, dtype=np.float64), [0.0, 0.0, 1.0]])
    b = np.vstack([np.asarray(before, dtype=np.float64), [0.0, 0.0, 1.0]])
    return (a @ b)[:2].astype(np.float32)


def _matrix_scale(matrix: np.ndarray) -> float:
    m = np.asarray(matrix, dtype=np.float64)
    sx = float(np.hypot(m[0, 0], m[1, 0]))
    sy = float(np.hypot(m[0, 1], m[1, 1]))
    if abs(sx - sy) > 1e-3 * max(1.0, sx, sy):
        raise RuntimeError("Non-uniform scale detected; refusing non-rigid transform.")
    return (sx + sy) * 0.5


def _matrix_rotation_deg(matrix: np.ndarray) -> float:
    m = np.asarray(matrix, dtype=np.float64)
    return float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))


def _solve_rigid(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    """Least-squares 2D rigid transform dst = R*src + t, with scale fixed at 1."""
    src = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_xy, dtype=np.float64).reshape(-1, 2)
    if len(src) < 2 or len(src) != len(dst):
        raise RuntimeError("Rigid solve needs at least two paired points.")

    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)
    x = src - src_center
    y = dst - dst_center
    u, _, vt = np.linalg.svd(x.T @ y)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T

    translation = dst_center - rotation @ src_center
    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[:, :2] = rotation.astype(np.float32)
    matrix[:, 2] = translation.astype(np.float32)

    scale = _matrix_scale(matrix)
    orthogonality = abs(float(np.dot(matrix[:, 0], matrix[:, 1])))
    if abs(scale - 1.0) > 1e-5 or orthogonality > 1e-5:
        raise RuntimeError("Rigid solve produced non-rigid geometry.")
    return matrix


def _estimate_rigid_from_features(
    image: np.ndarray,
    reference: np.ndarray,
    cfg: ProductLocatorConfig,
) -> tuple[np.ndarray, int, int, float]:
    moving_small, moving_scale = _resize_max(image, int(cfg.feature_max_dim))
    reference_small, reference_scale = _resize_max(reference, int(cfg.feature_max_dim))

    sift = cv2.SIFT_create(nfeatures=max(500, int(cfg.feature_nfeatures)))
    moving_kp, moving_desc = sift.detectAndCompute(_feature_image(moving_small), None)
    reference_kp, reference_desc = sift.detectAndCompute(
        _feature_image(reference_small), None
    )
    if moving_desc is None or reference_desc is None:
        raise RuntimeError("SIFT descriptors not found.")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(moving_desc, reference_desc, k=2)
    good = []
    ratio = float(cfg.feature_ratio_test)
    for pair in pairs:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            good.append(first)

    if len(good) < int(cfg.feature_min_matches):
        raise RuntimeError(
            f"Alignment skipped: only {len(good)} reliable feature matches "
            f"(< {cfg.feature_min_matches})."
        )

    src = np.float32(
        [
            [
                moving_kp[m.queryIdx].pt[0] / moving_scale,
                moving_kp[m.queryIdx].pt[1] / moving_scale,
            ]
            for m in good
        ]
    ).reshape(-1, 1, 2)
    dst = np.float32(
        [
            [
                reference_kp[m.trainIdx].pt[0] / reference_scale,
                reference_kp[m.trainIdx].pt[1] / reference_scale,
            ]
            for m in good
        ]
    ).reshape(-1, 1, 2)

    # Similarity RANSAC is used ONLY to identify a robust correspondence set.
    # Its estimated scale is never used in the final geometry.
    _, inlier_mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(cfg.feature_ransac_threshold_px),
        maxIters=5000,
        confidence=0.995,
        refineIters=20,
    )
    if inlier_mask is None:
        raise RuntimeError("RANSAC inlier estimation failed.")

    inlier_flags = inlier_mask.reshape(-1).astype(bool)
    inliers = int(np.count_nonzero(inlier_flags))
    inlier_ratio = float(inliers / max(1, len(good)))
    if inliers < int(cfg.feature_min_inliers):
        raise RuntimeError(
            f"Alignment skipped: {inliers} inliers (< {cfg.feature_min_inliers})."
        )
    if inlier_ratio < float(cfg.feature_min_inlier_ratio):
        raise RuntimeError(
            f"Alignment skipped: inlier ratio {inlier_ratio:.3f} "
            f"(< {cfg.feature_min_inlier_ratio:.3f})."
        )

    if cfg.canonical_scale is not None and abs(float(cfg.canonical_scale) - 1.0) > 1e-6:
        raise RuntimeError(
            "Rigid production path requires canonical_scale=1.0. "
            "Per-frame scale estimation is disabled."
        )

    rigid = _solve_rigid(
        src.reshape(-1, 2)[inlier_flags],
        dst.reshape(-1, 2)[inlier_flags],
    )
    return rigid, len(good), inliers, inlier_ratio


def _ecc_refine(
    reference: np.ndarray,
    coarse: np.ndarray,
    cfg: ProductLocatorConfig,
    ecc_iterations: int | None,
    ecc_epsilon: float | None,
) -> tuple[float, np.ndarray]:
    reference_small, scale = _resize_max(reference, int(cfg.ecc_max_dim))
    coarse_small, coarse_scale = _resize_max(coarse, int(cfg.ecc_max_dim))
    if (
        abs(scale - coarse_scale) > 1e-6
        or reference_small.shape[:2] != coarse_small.shape[:2]
    ):
        raise RuntimeError("Internal ECC size mismatch.")

    template = _registration_image(reference_small)
    moving = _registration_image(coarse_small)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max(20, int(cfg.ecc_iterations if ecc_iterations is None else ecc_iterations)),
        max(1e-8, float(cfg.ecc_epsilon if ecc_epsilon is None else ecc_epsilon)),
    )
    try:
        score, warp_template_to_moving = cv2.findTransformECC(
            template,
            moving,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            5,
        )
    except cv2.error as exc:
        raise RuntimeError(f"ECC refinement failed: {exc}") from exc

    refine_small = cv2.invertAffineTransform(
        warp_template_to_moving
    ).astype(np.float32)
    refine_full = refine_small.copy()
    refine_full[:, 2] /= max(scale, 1e-12)
    return float(score), refine_full


def coarse_align(
    image: np.ndarray,
    cfg: ProductLocatorConfig | None = None,
    target_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, ProductLocation, np.ndarray]:
    cfg = cfg or ProductLocatorConfig()
    location = _diagnostic_location(image, cfg)
    if target_size is not None and tuple(map(int, target_size)) != (
        image.shape[1],
        image.shape[0],
    ):
        raise RuntimeError(
            "coarse_align cannot map to a different coordinate system without a reference image."
        )
    return image.copy(), location, location.mask


def align_to_reference(
    image: np.ndarray,
    reference: np.ndarray,
    cfg: ProductLocatorConfig | None = None,
    ecc_iterations: int | None = None,
    ecc_epsilon: float | None = None,
    motion: int = cv2.MOTION_EUCLIDEAN,
) -> AlignmentResult:
    """Align RAW input directly into the canonical detection-template canvas.

    The input/reference canvas sizes may differ, but the physical product pixel
    scale is fixed. Per-frame geometry is therefore strictly:
      - rotation
      - X translation
      - Y translation

    SIFT/RANSAC supplies robust point correspondences only. The final transform
    is re-solved with scale=1.0, then refined by Euclidean ECC. No scale, shear,
    non-uniform resize, or perspective is permitted.
    """
    del motion
    cfg = cfg or ProductLocatorConfig()
    if image is None or reference is None or image.size == 0 or reference.size == 0:
        raise RuntimeError("Input/reference image is empty.")

    coarse_matrix, matches, inliers, inlier_ratio = _estimate_rigid_from_features(
        image, reference, cfg
    )
    if abs(_matrix_scale(coarse_matrix) - 1.0) > 1e-5:
        raise RuntimeError("Coarse rigid invariant violated.")

    ref_h, ref_w = reference.shape[:2]
    border = (255, 255, 255) if image.ndim == 3 else 255
    coarse = cv2.warpAffine(
        image,
        coarse_matrix,
        (ref_w, ref_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )

    ecc_score, refine = _ecc_refine(
        reference, coarse, cfg, ecc_iterations, ecc_epsilon
    )
    if ecc_score < float(cfg.ecc_accept_score):
        raise RuntimeError(
            f"Alignment skipped: ECC={ecc_score:.4f} < {cfg.ecc_accept_score:.2f}."
        )

    final_matrix = _compose_affine(refine, coarse_matrix)
    final_scale = _matrix_scale(final_matrix)
    orthogonality = abs(
        float(np.dot(final_matrix[:, 0], final_matrix[:, 1]))
    )
    if abs(final_scale - 1.0) > 1e-3 or orthogonality > 1e-3:
        raise RuntimeError(
            "Alignment invariant violated: final transform is not rigid scale=1."
        )

    aligned = cv2.warpAffine(
        image,
        final_matrix,
        (ref_w, ref_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    inverse = cv2.invertAffineTransform(final_matrix).astype(np.float32)
    location = _diagnostic_location(image, cfg)
    rotation = _matrix_rotation_deg(final_matrix)
    tx = float(final_matrix[0, 2])
    ty = float(final_matrix[1, 2])

    return AlignmentResult(
        aligned=aligned,
        coarse=coarse,
        foreground_mask=location.mask,
        location=location,
        ecc_score=ecc_score,
        ecc_matrix=refine,
        method="sift_ransac_rigid+euclidean_ecc",
        feature_matches=matches,
        feature_inliers=inliers,
        feature_inlier_ratio=inlier_ratio,
        feature_matrix=final_matrix,
        input_to_reference=final_matrix,
        reference_to_input=inverse,
        rigid_rotation_deg=rotation,
        rigid_translation_xy=(tx, ty),
        candidate_score=inlier_ratio,
        canonical_scale=1.0,
    )


def make_overlay(reference: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    if reference.shape[:2] != aligned.shape[:2]:
        raise RuntimeError(
            f"Overlay size mismatch: reference={reference.shape[1]}x{reference.shape[0]}, "
            f"aligned={aligned.shape[1]}x{aligned.shape[0]}"
        )
    return cv2.addWeighted(reference, 0.5, aligned, 0.5, 0.0)
