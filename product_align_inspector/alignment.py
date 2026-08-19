from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class ProductLocatorConfig:
    # Foreground extraction. The background is expected to be brighter than
    # the product, but the threshold is adapted from each image so lighting
    # changes do not change the geometry model.
    foreground_threshold: int = 238
    background_margin: int = 8
    border_margin_ratio: float = 0.004
    min_component_area_ratio: float = 0.00002
    close_kernel_ratio: float = 0.006

    # Fixed-camera rigid registration. Only rotation + X/Y translation are
    # allowed. Scale is always exactly 1.0.
    geometry_max_dim: int = 1200
    candidate_max_dim: int = 700
    ecc_max_dim: int = 1600
    candidate_ambiguity_margin: float = 0.03
    ecc_iterations: int = 120
    ecc_epsilon: float = 1e-5
    ecc_accept_score: float = 0.70

    # Kept only for compatibility with older tools/configuration code.
    crop_padding_ratio: float = 0.08
    max_abs_coarse_rotation_deg: float = 90.0
    fallback_quadrant_search: bool = True
    fallback_preview_max_dim: int = 900
    fallback_preview_ecc_iterations: int = 80
    fallback_full_ecc_iterations: int = 350
    feature_max_dim: int = 1800
    feature_nfeatures: int = 5000
    feature_ratio_test: float = 0.72
    feature_min_matches: int = 12
    feature_min_inliers: int = 8
    feature_min_inlier_ratio: float = 0.25
    feature_ransac_threshold_px: float = 5.0
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
    aligned: np.ndarray
    coarse: np.ndarray
    foreground_mask: np.ndarray
    location: ProductLocation
    ecc_score: float | None
    ecc_matrix: np.ndarray | None
    method: str = "rigid_pca_gradient_ecc"
    feature_matches: int = 0
    feature_inliers: int = 0
    feature_inlier_ratio: float = 0.0
    feature_matrix: np.ndarray | None = None
    fallback_rotation_deg: float | None = None
    rigid_rotation_deg: float | None = None
    rigid_translation_xy: tuple[float, float] | None = None
    candidate_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "location": self.location.to_dict(),
            "feature_matches": int(self.feature_matches),
            "feature_inliers": int(self.feature_inliers),
            "feature_inlier_ratio": float(self.feature_inlier_ratio),
            # Compatibility: feature_matrix now stores the FINAL rigid
            # input->reference matrix.
            "feature_matrix": None if self.feature_matrix is None else self.feature_matrix.tolist(),
            "fallback_rotation_deg": None
            if self.fallback_rotation_deg is None
            else float(self.fallback_rotation_deg),
            "ecc_score": None if self.ecc_score is None else float(self.ecc_score),
            # ecc_matrix stores the coarse->reference rigid ECC refinement.
            "ecc_matrix": None if self.ecc_matrix is None else self.ecc_matrix.tolist(),
            "rigid_rotation_deg": None
            if self.rigid_rotation_deg is None
            else float(self.rigid_rotation_deg),
            "rigid_translation_xy": None
            if self.rigid_translation_xy is None
            else [float(self.rigid_translation_xy[0]), float(self.rigid_translation_xy[1])],
            "candidate_score": None if self.candidate_score is None else float(self.candidate_score),
            "aligned_shape": list(self.aligned.shape),
            "coarse_shape": list(self.coarse.shape),
        }


@dataclass
class _Geometry:
    center_xy: tuple[float, float]
    angle_deg: float
    bbox_xywh: tuple[int, int, int, int]
    area_px: float
    small_mask: np.ndarray
    small_scale: float


@dataclass
class _ReferenceState:
    geometry: _Geometry
    candidate_image: np.ndarray
    candidate_registration: np.ndarray
    candidate_scale: float
    ecc_image: np.ndarray
    ecc_registration: np.ndarray
    ecc_mask: np.ndarray
    ecc_scale: float


_REFERENCE_CACHE_KEY: tuple[Any, ...] | None = None
_REFERENCE_CACHE_STATE: _ReferenceState | None = None


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
    """Extract the product silhouette while tolerating moderate lighting change."""
    cfg = cfg or ProductLocatorConfig()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Estimate the current bright-background level, then keep the historical
    # threshold as an upper bound. This prevents a darker exposure from turning
    # the whole frame into foreground.
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

    # Merge nearby product fragments, then keep only the largest object.
    dilate_k = _odd(max(3, k // 2))
    merged = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k)),
        iterations=1,
    )
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError(
            "Product foreground not found. Check lighting/background or foreground_threshold."
        )

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


def _locate_geometry_fast(image: np.ndarray, cfg: ProductLocatorConfig) -> _Geometry:
    small, scale = _resize_max(image, int(cfg.geometry_max_dim))
    mask = build_foreground_mask(small, cfg)
    loc = _location_from_mask(mask)
    inv = 1.0 / max(scale, 1e-12)
    x, y, w, h = loc.bbox_xywh
    return _Geometry(
        center_xy=(loc.center_xy[0] * inv, loc.center_xy[1] * inv),
        angle_deg=float(loc.angle_deg),
        bbox_xywh=(
            int(round(x * inv)),
            int(round(y * inv)),
            int(round(w * inv)),
            int(round(h * inv)),
        ),
        area_px=float(loc.area_px * inv * inv),
        small_mask=mask,
        small_scale=scale,
    )


def _geometry_as_location(
    geometry: _Geometry,
    full_shape: tuple[int, ...],
) -> ProductLocation:
    h, w = full_shape[:2]
    full_mask = cv2.resize(
        geometry.small_mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )
    return ProductLocation(
        bbox_xywh=geometry.bbox_xywh,
        center_xy=geometry.center_xy,
        angle_deg=geometry.angle_deg,
        area_px=geometry.area_px,
        mask=full_mask,
    )


def _registration_image(image: np.ndarray) -> np.ndarray:
    """Lighting-resistant image used only for geometric registration."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    magnitude = cv2.GaussianBlur(magnitude, (5, 5), 0)
    return cv2.normalize(
        magnitude,
        None,
        0.0,
        1.0,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )


def _rigid_matrix(
    source_center: tuple[float, float],
    target_center: tuple[float, float],
    rotation_deg: float,
) -> np.ndarray:
    # scale=1.0 is deliberate and must never change in the fixed-camera path.
    matrix = cv2.getRotationMatrix2D(source_center, rotation_deg, 1.0).astype(np.float32)
    source = np.asarray([source_center[0], source_center[1], 1.0], dtype=np.float32)
    mapped = matrix @ source
    matrix[0, 2] += float(target_center[0]) - float(mapped[0])
    matrix[1, 2] += float(target_center[1]) - float(mapped[1])
    return matrix


def _scale_affine(matrix: np.ndarray, scale: float) -> np.ndarray:
    scaled = matrix.astype(np.float32).copy()
    scaled[:, 2] *= float(scale)
    return scaled


def _compose_affine(after: np.ndarray, before: np.ndarray) -> np.ndarray:
    a = np.vstack([after.astype(np.float64), [0.0, 0.0, 1.0]])
    b = np.vstack([before.astype(np.float64), [0.0, 0.0, 1.0]])
    return (a @ b)[:2].astype(np.float32)


def _normalized_dot(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1).astype(np.float32, copy=False)
    bv = b.reshape(-1).astype(np.float32, copy=False)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-12:
        return -1.0
    return float(np.dot(av, bv) / denom)


def _ecc_refine_rigid(
    reference_registration: np.ndarray,
    moving_registration: np.ndarray,
    reference_mask: np.ndarray | None,
    iterations: int,
    epsilon: float,
) -> tuple[float, np.ndarray]:
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max(20, int(iterations)),
        max(1e-8, float(epsilon)),
    )
    score, warp = cv2.findTransformECC(
        reference_registration,
        moving_registration,
        warp,
        cv2.MOTION_EUCLIDEAN,
        criteria,
        reference_mask,
        5,
    )
    return float(score), warp.astype(np.float32)


def _reference_cache_key(
    reference: np.ndarray,
    cfg: ProductLocatorConfig,
) -> tuple[Any, ...]:
    return (
        id(reference),
        reference.shape,
        str(reference.dtype),
        int(cfg.geometry_max_dim),
        int(cfg.candidate_max_dim),
        int(cfg.ecc_max_dim),
        int(cfg.foreground_threshold),
        int(cfg.background_margin),
        float(cfg.border_margin_ratio),
        float(cfg.min_component_area_ratio),
        float(cfg.close_kernel_ratio),
    )


def _prepare_reference(
    reference: np.ndarray,
    cfg: ProductLocatorConfig,
) -> _ReferenceState:
    global _REFERENCE_CACHE_KEY, _REFERENCE_CACHE_STATE
    key = _reference_cache_key(reference, cfg)
    if _REFERENCE_CACHE_KEY == key and _REFERENCE_CACHE_STATE is not None:
        return _REFERENCE_CACHE_STATE

    geometry = _locate_geometry_fast(reference, cfg)

    candidate_image, candidate_scale = _resize_max(reference, int(cfg.candidate_max_dim))
    candidate_registration = _registration_image(candidate_image)

    ecc_image, ecc_scale = _resize_max(reference, int(cfg.ecc_max_dim))
    ecc_registration = _registration_image(ecc_image)

    # Focus ECC on the product and a small margin around it.
    ref_mask_small = geometry.small_mask
    ref_mask_ecc = cv2.resize(
        ref_mask_small,
        (ecc_image.shape[1], ecc_image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    dilate_px = max(3, int(round(min(ref_mask_ecc.shape) * 0.02)))
    if dilate_px % 2 == 0:
        dilate_px += 1
    ecc_mask = cv2.dilate(
        ref_mask_ecc,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px)),
        iterations=1,
    )

    state = _ReferenceState(
        geometry=geometry,
        candidate_image=candidate_image,
        candidate_registration=candidate_registration,
        candidate_scale=candidate_scale,
        ecc_image=ecc_image,
        ecc_registration=ecc_registration,
        ecc_mask=ecc_mask,
        ecc_scale=ecc_scale,
    )
    _REFERENCE_CACHE_KEY = key
    _REFERENCE_CACHE_STATE = state
    return state


def _candidate_matrices(
    source: _Geometry,
    target: _Geometry,
) -> list[tuple[float, np.ndarray]]:
    # PCA angle follows image-coordinate convention. To undo the source pose,
    # the correct OpenCV rotation is source_angle - target_angle.
    base_rotation = float(source.angle_deg - target.angle_deg)
    candidates: list[tuple[float, np.ndarray]] = []
    for offset in (0.0, 90.0, 180.0, 270.0):
        rotation = float((base_rotation + offset + 180.0) % 360.0 - 180.0)
        matrix = _rigid_matrix(source.center_xy, target.center_xy, rotation)
        candidates.append((rotation, matrix))
    return candidates


def coarse_align(
    image: np.ndarray,
    cfg: ProductLocatorConfig | None = None,
    target_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, ProductLocation, np.ndarray]:
    """Compatibility helper.

    In the rigid workflow a canonical reference is a real RAW GOOD image, so
    this function never crops, stretches, or rescales the product.
    """
    cfg = cfg or ProductLocatorConfig()
    h, w = image.shape[:2]
    if target_size is not None and tuple(map(int, target_size)) != (w, h):
        raise RuntimeError(
            "Rigid workflow forbids resize. Reference and input must use the same raw resolution."
        )
    geometry = _locate_geometry_fast(image, cfg)
    location = _geometry_as_location(geometry, image.shape)
    return image.copy(), location, location.mask


def align_to_reference(
    image: np.ndarray,
    reference: np.ndarray,
    cfg: ProductLocatorConfig | None = None,
    ecc_iterations: int | None = None,
    ecc_epsilon: float | None = None,
    motion: int = cv2.MOTION_EUCLIDEAN,
) -> AlignmentResult:
    """Rigidly align an image to a RAW fixed-camera reference.

    Allowed geometry:
      - X translation
      - Y translation
      - in-plane rotation

    Forbidden geometry:
      - scale
      - independent X/Y resize
      - shear
      - perspective

    `motion` is accepted only for API compatibility; runtime always uses
    cv2.MOTION_EUCLIDEAN.
    """
    del motion
    cfg = cfg or ProductLocatorConfig()

    if image.shape[:2] != reference.shape[:2]:
        ih, iw = image.shape[:2]
        rh, rw = reference.shape[:2]
        raise RuntimeError(
            "Rigid alignment requires identical RAW resolution: "
            f"input={iw}x{ih}, reference={rw}x{rh}. "
            "Do not use a resized/inpainted preview as the geometry reference."
        )

    ref_state = _prepare_reference(reference, cfg)
    source_geometry = _locate_geometry_fast(image, cfg)

    candidate_image, candidate_scale = _resize_max(image, int(cfg.candidate_max_dim))
    candidate_registration = _registration_image(candidate_image)
    if abs(candidate_scale - ref_state.candidate_scale) > 1e-6:
        raise RuntimeError("Internal candidate scale mismatch.")

    scored: list[tuple[float, float, np.ndarray]] = []
    target_size_candidate = (
        ref_state.candidate_registration.shape[1],
        ref_state.candidate_registration.shape[0],
    )
    for rotation, matrix in _candidate_matrices(source_geometry, ref_state.geometry):
        small_matrix = _scale_affine(matrix, candidate_scale)
        warped = cv2.warpAffine(
            candidate_registration,
            small_matrix,
            target_size_candidate,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        score = _normalized_dot(ref_state.candidate_registration, warped)
        scored.append((score, rotation, matrix))

    scored.sort(key=lambda row: row[0], reverse=True)
    if not scored:
        raise RuntimeError("Rigid alignment could not create a pose candidate.")

    # Usually one candidate is clearly best. If the product is close to
    # 180-degree symmetric, evaluate the top two with ECC and let the internal
    # structure choose the orientation.
    selected = [scored[0]]
    if len(scored) > 1 and scored[0][0] - scored[1][0] < float(cfg.candidate_ambiguity_margin):
        selected.append(scored[1])

    input_ecc, input_ecc_scale = _resize_max(image, int(cfg.ecc_max_dim))
    if abs(input_ecc_scale - ref_state.ecc_scale) > 1e-6:
        raise RuntimeError("Internal ECC scale mismatch.")
    input_ecc_registration = _registration_image(input_ecc)

    iterations = cfg.ecc_iterations if ecc_iterations is None else int(ecc_iterations)
    epsilon = cfg.ecc_epsilon if ecc_epsilon is None else float(ecc_epsilon)

    best: tuple[float, float, float, np.ndarray, np.ndarray] | None = None
    # score, candidate_score, rotation, final_matrix, refinement_matrix
    for candidate_score, rotation, coarse_matrix in selected:
        coarse_small = cv2.warpAffine(
            input_ecc_registration,
            _scale_affine(coarse_matrix, input_ecc_scale),
            (
                ref_state.ecc_registration.shape[1],
                ref_state.ecc_registration.shape[0],
            ),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        try:
            ecc_score, warp_template_to_moving = _ecc_refine_rigid(
                ref_state.ecc_registration,
                coarse_small,
                ref_state.ecc_mask,
                iterations,
                epsilon,
            )
        except cv2.error:
            continue

        # OpenCV's ECC warp is applied to the moving image with WARP_INVERSE_MAP,
        # therefore invert it to get coarse->reference.
        refine_small = cv2.invertAffineTransform(warp_template_to_moving).astype(np.float32)
        refine_full = refine_small.copy()
        refine_full[:, 2] /= max(input_ecc_scale, 1e-12)
        final_matrix = _compose_affine(refine_full, coarse_matrix)

        if best is None or ecc_score > best[0]:
            best = (
                float(ecc_score),
                float(candidate_score),
                float(rotation),
                final_matrix,
                refine_full,
            )

    if best is None:
        raise RuntimeError(
            "Rigid ECC refinement failed. Check foreground extraction and lighting."
        )

    ecc_score, candidate_score, coarse_rotation, final_matrix, refine_full = best
    if ecc_score < float(cfg.ecc_accept_score):
        raise RuntimeError(
            "Rigid alignment rejected: "
            f"ECC={ecc_score:.4f} < {cfg.ecc_accept_score:.2f}, "
            f"candidate_score={candidate_score:.4f}, "
            f"coarse_rotation={coarse_rotation:.2f} deg."
        )

    # Guard the core assumption explicitly: the final 2x2 block must stay a
    # pure rotation with unit scale.
    a = float(final_matrix[0, 0])
    b = float(final_matrix[0, 1])
    c = float(final_matrix[1, 0])
    d = float(final_matrix[1, 1])
    scale_x = float(np.hypot(a, c))
    scale_y = float(np.hypot(b, d))
    orthogonality = abs(a * b + c * d)
    if (
        abs(scale_x - 1.0) > 1e-3
        or abs(scale_y - 1.0) > 1e-3
        or orthogonality > 1e-3
    ):
        raise RuntimeError(
            "Rigid transform invariant violated; refusing non-rigid geometry."
        )

    h, w = reference.shape[:2]
    border = (255, 255, 255) if image.ndim == 3 else 255
    aligned = cv2.warpAffine(
        image,
        final_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )

    source_location = _geometry_as_location(source_geometry, image.shape)
    # Avoid a second full-resolution warp purely for diagnostics.
    coarse = aligned.copy()

    rotation_deg = float(np.degrees(np.arctan2(final_matrix[0, 1], final_matrix[0, 0])))
    tx = float(final_matrix[0, 2])
    ty = float(final_matrix[1, 2])

    return AlignmentResult(
        aligned=aligned,
        coarse=coarse,
        foreground_mask=source_location.mask,
        location=source_location,
        ecc_score=ecc_score,
        ecc_matrix=refine_full,
        method="rigid_pca_gradient_ecc",
        feature_matrix=final_matrix,
        fallback_rotation_deg=coarse_rotation,
        rigid_rotation_deg=rotation_deg,
        rigid_translation_xy=(tx, ty),
        candidate_score=candidate_score,
    )


def make_overlay(reference: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    if reference.shape[:2] != aligned.shape[:2]:
        raise RuntimeError(
            "Rigid overlay requires identical image size; resize is intentionally forbidden."
        )
    return cv2.addWeighted(reference, 0.5, aligned, 0.5, 0.0)
