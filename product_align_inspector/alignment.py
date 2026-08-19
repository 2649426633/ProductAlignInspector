from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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

    # Fixed-camera fast path. Geometry is estimated on a downscaled image,
    # then only similarity transforms are used: uniform scale + rotation + translation.
    geometry_max_dim: int = 1200
    max_abs_coarse_rotation_deg: float = 90.0
    fallback_quadrant_search: bool = True
    fallback_preview_max_dim: int = 520
    fallback_preview_ecc_iterations: int = 24
    fallback_full_ecc_iterations: int = 70
    foreground_scale_search: float = 0.015
    foreground_scale_steps: int = 3
    foreground_ecc_accept_score: float = 0.82

    # SIFT is only a recovery path.
    feature_max_dim: int = 1500
    feature_nfeatures: int = 3500
    feature_ratio_test: float = 0.76
    feature_min_matches: int = 10
    feature_min_inliers: int = 6
    feature_min_inlier_ratio: float = 0.12
    feature_ransac_threshold_px: float = 7.0
    feature_min_scale: float = 0.20
    feature_max_scale: float = 2.00
    ecc_accept_score: float = 0.84


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
    method: str = "unknown"
    feature_matches: int = 0
    feature_inliers: int = 0
    feature_inlier_ratio: float = 0.0
    feature_matrix: np.ndarray | None = None
    fallback_rotation_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "location": self.location.to_dict(),
            "feature_matches": int(self.feature_matches),
            "feature_inliers": int(self.feature_inliers),
            "feature_inlier_ratio": float(self.feature_inlier_ratio),
            "feature_matrix": None if self.feature_matrix is None else self.feature_matrix.tolist(),
            "fallback_rotation_deg": None if self.fallback_rotation_deg is None else float(self.fallback_rotation_deg),
            "ecc_score": None if self.ecc_score is None else float(self.ecc_score),
            "ecc_matrix": None if self.ecc_matrix is None else self.ecc_matrix.tolist(),
            "aligned_shape": list(self.aligned.shape),
            "coarse_shape": list(self.coarse.shape),
        }


@dataclass
class _Geometry:
    location: ProductLocation
    short_px: float
    long_px: float


_REFERENCE_GEOMETRY_CACHE: dict[tuple[int, tuple[int, ...], int, int], _Geometry] = {}


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
        if x <= border_margin or y <= border_margin or x + cw >= w - border_margin or y + ch >= h - border_margin:
            continue
        out[labels == label] = 255
    return out


def build_foreground_mask(image: np.ndarray, cfg: ProductLocatorConfig | None = None) -> np.ndarray:
    cfg = cfg or ProductLocatorConfig()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = np.where(gray < cfg.foreground_threshold, 255, 0).astype(np.uint8)
    mask = _remove_border_components(mask, cfg)
    k = _odd(min(gray.shape) * cfg.close_kernel_ratio)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    dk = _odd(max(3, k // 2))
    merged = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (dk, dk)), iterations=1)
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
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    else:
        cx, cy = x + w / 2.0, y + h / 2.0
    return ProductLocation(
        bbox_xywh=(int(x), int(y), int(w), int(h)),
        center_xy=(float(cx), float(cy)),
        angle_deg=_principal_angle_deg(contour),
        area_px=float(cv2.contourArea(contour)),
        mask=mask,
    )


def _resize_for_geometry(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w))) if max_dim > 0 else 1.0
    if scale >= 0.9999:
        return image, 1.0
    resized = cv2.resize(
        image,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _fast_geometry(image: np.ndarray, cfg: ProductLocatorConfig) -> _Geometry:
    small, scale = _resize_for_geometry(image, int(cfg.geometry_max_dim))
    loc_small = locate_product(small, cfg)
    contours, _ = cv2.findContours(loc_small.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    (_cx, _cy), (rw, rh), _ = cv2.minAreaRect(contour)
    short = max(1.0, float(min(rw, rh)) / scale)
    long = max(1.0, float(max(rw, rh)) / scale)
    x, y, w, h = loc_small.bbox_xywh
    inv = 1.0 / scale
    loc = ProductLocation(
        bbox_xywh=(int(round(x * inv)), int(round(y * inv)), int(round(w * inv)), int(round(h * inv))),
        center_xy=(float(loc_small.center_xy[0] * inv), float(loc_small.center_xy[1] * inv)),
        angle_deg=float(loc_small.angle_deg),
        area_px=float(loc_small.area_px * inv * inv),
        mask=loc_small.mask,
    )
    return _Geometry(location=loc, short_px=short, long_px=long)


def _reference_geometry(reference: np.ndarray, cfg: ProductLocatorConfig) -> _Geometry:
    ptr = int(reference.__array_interface__["data"][0])
    key = (ptr, tuple(reference.shape), int(cfg.foreground_threshold), int(cfg.geometry_max_dim))
    cached = _REFERENCE_GEOMETRY_CACHE.get(key)
    if cached is None:
        cached = _fast_geometry(reference, cfg)
        _REFERENCE_GEOMETRY_CACHE.clear()
        _REFERENCE_GEOMETRY_CACHE[key] = cached
    return cached


def _rotate_about_bound(image: np.ndarray, center: tuple[float, float], angle_deg: float, border_value: int = 255) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    c, s = abs(float(matrix[0, 0])), abs(float(matrix[0, 1]))
    nw = max(1, int(np.ceil(h * s + w * c)))
    nh = max(1, int(np.ceil(h * c + w * s)))
    matrix[0, 2] += nw / 2.0 - center[0]
    matrix[1, 2] += nh / 2.0 - center[1]
    border = (border_value, border_value, border_value) if image.ndim == 3 else border_value
    return cv2.warpAffine(image, matrix, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def _crop_with_padding(image: np.ndarray, bbox: tuple[int, int, int, int], padding_ratio: float) -> np.ndarray:
    x, y, w, h = bbox
    px, py = int(round(w * padding_ratio)), int(round(h * padding_ratio))
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(image.shape[1], x + w + px), min(image.shape[0], y + h + py)
    return image[y0:y1, x0:x1].copy()


def _resize_letterbox(image: np.ndarray, target_size: tuple[int, int], value: int = 255) -> np.ndarray:
    tw, th = target_size
    h, w = image.shape[:2]
    scale = min(float(tw) / max(1, w), float(th) / max(1, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
    shape = (th, tw, image.shape[2]) if image.ndim == 3 else (th, tw)
    canvas = np.full(shape, value, dtype=image.dtype)
    x0, y0 = (tw - nw) // 2, (th - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def coarse_align(image: np.ndarray, cfg: ProductLocatorConfig | None = None, target_size: tuple[int, int] | None = None) -> tuple[np.ndarray, ProductLocation, np.ndarray]:
    cfg = cfg or ProductLocatorConfig()
    initial = locate_product(image, cfg)
    limit = max(0.0, float(cfg.max_abs_coarse_rotation_deg))
    rotation = float(np.clip(-initial.angle_deg, -limit, limit)) if limit > 0.0 else 0.0
    rotated = _rotate_about_bound(image, initial.center_xy, rotation) if abs(rotation) > 1e-3 else image.copy()
    rotated_location = locate_product(rotated, cfg)
    crop = _crop_with_padding(rotated, rotated_location.bbox_xywh, cfg.crop_padding_ratio)
    if crop.size == 0:
        raise RuntimeError("Coarse alignment produced an empty crop.")
    if target_size is not None:
        crop = _resize_letterbox(crop, target_size)
    return crop, initial, initial.mask


def _ecc_ready(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.normalize(gray, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def _scaled_pair(reference: np.ndarray, moving: np.ndarray, max_dim: int) -> tuple[np.ndarray, np.ndarray, float]:
    h, w = reference.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w))) if max_dim > 0 else 1.0
    if scale >= 0.9999:
        return reference, moving, 1.0
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return (
        cv2.resize(reference, size, interpolation=cv2.INTER_AREA),
        cv2.resize(moving, size, interpolation=cv2.INTER_AREA),
        scale,
    )


def _ecc_refine_fast(
    reference: np.ndarray,
    moving: np.ndarray,
    max_dim: int,
    iterations: int,
    epsilon: float,
) -> tuple[np.ndarray, float | None, np.ndarray | None]:
    ref_small, mov_small, scale = _scaled_pair(reference, moving, max_dim)
    template = _ecc_ready(ref_small)
    moving_gray = _ecc_ready(mov_small)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max(10, int(iterations)), epsilon)
    try:
        score, warp = cv2.findTransformECC(template, moving_gray, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
    except cv2.error:
        return moving.copy(), None, None
    full_warp = warp.astype(np.float32).copy()
    if scale < 0.9999:
        full_warp[0, 2] /= scale
        full_warp[1, 2] /= scale
    refined = cv2.warpAffine(
        moving,
        full_warp,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return refined, float(score), full_warp


def _ecc_refine(reference: np.ndarray, moving: np.ndarray, iterations: int, epsilon: float, motion: int) -> tuple[np.ndarray, float | None, np.ndarray | None]:
    # Compatibility helper. We intentionally constrain runtime refinement to EUCLIDEAN.
    return _ecc_refine_fast(reference, moving, max_dim=max(reference.shape[:2]), iterations=iterations, epsilon=epsilon)


def _similarity_matrix(source_center: tuple[float, float], target_center: tuple[float, float], rotation_deg: float, scale: float) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D(source_center, rotation_deg, scale).astype(np.float32)
    mapped = matrix @ np.asarray([source_center[0], source_center[1], 1.0], dtype=np.float32)
    matrix[0, 2] += float(target_center[0]) - float(mapped[0])
    matrix[1, 2] += float(target_center[1]) - float(mapped[1])
    return matrix


def _warp_similarity(image: np.ndarray, source: ProductLocation, target: ProductLocation, target_shape: tuple[int, ...], rotation_deg: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    matrix = _similarity_matrix(source.center_xy, target.center_xy, rotation_deg, scale)
    h, w = target_shape[:2]
    warped = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return warped, matrix


def _normalize_angle(angle: float) -> float:
    return float((angle + 180.0) % 360.0 - 180.0)


def _foreground_similarity_result(image: np.ndarray, reference: np.ndarray, cfg: ProductLocatorConfig, ecc_iterations: int, ecc_epsilon: float) -> AlignmentResult | None:
    try:
        src_geom = _fast_geometry(image, cfg)
        dst_geom = _reference_geometry(reference, cfg)
    except (RuntimeError, cv2.error):
        return None

    source, target = src_geom.location, dst_geom.location
    ratios = np.asarray([
        dst_geom.short_px / src_geom.short_px,
        dst_geom.long_px / src_geom.long_px,
    ], dtype=np.float64)
    base_scale = float(np.sqrt(max(1e-12, ratios[0] * ratios[1])))
    base_rotation = _normalize_angle(target.angle_deg - source.angle_deg)

    # PCA's long-axis direction has only a 180-degree ambiguity. A 90-degree
    # quadrant search is unnecessary for an elongated rigid product and was a major slowdown.
    rotations = [base_rotation, _normalize_angle(base_rotation + 180.0)] if cfg.fallback_quadrant_search else [base_rotation]
    rotations = list(dict.fromkeys(round(x, 6) for x in rotations))

    spread = max(0.0, float(cfg.foreground_scale_search))
    steps = max(1, int(cfg.foreground_scale_steps))
    multipliers = [1.0] if steps == 1 or spread <= 1e-9 else np.linspace(1.0 - spread, 1.0 + spread, steps).tolist()

    # Cheap low-resolution ranking only. 2 orientations x 3 tiny scale choices = 6 candidates.
    candidates: list[tuple[float, float, float, np.ndarray, np.ndarray]] = []
    for rotation in rotations:
        for multiplier in multipliers:
            scale = float(base_scale * multiplier)
            coarse, seed_matrix = _warp_similarity(image, source, target, reference.shape, float(rotation), scale)
            _tmp, score, _ = _ecc_refine_fast(
                reference,
                coarse,
                max_dim=int(cfg.fallback_preview_max_dim),
                iterations=int(cfg.fallback_preview_ecc_iterations),
                epsilon=max(float(ecc_epsilon), 1e-5),
            )
            candidates.append(((-1.0 if score is None else float(score)), float(rotation), scale, coarse, seed_matrix))

    candidates.sort(key=lambda row: row[0], reverse=True)
    if not candidates or candidates[0][0] < 0:
        return None

    _, rotation, _scale, coarse, seed_matrix = candidates[0]
    refined, score, ecc_matrix = _ecc_refine_fast(
        reference,
        coarse,
        max_dim=min(1100, max(reference.shape[:2])),
        iterations=min(max(30, int(ecc_iterations)), int(cfg.fallback_full_ecc_iterations)),
        epsilon=ecc_epsilon,
    )
    if score is None or ecc_matrix is None or score < float(cfg.foreground_ecc_accept_score):
        return None

    return AlignmentResult(
        aligned=refined,
        coarse=coarse,
        foreground_mask=source.mask,
        location=source,
        ecc_score=float(score),
        ecc_matrix=ecc_matrix,
        method="foreground_similarity+ecc_fast",
        feature_matrix=seed_matrix,
        fallback_rotation_deg=float(rotation),
    )


def _resize_for_features(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale < 0.9999:
        resized = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    else:
        resized = image
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized.copy()
    return gray, scale


def _location_from_affine(image_shape: tuple[int, ...], reference_shape: tuple[int, ...], input_to_reference: np.ndarray) -> ProductLocation:
    ih, iw = image_shape[:2]
    rh, rw = reference_shape[:2]
    inv = cv2.invertAffineTransform(input_to_reference.astype(np.float64))
    corners = np.float32([[0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]]).reshape(-1, 1, 2)
    pts = cv2.transform(corners, inv.astype(np.float32)).reshape(-1, 2)
    xs, ys = pts[:, 0], pts[:, 1]
    x0, y0 = max(0, int(np.floor(xs.min()))), max(0, int(np.floor(ys.min())))
    x1, y1 = min(iw, int(np.ceil(xs.max())) + 1), min(ih, int(np.ceil(ys.max())) + 1)
    p0, p1 = pts[0], pts[1]
    angle = float(np.degrees(np.arctan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))))
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    mask = np.zeros((1, 1), dtype=np.uint8)
    center = pts.mean(axis=0)
    return ProductLocation(
        bbox_xywh=(x0, y0, max(0, x1 - x0), max(0, y1 - y0)),
        center_xy=(float(center[0]), float(center[1])),
        angle_deg=angle,
        area_px=float(abs(cv2.contourArea(pts.astype(np.float32)))),
        mask=mask,
    )


def _feature_align(image: np.ndarray, reference: np.ndarray, cfg: ProductLocatorConfig) -> tuple[np.ndarray, ProductLocation, np.ndarray, int, int, float]:
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("This OpenCV build does not provide SIFT.")
    ref_gray, ref_scale = _resize_for_features(reference, cfg.feature_max_dim)
    img_gray, img_scale = _resize_for_features(image, cfg.feature_max_dim)
    sift = cv2.SIFT_create(nfeatures=cfg.feature_nfeatures, contrastThreshold=0.02, edgeThreshold=12)
    ref_kp, ref_desc = sift.detectAndCompute(ref_gray, None)
    img_kp, img_desc = sift.detectAndCompute(img_gray, None)
    if ref_desc is None or img_desc is None or len(ref_kp) < 4 or len(img_kp) < 4:
        raise RuntimeError("Not enough SIFT features.")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(ref_desc, img_desc, k=2)
    good = []
    for pair in pairs:
        if len(pair) >= 2 and pair[0].distance < cfg.feature_ratio_test * pair[1].distance:
            good.append(pair[0])
    if len(good) < cfg.feature_min_matches:
        raise RuntimeError(f"Not enough feature matches: {len(good)} < {cfg.feature_min_matches}")
    ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in good]) / ref_scale
    img_pts = np.float32([img_kp[m.trainIdx].pt for m in good]) / img_scale
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        img_pts,
        ref_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=cfg.feature_ransac_threshold_px,
        maxIters=3000,
        confidence=0.999,
        refineIters=30,
    )
    if matrix is None or inlier_mask is None:
        raise RuntimeError("RANSAC could not estimate product similarity transform.")
    inliers = int(inlier_mask.ravel().sum())
    ratio = float(inliers / max(1, len(good)))
    if inliers < cfg.feature_min_inliers or ratio < cfg.feature_min_inlier_ratio:
        raise RuntimeError(f"Weak feature geometry: inliers={inliers}, ratio={ratio:.3f}")
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    scale = float(np.sqrt(a * a + b * b))
    if not (cfg.feature_min_scale <= scale <= cfg.feature_max_scale):
        raise RuntimeError(f"Estimated scale out of range: {scale:.3f}")
    aligned = cv2.warpAffine(image, matrix.astype(np.float32), (reference.shape[1], reference.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return aligned, _location_from_affine(image.shape, reference.shape, matrix), matrix.astype(np.float32), len(good), inliers, ratio


def _feature_result(image: np.ndarray, reference: np.ndarray, cfg: ProductLocatorConfig, ecc_epsilon: float) -> AlignmentResult | None:
    try:
        coarse, location, matrix, matches, inliers, ratio = _feature_align(image, reference, cfg)
    except (RuntimeError, cv2.error):
        return None
    refined, score, ecc_matrix = _ecc_refine_fast(reference, coarse, max_dim=min(1100, max(reference.shape[:2])), iterations=60, epsilon=ecc_epsilon)
    if score is None or ecc_matrix is None or score < cfg.ecc_accept_score:
        return None
    return AlignmentResult(
        aligned=refined,
        coarse=coarse,
        foreground_mask=location.mask,
        location=location,
        ecc_score=float(score),
        ecc_matrix=ecc_matrix,
        method="sift_similarity+ecc_fast",
        feature_matches=matches,
        feature_inliers=inliers,
        feature_inlier_ratio=ratio,
        feature_matrix=matrix,
    )


def align_to_reference(
    image: np.ndarray,
    reference: np.ndarray,
    cfg: ProductLocatorConfig | None = None,
    ecc_iterations: int = 70,
    ecc_epsilon: float = 1e-5,
    motion: int = cv2.MOTION_EUCLIDEAN,
) -> AlignmentResult:
    """Fast fixed-camera alignment.

    1) downscaled foreground geometry -> similarity seed;
    2) low-resolution ECC selects/refines the seed;
    3) one SIFT similarity recovery only if the fast foreground path fails.

    No X/Y stretch and no affine shear are allowed.
    """
    cfg = cfg or ProductLocatorConfig()
    result = _foreground_similarity_result(image, reference, cfg, ecc_iterations, ecc_epsilon)
    if result is not None:
        return result
    result = _feature_result(image, reference, cfg, ecc_epsilon)
    if result is not None:
        return result
    relaxed = replace(
        cfg,
        feature_max_dim=2200,
        feature_nfeatures=6500,
        feature_ratio_test=0.80,
        feature_min_matches=8,
        feature_min_inliers=6,
        feature_min_inlier_ratio=0.10,
        feature_ransac_threshold_px=9.0,
    )
    result = _feature_result(image, reference, relaxed, ecc_epsilon)
    if result is not None:
        result.method = "sift_similarity_relaxed+ecc_fast"
        return result
    raise RuntimeError("Alignment failed: fast foreground geometry and SIFT recovery both failed.")


def make_overlay(reference: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    if reference.shape[:2] != aligned.shape[:2]:
        aligned = cv2.resize(aligned, (reference.shape[1], reference.shape[0]))
    return cv2.addWeighted(reference, 0.5, aligned, 0.5, 0.0)
