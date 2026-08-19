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

    # Fixed-camera geometry path. The foreground seed is always a similarity
    # transform (uniform scale + rotation + translation), never an X/Y stretch.
    max_abs_coarse_rotation_deg: float = 90.0
    fallback_quadrant_search: bool = True
    fallback_preview_max_dim: int = 900
    fallback_preview_ecc_iterations: int = 80
    fallback_full_ecc_iterations: int = 350
    foreground_scale_search: float = 0.025
    foreground_scale_steps: int = 3
    foreground_ecc_accept_score: float = 0.90

    # SIFT remains as a recovery path. estimateAffinePartial2D is also a
    # similarity transform, so it cannot independently stretch X and Y.
    feature_max_dim: int = 1800
    feature_nfeatures: int = 5000
    feature_ratio_test: float = 0.72
    feature_min_matches: int = 12
    feature_min_inliers: int = 8
    feature_min_inlier_ratio: float = 0.25
    feature_ransac_threshold_px: float = 5.0
    feature_min_scale: float = 0.20
    feature_max_scale: float = 2.00

    # Whole-image quality gate for SIFT paths.
    ecc_accept_score: float = 0.88


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
            "fallback_rotation_deg": None
            if self.fallback_rotation_deg is None
            else float(self.fallback_rotation_deg),
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
    """Return the largest non-border foreground component."""
    cfg = cfg or ProductLocatorConfig()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    mask = np.where(gray < cfg.foreground_threshold, 255, 0).astype(np.uint8)
    mask = _remove_border_components(mask, cfg)

    k = _odd(min(gray.shape) * cfg.close_kernel_ratio)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    dilate_k = _odd(max(3, k // 2))
    merged = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k)),
        iterations=1,
    )

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


def _rotate_about_bound(
    image: np.ndarray,
    center: tuple[float, float],
    angle_deg: float,
    border_value: int = 255,
) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos_value = abs(float(matrix[0, 0]))
    sin_value = abs(float(matrix[0, 1]))
    new_w = max(1, int(np.ceil(h * sin_value + w * cos_value)))
    new_h = max(1, int(np.ceil(h * cos_value + w * sin_value)))
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    border = (border_value, border_value, border_value) if image.ndim == 3 else border_value
    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def _crop_with_padding(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding_ratio: float,
) -> np.ndarray:
    x, y, w, h = bbox
    pad_x = int(round(w * padding_ratio))
    pad_y = int(round(h * padding_ratio))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(image.shape[1], x + w + pad_x)
    y1 = min(image.shape[0], y + h + pad_y)
    return image[y0:y1, x0:x1].copy()


def _resize_letterbox(image: np.ndarray, target_size: tuple[int, int], value: int = 255) -> np.ndarray:
    """Uniformly resize into target_size without X/Y distortion."""
    tw, th = target_size
    h, w = image.shape[:2]
    scale = min(float(tw) / max(1, w), float(th) / max(1, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (nw, nh), interpolation=interpolation)
    if image.ndim == 3:
        canvas = np.full((th, tw, image.shape[2]), value, dtype=image.dtype)
    else:
        canvas = np.full((th, tw), value, dtype=image.dtype)
    x0 = (tw - nw) // 2
    y0 = (th - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _foreground_candidate(
    image: np.ndarray,
    initial: ProductLocation,
    cfg: ProductLocatorConfig,
    rotation_deg: float,
    target_size: tuple[int, int] | None,
) -> np.ndarray:
    """Legacy coarse helper used by reference creation; now preserves aspect ratio."""
    rotated = (
        _rotate_about_bound(image, initial.center_xy, rotation_deg)
        if abs(rotation_deg) > 1e-3
        else image.copy()
    )
    rotated_location = locate_product(rotated, cfg)
    crop = _crop_with_padding(rotated, rotated_location.bbox_xywh, cfg.crop_padding_ratio)
    if crop.size == 0:
        raise RuntimeError("Foreground fallback produced an empty crop.")
    if target_size is not None:
        crop = _resize_letterbox(crop, target_size)
    return crop


def coarse_align(
    image: np.ndarray,
    cfg: ProductLocatorConfig | None = None,
    target_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, ProductLocation, np.ndarray]:
    cfg = cfg or ProductLocatorConfig()
    initial = locate_product(image, cfg)
    limit = max(0.0, float(cfg.max_abs_coarse_rotation_deg))
    rotation = float(np.clip(-initial.angle_deg, -limit, limit)) if limit > 0.0 else 0.0
    crop = _foreground_candidate(image, initial, cfg, rotation, target_size)
    return crop, initial, initial.mask


def _ecc_ready(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.normalize(gray, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def _ecc_refine(
    reference: np.ndarray,
    moving: np.ndarray,
    iterations: int,
    epsilon: float,
    motion: int,
) -> tuple[np.ndarray, float | None, np.ndarray | None]:
    """ECC refinement. Runtime uses EUCLIDEAN to prevent shear/X-Y stretching."""
    target_size = (reference.shape[1], reference.shape[0])
    template = _ecc_ready(reference)
    moving_gray = _ecc_ready(moving)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, epsilon)
    try:
        score, warp = cv2.findTransformECC(template, moving_gray, warp, motion, criteria, None, 5)
        refined = cv2.warpAffine(
            moving,
            warp,
            target_size,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return refined, float(score), warp
    except cv2.error:
        return moving.copy(), None, None


def _oriented_size(mask: np.ndarray) -> tuple[float, float]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No contour for oriented-size estimate.")
    contour = max(contours, key=cv2.contourArea)
    (_cx, _cy), (rw, rh), _angle = cv2.minAreaRect(contour)
    short = max(1.0, float(min(rw, rh)))
    long = max(1.0, float(max(rw, rh)))
    return short, long


def _foreground_scale(source: ProductLocation, target: ProductLocation) -> float:
    src_short, src_long = _oriented_size(source.mask)
    dst_short, dst_long = _oriented_size(target.mask)
    ratios = np.asarray([dst_short / src_short, dst_long / src_long], dtype=np.float64)
    return float(np.sqrt(max(1e-12, ratios[0] * ratios[1])))


def _similarity_matrix(
    source_center: tuple[float, float],
    target_center: tuple[float, float],
    rotation_deg: float,
    scale: float,
) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D(source_center, rotation_deg, scale).astype(np.float32)
    src = np.asarray([source_center[0], source_center[1], 1.0], dtype=np.float32)
    mapped = matrix @ src
    matrix[0, 2] += float(target_center[0]) - float(mapped[0])
    matrix[1, 2] += float(target_center[1]) - float(mapped[1])
    return matrix


def _warp_similarity(
    image: np.ndarray,
    source: ProductLocation,
    target: ProductLocation,
    target_shape: tuple[int, ...],
    rotation_deg: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = _similarity_matrix(source.center_xy, target.center_xy, rotation_deg, scale)
    h, w = target_shape[:2]
    warped = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return warped, matrix


def _preview_pair(
    reference: np.ndarray,
    moving: np.ndarray,
    max_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = reference.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w))) if max_dim > 0 else 1.0
    if scale >= 0.9999:
        return reference, moving
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return (
        cv2.resize(reference, size, interpolation=cv2.INTER_AREA),
        cv2.resize(moving, size, interpolation=cv2.INTER_AREA),
    )


def _foreground_similarity_result(
    image: np.ndarray,
    reference: np.ndarray,
    cfg: ProductLocatorConfig,
    ecc_iterations: int,
    ecc_epsilon: float,
) -> AlignmentResult | None:
    """Fixed-camera alignment using only rigid/similarity geometry."""
    try:
        source = locate_product(image, cfg)
        target = locate_product(reference, cfg)
        base_scale = _foreground_scale(source, target)
    except (RuntimeError, cv2.error):
        return None

    base_rotation = float(target.angle_deg - source.angle_deg)
    rotations = [base_rotation]
    if cfg.fallback_quadrant_search:
        rotations.extend(base_rotation + x for x in (90.0, 180.0, 270.0))

    clean_rotations: list[float] = []
    for angle in rotations:
        normalized = float((angle + 180.0) % 360.0 - 180.0)
        if not any(abs(normalized - old) < 1e-4 for old in clean_rotations):
            clean_rotations.append(normalized)

    spread = max(0.0, float(cfg.foreground_scale_search))
    steps = max(1, int(cfg.foreground_scale_steps))
    if steps == 1 or spread <= 1e-9:
        multipliers = [1.0]
    else:
        multipliers = np.linspace(1.0 - spread, 1.0 + spread, steps).tolist()

    candidates: list[tuple[float, float, float, np.ndarray, np.ndarray]] = []
    preview_iterations = max(40, int(cfg.fallback_preview_ecc_iterations))
    preview_epsilon = max(float(ecc_epsilon), 1e-5)

    for rotation in clean_rotations:
        for multiplier in multipliers:
            scale = float(base_scale * multiplier)
            coarse, seed_matrix = _warp_similarity(
                image,
                source,
                target,
                reference.shape,
                rotation,
                scale,
            )
            pr, pm = _preview_pair(reference, coarse, int(cfg.fallback_preview_max_dim))
            _refined, score, _warp = _ecc_refine(
                pr,
                pm,
                preview_iterations,
                preview_epsilon,
                cv2.MOTION_EUCLIDEAN,
            )
            candidates.append(((-1.0 if score is None else float(score)), rotation, scale, coarse, seed_matrix))

    if not candidates:
        return None
    candidates.sort(key=lambda row: row[0], reverse=True)

    full_iterations = max(int(ecc_iterations), int(cfg.fallback_full_ecc_iterations))
    best: tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    for _preview_score, rotation, scale, coarse, seed_matrix in candidates[:4]:
        refined, score, ecc_matrix = _ecc_refine(
            reference,
            coarse,
            full_iterations,
            ecc_epsilon,
            cv2.MOTION_EUCLIDEAN,
        )
        if score is None or ecc_matrix is None:
            continue
        if best is None or float(score) > best[0]:
            best = (float(score), rotation, scale, coarse, refined, seed_matrix, ecc_matrix)

    if best is None or best[0] < float(cfg.foreground_ecc_accept_score):
        return None

    score, rotation, _scale, coarse, refined, seed_matrix, ecc_matrix = best
    return AlignmentResult(
        aligned=refined,
        coarse=coarse,
        foreground_mask=source.mask,
        location=source,
        ecc_score=score,
        ecc_matrix=ecc_matrix,
        method="foreground_similarity+ecc",
        feature_matrix=seed_matrix,
        fallback_rotation_deg=rotation,
    )


def _resize_for_features(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale < 0.9999:
        resized = cv2.resize(
            image,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized.copy()
    return gray, scale


def _location_from_affine(
    image_shape: tuple[int, ...],
    reference_shape: tuple[int, ...],
    input_to_reference: np.ndarray,
) -> ProductLocation:
    ih, iw = image_shape[:2]
    rh, rw = reference_shape[:2]
    reference_to_input = cv2.invertAffineTransform(input_to_reference.astype(np.float64))
    ref_corners = np.float32(
        [[0.0, 0.0], [rw - 1.0, 0.0], [rw - 1.0, rh - 1.0], [0.0, rh - 1.0]]
    ).reshape(-1, 1, 2)
    input_corners = cv2.transform(ref_corners, reference_to_input.astype(np.float32)).reshape(-1, 2)
    xs, ys = input_corners[:, 0], input_corners[:, 1]
    x0 = max(0, int(np.floor(xs.min())))
    y0 = max(0, int(np.floor(ys.min())))
    x1 = min(iw, int(np.ceil(xs.max())) + 1)
    y1 = min(ih, int(np.ceil(ys.max())) + 1)
    p0, p1 = input_corners[0], input_corners[1]
    angle = float(np.degrees(np.arctan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))))
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    mask = np.zeros((ih, iw), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(input_corners).astype(np.int32), 255)
    center = input_corners.mean(axis=0)
    area = float(abs(cv2.contourArea(input_corners.astype(np.float32))))
    return ProductLocation(
        bbox_xywh=(x0, y0, max(0, x1 - x0), max(0, y1 - y0)),
        center_xy=(float(center[0]), float(center[1])),
        angle_deg=angle,
        area_px=area,
        mask=mask,
    )


def _feature_align(
    image: np.ndarray,
    reference: np.ndarray,
    cfg: ProductLocatorConfig,
) -> tuple[np.ndarray, ProductLocation, np.ndarray, int, int, float]:
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("This OpenCV build does not provide SIFT.")

    ref_gray, ref_scale = _resize_for_features(reference, cfg.feature_max_dim)
    img_gray, img_scale = _resize_for_features(image, cfg.feature_max_dim)
    sift = cv2.SIFT_create(nfeatures=cfg.feature_nfeatures, contrastThreshold=0.02, edgeThreshold=12)
    ref_kp, ref_desc = sift.detectAndCompute(ref_gray, None)
    img_kp, img_desc = sift.detectAndCompute(img_gray, None)
    if ref_desc is None or img_desc is None or len(ref_kp) < 4 or len(img_kp) < 4:
        raise RuntimeError("Not enough SIFT features.")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(ref_desc, img_desc, k=2)
    good_matches = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < cfg.feature_ratio_test * n.distance:
            good_matches.append(m)
    if len(good_matches) < cfg.feature_min_matches:
        raise RuntimeError(f"Not enough feature matches: {len(good_matches)} < {cfg.feature_min_matches}")

    ref_points = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]) / ref_scale
    img_points = np.float32([img_kp[m.trainIdx].pt for m in good_matches]) / img_scale
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        img_points,
        ref_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=cfg.feature_ransac_threshold_px,
        maxIters=5000,
        confidence=0.999,
        refineIters=50,
    )
    if matrix is None or inlier_mask is None:
        raise RuntimeError("RANSAC could not estimate product similarity transform.")

    inliers = int(inlier_mask.ravel().sum())
    inlier_ratio = float(inliers / max(1, len(good_matches)))
    if inliers < cfg.feature_min_inliers:
        raise RuntimeError(f"Not enough feature inliers: {inliers} < {cfg.feature_min_inliers}")
    if inlier_ratio < cfg.feature_min_inlier_ratio:
        raise RuntimeError(
            f"Feature inlier ratio too low: {inlier_ratio:.3f} < {cfg.feature_min_inlier_ratio:.3f}"
        )

    a = float(matrix[0, 0])
    b = float(matrix[0, 1])
    scale = float(np.sqrt(a * a + b * b))
    if not (cfg.feature_min_scale <= scale <= cfg.feature_max_scale):
        raise RuntimeError(
            f"Estimated scale out of range: {scale:.3f} "
            f"(allowed {cfg.feature_min_scale:.2f}..{cfg.feature_max_scale:.2f})"
        )

    target_size = (reference.shape[1], reference.shape[0])
    aligned = cv2.warpAffine(
        image,
        matrix.astype(np.float32),
        target_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    location = _location_from_affine(image.shape, reference.shape, matrix)
    return aligned, location, matrix.astype(np.float32), len(good_matches), inliers, inlier_ratio


def _feature_result(
    image: np.ndarray,
    reference: np.ndarray,
    cfg: ProductLocatorConfig,
    method_prefix: str,
    ecc_accept: float,
    ecc_iterations: int,
    ecc_epsilon: float,
    require_ecc: bool,
) -> AlignmentResult | None:
    try:
        feature_aligned, location, feature_matrix, matches, inliers, inlier_ratio = _feature_align(
            image, reference, cfg
        )
    except (RuntimeError, cv2.error):
        return None

    refined, ecc_score, ecc_matrix = _ecc_refine(
        reference,
        feature_aligned,
        ecc_iterations,
        ecc_epsilon,
        cv2.MOTION_EUCLIDEAN,
    )
    ecc_ok = ecc_score is not None and ecc_score >= ecc_accept
    if require_ecc and not ecc_ok:
        return None
    aligned = refined if ecc_ok else feature_aligned
    method = f"{method_prefix}+ecc" if ecc_ok else method_prefix
    return AlignmentResult(
        aligned=aligned,
        coarse=feature_aligned,
        foreground_mask=location.mask,
        location=location,
        ecc_score=ecc_score,
        ecc_matrix=ecc_matrix if ecc_ok else None,
        method=method,
        feature_matches=matches,
        feature_inliers=inliers,
        feature_inlier_ratio=inlier_ratio,
        feature_matrix=feature_matrix,
    )


def align_to_reference(
    image: np.ndarray,
    reference: np.ndarray,
    cfg: ProductLocatorConfig | None = None,
    ecc_iterations: int = 200,
    ecc_epsilon: float = 1e-6,
    motion: int = cv2.MOTION_EUCLIDEAN,
) -> AlignmentResult:
    """Align a product to the canonical reference without geometric stretching.

    Runtime order for the fixed-camera application:
      1. foreground/PCA similarity seed + EUCLIDEAN ECC;
      2. strict SIFT similarity + EUCLIDEAN ECC;
      3. one relaxed SIFT similarity recovery + EUCLIDEAN ECC.

    `motion` is kept for API compatibility, but runtime refinement is intentionally
    constrained to EUCLIDEAN motion so fixed ROI geometry cannot be sheared.
    """
    cfg = cfg or ProductLocatorConfig()

    foreground = _foreground_similarity_result(
        image,
        reference,
        cfg,
        ecc_iterations=max(300, ecc_iterations),
        ecc_epsilon=ecc_epsilon,
    )
    if foreground is not None:
        return foreground

    strict = _feature_result(
        image,
        reference,
        cfg,
        method_prefix="sift_affine",
        ecc_accept=cfg.ecc_accept_score,
        ecc_iterations=ecc_iterations,
        ecc_epsilon=ecc_epsilon,
        require_ecc=True,
    )
    if strict is not None:
        return strict

    relaxed_cfg = replace(
        cfg,
        feature_max_dim=3200,
        feature_nfeatures=12000,
        feature_ratio_test=0.80,
        feature_min_matches=8,
        feature_min_inliers=6,
        feature_min_inlier_ratio=0.12,
        feature_ransac_threshold_px=8.0,
    )
    recovered = _feature_result(
        image,
        reference,
        relaxed_cfg,
        method_prefix="recovery_relaxed",
        ecc_accept=max(0.90, cfg.ecc_accept_score),
        ecc_iterations=max(300, ecc_iterations),
        ecc_epsilon=ecc_epsilon,
        require_ecc=True,
    )
    if recovered is not None:
        return recovered

    raise RuntimeError(
        "Alignment failed: foreground similarity and SIFT similarity paths did not pass ECC quality gates."
    )


def make_overlay(reference: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    if reference.shape[:2] != aligned.shape[:2]:
        aligned = cv2.resize(aligned, (reference.shape[1], reference.shape[0]))
    return cv2.addWeighted(reference, 0.5, aligned, 0.5, 0.0)
