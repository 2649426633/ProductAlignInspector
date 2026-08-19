from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from product_align_inspector.canonical_frame import CanonicalCalibration, invert_affine
from product_align_inspector.io_utils import read_image, write_image, write_json


def resize_for_features(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray, scale


def estimate_fixed_similarity(
    raw_reference: np.ndarray,
    canonical_reference: np.ndarray,
    *,
    max_dim: int = 2200,
    nfeatures: int = 10000,
    ratio_test: float = 0.78,
) -> tuple[np.ndarray, int, int, float]:
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("This OpenCV build does not provide SIFT.")

    raw_gray, raw_scale = resize_for_features(raw_reference, max_dim)
    can_gray, can_scale = resize_for_features(canonical_reference, max_dim)

    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=0.015,
        edgeThreshold=14,
    )
    raw_kp, raw_desc = sift.detectAndCompute(raw_gray, None)
    can_kp, can_desc = sift.detectAndCompute(can_gray, None)
    if raw_desc is None or can_desc is None:
        raise RuntimeError("Not enough SIFT features for canonical calibration.")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(can_desc, raw_desc, k=2)

    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_test * n.distance:
            good.append(m)

    if len(good) < 12:
        raise RuntimeError(f"Not enough calibration matches: {len(good)} < 12")

    canonical_points = np.float32([can_kp[m.queryIdx].pt for m in good]) / can_scale
    raw_points = np.float32([raw_kp[m.trainIdx].pt for m in good]) / raw_scale

    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        raw_points,
        canonical_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=6.0,
        maxIters=8000,
        confidence=0.999,
        refineIters=100,
    )
    if matrix is None or inlier_mask is None:
        raise RuntimeError("Could not estimate RAW-reference -> canonical similarity transform.")

    inliers = int(inlier_mask.ravel().sum())
    ratio = float(inliers / max(1, len(good)))
    if inliers < 10 or ratio < 0.18:
        raise RuntimeError(
            f"Calibration transform is too weak: inliers={inliers}, ratio={ratio:.3f}"
        )

    return matrix.astype(np.float32), len(good), inliers, ratio


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate one fixed mapping from the RAW geometry reference into the "
            "existing canonical preview reference. This fixed scale is not re-estimated per frame."
        )
    )
    parser.add_argument("--raw-reference", required=True)
    parser.add_argument("--canonical-reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw_path = Path(args.raw_reference).resolve()
    canonical_path = Path(args.canonical_reference).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    raw = read_image(raw_path)
    canonical = read_image(canonical_path)

    matrix, matches, inliers, ratio = estimate_fixed_similarity(raw, canonical)
    width, height = canonical.shape[1], canonical.shape[0]
    warped = cv2.warpAffine(
        raw,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    overlay = cv2.addWeighted(canonical, 0.5, warped, 0.5, 0.0)
    write_image(out / "raw_reference_in_canonical.png", warped)
    write_image(out / "calibration_overlay.png", overlay)

    calibration = CanonicalCalibration(
        raw_reference_size=(raw.shape[1], raw.shape[0]),
        canonical_size=(canonical.shape[1], canonical.shape[0]),
        raw_reference_to_canonical=matrix,
        canonical_to_raw_reference=invert_affine(matrix),
        raw_reference=str(raw_path),
        canonical_reference=str(canonical_path),
        feature_matches=matches,
        feature_inliers=inliers,
        feature_inlier_ratio=ratio,
    )
    write_json(out / "canonical_calibration.json", calibration.to_dict())

    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    fixed_scale = float(np.hypot(a, b))
    rotation = float(np.degrees(np.arctan2(b, a)))

    print("=== Canonical calibration ===")
    print(f"RAW reference:       {raw_path}")
    print(f"Canonical reference: {canonical_path}")
    print(f"RAW size:            {raw.shape[1]}x{raw.shape[0]}")
    print(f"Canonical size:      {canonical.shape[1]}x{canonical.shape[0]}")
    print(f"Matches / inliers:   {matches} / {inliers} ({ratio:.1%})")
    print(f"Fixed scale:         {fixed_scale:.6f}")
    print(f"Fixed rotation:      {rotation:.3f} deg")
    print(f"Calibration JSON:    {out / 'canonical_calibration.json'}")
    print(f"Overlay:             {out / 'calibration_overlay.png'}")
    print("")
    print("Inspect calibration_overlay.png once. If it is correct, keep this calibration fixed.")


if __name__ == "__main__":
    main()
