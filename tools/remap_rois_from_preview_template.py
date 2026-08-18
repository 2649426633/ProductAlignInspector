from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.io_utils import read_image, write_image

# BGR colors used by tools/annotate_rois.py. We use a tolerance instead of
# requiring exact pixels because old previews may have been re-saved.
COLOR_S = np.array([60, 210, 60], dtype=np.int16)
COLOR_E = np.array([0, 170, 255], dtype=np.int16)
COLOR_P = np.array([255, 120, 40], dtype=np.int16)


@dataclass
class Orientation:
    name: str
    image: np.ndarray
    source_to_oriented: np.ndarray


def _orientation_candidates(image: np.ndarray) -> list[Orientation]:
    h, w = image.shape[:2]
    eye = np.eye(3, dtype=np.float64)
    hflip = np.array([[-1.0, 0.0, w - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    vflip = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, h - 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    rot180 = vflip @ hflip
    return [
        Orientation("identity", image.copy(), eye),
        Orientation("hflip", cv2.flip(image, 1), hflip),
        Orientation("vflip", cv2.flip(image, 0), vflip),
        Orientation("rot180", cv2.flip(image, -1), rot180),
    ]


def _color_mask(image: np.ndarray, color: np.ndarray, tolerance: int = 18) -> np.ndarray:
    c = color.astype(np.int16)
    lower = np.clip(c - tolerance, 0, 255).astype(np.uint8)
    upper = np.clip(c + tolerance, 0, 255).astype(np.uint8)
    return cv2.inRange(image, lower, upper)


def _annotation_mask(image: np.ndarray) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for color in (COLOR_S, COLOR_E, COLOR_P):
        mask = cv2.bitwise_or(mask, _color_mask(image, color, tolerance=22))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    return cv2.dilate(mask, kernel, iterations=1)


def _detect_preview_rectangles(
    image: np.ndarray,
    color: np.ndarray,
    expected_count: int,
    *,
    required: bool,
) -> list[list[int]]:
    if expected_count <= 0:
        return []

    mask = _color_mask(image, color, tolerance=22)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image.shape[:2]
    min_w = max(18, int(round(w * 0.012)))
    min_h = max(18, int(round(h * 0.012)))
    max_w = int(round(w * 0.40))
    max_h = int(round(h * 0.40))

    candidates: list[list[int]] = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if not (min_w <= cw <= max_w and min_h <= ch <= max_h):
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 2.0 * (cw + ch) * 0.38:
            continue
        candidates.append([int(x), int(y), int(cw), int(ch)])

    candidates.sort(key=lambda r: r[2] * r[3], reverse=True)
    kept: list[list[int]] = []
    for rect in candidates:
        x, y, cw, ch = rect
        cx, cy = x + cw / 2.0, y + ch / 2.0
        duplicate = False
        for other in kept:
            ox, oy, ow, oh = other
            ocx, ocy = ox + ow / 2.0, oy + oh / 2.0
            if abs(cx - ocx) <= 10 and abs(cy - ocy) <= 10 and abs(cw - ow) <= 18 and abs(ch - oh) <= 18:
                duplicate = True
                break
        if not duplicate:
            kept.append(rect)

    if len(kept) < expected_count and required:
        raise RuntimeError(
            f"Could only recover {len(kept)}/{expected_count} ROI rectangles for color {color.tolist()} from preview."
        )
    return kept[:expected_count]


def _center(rect: list[int]) -> tuple[float, float]:
    x, y, w, h = rect
    return x + w / 2.0, y + h / 2.0


def _orient_point(x: float, y: float, name: str, w: float, h: float) -> tuple[float, float]:
    if name == "identity":
        return x, y
    if name == "hflip":
        return w - x, y
    if name == "vflip":
        return x, h - y
    if name == "rot180":
        return w - x, h - y
    raise ValueError(name)


def _greedy_assignment(
    items: list[dict[str, object]],
    detected: list[list[int]],
    *,
    orientation: str,
    source_w: int,
    source_h: int,
    template_w: int,
    template_h: int,
) -> tuple[dict[str, list[int]], float]:
    if len(detected) < len(items):
        raise ValueError("not enough detected rectangles")

    remaining = detected[:]
    result: dict[str, list[int]] = {}
    total = 0.0
    sx = template_w / max(1.0, float(source_w))
    sy = template_h / max(1.0, float(source_h))

    # Use the most spatially distinctive items first: farthest from image center.
    def key(item: dict[str, object]) -> float:
        cx, cy = _center([int(v) for v in item["roi"]])
        ox, oy = _orient_point(cx, cy, orientation, source_w, source_h)
        tx, ty = ox * sx, oy * sy
        return (tx - template_w / 2.0) ** 2 + (ty - template_h / 2.0) ** 2

    for item in sorted(items, key=key, reverse=True):
        roi = [int(v) for v in item["roi"]]
        cx, cy = _center(roi)
        ox, oy = _orient_point(cx, cy, orientation, source_w, source_h)
        tx, ty = ox * sx, oy * sy
        best_i = min(
            range(len(remaining)),
            key=lambda i: (_center(remaining[i])[0] - tx) ** 2 + (_center(remaining[i])[1] - ty) ** 2,
        )
        rect = remaining.pop(best_i)
        rx, ry = _center(rect)
        total += (rx - tx) ** 2 + (ry - ty) ** 2
        result[str(item["id"])] = rect
    return result, total


def _associate_ids_with_best_config_orientation(
    s_items: list[dict[str, object]],
    e_items: list[dict[str, object]],
    detected_s: list[list[int]],
    detected_e: list[list[int]],
    *,
    source_w: int,
    source_h: int,
    template_w: int,
    template_h: int,
) -> tuple[dict[str, list[int]], str, float]:
    best_map: dict[str, list[int]] | None = None
    best_orientation = "identity"
    best_cost = float("inf")

    for orientation in ("identity", "hflip", "vflip", "rot180"):
        mapping: dict[str, list[int]] = {}
        cost = 0.0
        if s_items:
            smap, scost = _greedy_assignment(
                s_items, detected_s, orientation=orientation,
                source_w=source_w, source_h=source_h,
                template_w=template_w, template_h=template_h,
            )
            mapping.update(smap)
            cost += scost
        if e_items:
            emap, ecost = _greedy_assignment(
                e_items, detected_e, orientation=orientation,
                source_w=source_w, source_h=source_h,
                template_w=template_w, template_h=template_h,
            )
            mapping.update(emap)
            cost += ecost
        if cost < best_cost:
            best_cost = cost
            best_orientation = orientation
            best_map = mapping

    if best_map is None:
        raise RuntimeError("Could not associate preview rectangles with S/E IDs")
    return best_map, best_orientation, best_cost


def _match_template_to_reference(template: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, str, int, int, float]:
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("This OpenCV build does not provide SIFT")

    best: tuple[np.ndarray, str, int, int, float] | None = None
    for oriented in _orientation_candidates(template):
        ann = _annotation_mask(oriented.image)
        usable = cv2.bitwise_not(ann)
        tgray = cv2.cvtColor(oriented.image, cv2.COLOR_BGR2GRAY) if oriented.image.ndim == 3 else oriented.image
        rgray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference

        max_dim = 2200
        ts = min(1.0, max_dim / max(tgray.shape[:2]))
        rs = min(1.0, max_dim / max(rgray.shape[:2]))
        if ts < 0.9999:
            tsmall = cv2.resize(tgray, (round(tgray.shape[1] * ts), round(tgray.shape[0] * ts)), interpolation=cv2.INTER_AREA)
            msmall = cv2.resize(usable, (tsmall.shape[1], tsmall.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            tsmall, msmall = tgray, usable
        if rs < 0.9999:
            rsmall = cv2.resize(rgray, (round(rgray.shape[1] * rs), round(rgray.shape[0] * rs)), interpolation=cv2.INTER_AREA)
        else:
            rsmall = rgray

        sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.015, edgeThreshold=12)
        tkp, tdesc = sift.detectAndCompute(tsmall, msmall)
        rkp, rdesc = sift.detectAndCompute(rsmall, None)
        if tdesc is None or rdesc is None or len(tkp) < 8 or len(rkp) < 8:
            continue

        matcher = cv2.BFMatcher(cv2.NORM_L2)
        pairs = matcher.knnMatch(tdesc, rdesc, k=2)
        good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.74 * n.distance]
        if len(good) < 8:
            continue

        src = np.float32([tkp[m.queryIdx].pt for m in good]) / ts
        dst = np.float32([rkp[m.trainIdx].pt for m in good]) / rs
        H_oriented_to_ref, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0, maxIters=8000, confidence=0.999)
        if H_oriented_to_ref is None or inlier_mask is None:
            continue
        inliers = int(inlier_mask.ravel().sum())
        ratio = inliers / max(1, len(good))
        if inliers < 8:
            continue

        H_source_to_ref = H_oriented_to_ref @ oriented.source_to_oriented
        candidate = (H_source_to_ref, oriented.name, len(good), inliers, ratio)
        if best is None or (inliers, ratio) > (best[3], best[4]):
            best = candidate

    if best is None:
        raise RuntimeError("Could not match brunei_preview.png to the current reference image")
    return best


def _transform_rect(rect: list[int], H: np.ndarray, out_w: int, out_h: int) -> list[int]:
    x, y, w, h = [float(v) for v in rect]
    pts = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    x0 = max(0, int(math.floor(float(mapped[:, 0].min()))))
    y0 = max(0, int(math.floor(float(mapped[:, 1].min()))))
    x1 = min(out_w, int(math.ceil(float(mapped[:, 0].max()))))
    y1 = min(out_h, int(math.ceil(float(mapped[:, 1].max()))))
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def _draw_preview(reference: np.ndarray, config: dict[str, object]) -> np.ndarray:
    canvas = reference.copy()
    for item in config.get("screw_slots", []):
        x, y, w, h = [int(v) for v in item["roi"]]
        expected = str(item.get("expected", ""))
        color = (60, 210, 60) if expected == "screw" else (0, 170, 255)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 3)
        cv2.putText(canvas, str(item.get("id")), (x, max(24, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    for item in config.get("spring_regions", []):
        x, y, w, h = [int(v) for v in item["roi"]]
        color = (255, 120, 40)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 3)
        cv2.putText(canvas, str(item.get("id")), (x, max(24, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return canvas


def _draw_recovered_template(template: np.ndarray, mapping: dict[str, list[int]]) -> np.ndarray:
    canvas = template.copy()
    for roi_id, roi in mapping.items():
        x, y, w, h = roi
        color = (60, 210, 60) if roi_id.upper().startswith("S") else (0, 170, 255)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 4)
        cv2.putText(canvas, roi_id, (x, max(28, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Use a known-good annotated *_preview.png as the S/E ROI template, match its product image "
            "to the current canonical reference, and remap S/E ROIs. Spring overlays are optional."
        )
    )
    p.add_argument("--template-preview", required=True)
    p.add_argument("--source-config", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--output-config", required=True)
    p.add_argument("--output-preview", required=True)
    p.add_argument(
        "--strict-springs",
        action="store_true",
        help="Fail if spring overlays cannot be recovered. Default: remap S/E only and omit springs when absent.",
    )
    args = p.parse_args()

    template_path = Path(args.template_preview).resolve()
    source_config_path = Path(args.source_config).resolve()
    reference_path = Path(args.reference).resolve()
    output_config_path = Path(args.output_config).resolve()
    output_preview_path = Path(args.output_preview).resolve()

    template = read_image(template_path)
    reference = read_image(reference_path)
    config = json.loads(source_config_path.read_text(encoding="utf-8"))

    screw_items = [dict(x) for x in config.get("screw_slots", []) if bool(x.get("enabled", True))]
    s_items = [x for x in screw_items if str(x.get("id", "")).upper().startswith("S")]
    e_items = [x for x in screw_items if str(x.get("id", "")).upper().startswith("E")]
    spring_items = [dict(x) for x in config.get("spring_regions", []) if bool(x.get("enabled", True))]

    detected_s = _detect_preview_rectangles(template, COLOR_S, len(s_items), required=True)
    detected_e = _detect_preview_rectangles(template, COLOR_E, len(e_items), required=True)
    detected_p = _detect_preview_rectangles(
        template,
        COLOR_P,
        len(spring_items),
        required=bool(args.strict_springs),
    ) if spring_items else []

    th, tw = template.shape[:2]
    source_w = int(config.get("reference_width") or config.get("coordinate_system", {}).get("image_width") or tw)
    source_h = int(config.get("reference_height") or config.get("coordinate_system", {}).get("image_height") or th)

    template_rois, id_orientation, id_cost = _associate_ids_with_best_config_orientation(
        s_items,
        e_items,
        detected_s,
        detected_e,
        source_w=source_w,
        source_h=source_h,
        template_w=tw,
        template_h=th,
    )

    H, image_orientation, matches, inliers, ratio = _match_template_to_reference(template, reference)
    rh, rw = reference.shape[:2]

    remapped = json.loads(json.dumps(config))
    remapped["reference_image"] = str(reference_path)
    remapped["reference_width"] = rw
    remapped["reference_height"] = rh
    remapped["coordinate_system"] = {
        "reference_image": str(reference_path),
        "image_width": rw,
        "image_height": rh,
        "note": "S/E ROIs remapped from the known-good annotated preview template.",
        "template_preview": str(template_path),
        "config_id_orientation_selected": id_orientation,
        "template_image_orientation_selected": image_orientation,
        "template_feature_matches": matches,
        "template_feature_inliers": inliers,
        "template_feature_inlier_ratio": ratio,
    }

    for item in remapped.get("screw_slots", []):
        roi_id = str(item.get("id"))
        if roi_id in template_rois:
            item["roi"] = _transform_rect(template_rois[roi_id], H, rw, rh)

    if spring_items and len(detected_p) >= len(spring_items):
        # Only use spring overlays if they actually exist in this preview. Association is kept
        # conservative because current work is S/E presence inspection.
        spring_map, _ = _greedy_assignment(
            spring_items,
            detected_p,
            orientation=id_orientation,
            source_w=source_w,
            source_h=source_h,
            template_w=tw,
            template_h=th,
        )
        for item in remapped.get("spring_regions", []):
            roi_id = str(item.get("id"))
            if roi_id in spring_map:
                item["roi"] = _transform_rect(spring_map[roi_id], H, rw, rh)
    elif spring_items:
        # Do not silently keep known-wrong spring coordinates in a remapped config.
        remapped["spring_regions"] = []

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_preview_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(json.dumps(remapped, ensure_ascii=False, indent=2), encoding="utf-8")
    write_image(output_preview_path, _draw_preview(reference, remapped))

    recovered_path = output_preview_path.with_name(output_preview_path.stem + "_template_recovered.png")
    write_image(recovered_path, _draw_recovered_template(template, template_rois))

    print("=== ROI template remap complete ===")
    print(f"Template: {template_path}")
    print(f"Reference: {reference_path}")
    print(f"Recovered template S/E/P: {len(detected_s)}/{len(detected_e)}/{len(detected_p)}")
    if spring_items and len(detected_p) < len(spring_items):
        print(f"WARNING: spring overlays not present/recoverable ({len(detected_p)}/{len(spring_items)}); spring_regions omitted from output.")
    print(f"Config-ID orientation vs template: {id_orientation} (cost={id_cost:.1f})")
    print(f"Template-image orientation vs reference: {image_orientation}")
    print(f"SIFT matches/inliers: {matches}/{inliers} ({ratio:.1%})")
    print(f"Recovered-template debug: {recovered_path}")
    print(f"Output config: {output_config_path}")
    print(f"Output preview: {output_preview_path}")
    print("IMPORTANT: first inspect *_template_recovered.png; then inspect output preview. Do not rebuild a model until both are correct.")


if __name__ == "__main__":
    main()
