from __future__ import annotations

import argparse
import csv
import html
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.io_utils import read_image, write_image

# Canonical Brunei S/E ROIs in the aligned reference coordinate system.
BRUNEI_ROIS: dict[str, tuple[int, int, int, int]] = {
    "S01": (483, 906, 186, 213),
    "S02": (1330, 969, 195, 220),
    "E01": (443, 238, 195, 152),
    "E02": (1521, 313, 131, 166),
    "E03": (2415, 386, 147, 156),
    "E04": (359, 1616, 165, 147),
    "E05": (1400, 1677, 189, 161),
    "E06": (2288, 1736, 204, 164),
    "E07": (2242, 1003, 198, 266),
    "E08": (2928, 1064, 242, 252),
    "E09": (293, 935, 143, 134),
}


def _affine3(matrix: np.ndarray) -> np.ndarray:
    out = np.eye(3, dtype=np.float64)
    out[:2, :] = np.asarray(matrix, dtype=np.float64)
    return out


def _reference_to_input_matrix(alignment) -> np.ndarray:
    """Return affine mapping from final aligned-reference coordinates to raw input.

    Feature alignment gives F: raw -> coarse-reference.
    ECC refinement uses WARP_INVERSE_MAP, so ECC matrix E maps final-reference -> coarse-reference.
    Therefore final-reference -> raw is inv(F) @ E.
    """
    if alignment.feature_matrix is None:
        raise RuntimeError(
            f"Cannot map ROI back to original image for alignment method {alignment.method}: "
            "feature_matrix is unavailable."
        )

    feature = _affine3(alignment.feature_matrix)
    feature_inv = np.linalg.inv(feature)
    if alignment.ecc_matrix is not None and "+ecc" in alignment.method:
        ecc = _affine3(alignment.ecc_matrix)
        combined = feature_inv @ ecc
    else:
        combined = feature_inv
    return combined[:2, :].astype(np.float32)


def _roi_polygon_in_input(roi: tuple[int, int, int, int], reference_to_input: np.ndarray) -> np.ndarray:
    x, y, w, h = roi
    corners = np.float32(
        [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ]
    ).reshape(-1, 1, 2)
    mapped = cv2.transform(corners, reference_to_input).reshape(-1, 2)
    return np.round(mapped).astype(np.int32)


def _draw_polygon(canvas: np.ndarray, polygon: np.ndarray, text: str, is_ng: bool) -> None:
    color = (0, 0, 255) if is_ng else (0, 190, 0)
    thickness = max(3, int(round(min(canvas.shape[:2]) / 850.0)))
    cv2.polylines(canvas, [polygon], True, color, thickness, cv2.LINE_AA)

    p = polygon[np.argmin(polygon[:, 1])]
    tx = int(np.clip(p[0], 0, max(0, canvas.shape[1] - 10)))
    ty = int(np.clip(p[1] - 8, 25, max(25, canvas.shape[0] - 5)))
    font_scale = max(0.55, min(1.0, min(canvas.shape[:2]) / 1900.0))
    text_thickness = max(1, thickness - 1)
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
    tx = min(tx, max(0, canvas.shape[1] - tw - 10))
    y0 = max(0, ty - th - baseline - 5)
    y1 = min(canvas.shape[0] - 1, ty + 4)
    cv2.rectangle(canvas, (tx, y0), (min(canvas.shape[1] - 1, tx + tw + 8), y1), color, -1)
    cv2.putText(
        canvas,
        text,
        (tx + 4, max(th + 1, ty - baseline)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def _read_rows(csv_path: Path, scenario: str | None) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status", "ok") != "ok":
                continue
            row_scenario = row.get("scenario", "").strip()
            if scenario and row_scenario.lower() != scenario.lower():
                continue
            source = row.get("source", "").strip()
            roi_id = row.get("roi_id", "").strip()
            if not source or roi_id not in BRUNEI_ROIS:
                continue
            grouped[source].append(row)
    return grouped


def _ratio_text(row: dict[str, str]) -> str:
    for key in ("score_over_decision_threshold", "score_over_threshold"):
        text = row.get(key, "").strip()
        if text:
            try:
                return f"{float(text):.2f}x"
            except ValueError:
                pass
    try:
        score = float(row.get("score", ""))
        threshold_text = row.get("decision_threshold", "") or row.get("threshold", "")
        threshold = float(threshold_text)
        if threshold > 0:
            return f"{score / threshold:.2f}x"
    except (TypeError, ValueError):
        pass
    return ""


def _make_html(output: Path, cards: list[dict[str, str]], scenario: str | None) -> None:
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Original Image Manual Review</title>",
        "<style>body{font-family:Arial,sans-serif;background:#eee;margin:20px}"
        ".card{background:white;margin:0 0 24px;padding:14px;border-radius:8px}"
        "img{max-width:100%;height:auto;border:1px solid #bbb}"
        ".ng{color:#c00;font-weight:bold}.pass{color:#087a08;font-weight:bold}"
        "code{background:#f4f4f4;padding:2px 4px}</style>",
        f"<h1>Original Image Manual Review - {html.escape(scenario or 'all')}</h1>",
        "<p><b>Red polygon = model predicted NG.</b> Green polygon = model predicted PASS. "
        "All polygons are mapped back onto the RAW ORIGINAL image. These are predictions, not human ground truth.</p>",
    ]
    for card in cards:
        css = "ng" if card["final"] == "NG" else "pass"
        parts.append("<div class='card'>")
        parts.append(f"<h2>{html.escape(card['name'])} - <span class='{css}'>{card['final']}</span></h2>")
        parts.append(f"<p>NG ROIs: <code>{html.escape(card['ng_rois'])}</code> | Alignment: {html.escape(card['alignment'])}</p>")
        parts.append(f"<img src='{html.escape(card['image_rel'])}'>")
        parts.append("</div>")
    (output / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Redraw existing evaluation CSV predictions back onto RAW original images. No DINO model required."
    )
    p.add_argument("--scores-csv", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--scenario", choices=["missing_screws", "excess_screws", "all_empty", "GOOD", "good"])
    p.add_argument("--output", required=True)
    p.add_argument("--foreground-threshold", type=int, default=238)
    p.add_argument("--only-ng", action="store_true", help="Draw only model-NG ROIs; default draws all 11 S/E ROIs.")
    args = p.parse_args()

    csv_path = Path(args.scores_csv).resolve()
    reference_path = Path(args.reference).resolve()
    output = Path(args.output).resolve()
    overlays = output / "overlays"
    output.mkdir(parents=True, exist_ok=True)
    overlays.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        raise SystemExit(f"Scores CSV not found: {csv_path}")
    if not reference_path.is_file():
        raise SystemExit(f"Reference image not found: {reference_path}")

    grouped = _read_rows(csv_path, args.scenario)
    if not grouped:
        raise SystemExit(f"No matching rows found in CSV for scenario={args.scenario!r}")

    reference = read_image(reference_path)
    cfg = ProductLocatorConfig(foreground_threshold=args.foreground_threshold)
    cards: list[dict[str, str]] = []
    failed = 0

    print("=== Redraw Existing Predictions On Original Images ===", flush=True)
    print(f"CSV:       {csv_path}", flush=True)
    print(f"Scenario:  {args.scenario or 'all'}", flush=True)
    print(f"Images:    {len(grouped)}", flush=True)
    print(f"Output:    {output}", flush=True)
    print("DINO/model/banks: NOT REQUIRED", flush=True)
    print("", flush=True)

    for index, (source_text, rows) in enumerate(sorted(grouped.items()), 1):
        source = Path(source_text)
        try:
            raw = read_image(source)
            alignment = align_to_reference(raw, reference, cfg)
            ref_to_input = _reference_to_input_matrix(alignment)
            canvas = raw.copy()
            ng_ids: list[str] = []

            by_roi = {row["roi_id"]: row for row in rows}
            for roi_id in BRUNEI_ROIS:
                row = by_roi.get(roi_id)
                if row is None:
                    continue
                is_ng = row.get("prediction", "").upper() == "DEFECT"
                if is_ng:
                    ng_ids.append(roi_id)
                if args.only_ng and not is_ng:
                    continue
                polygon = _roi_polygon_in_input(BRUNEI_ROIS[roi_id], ref_to_input)
                ratio = _ratio_text(row)
                status = "NG" if is_ng else "PASS"
                label = f"{roi_id} {status}" + (f" {ratio}" if ratio else "")
                _draw_polygon(canvas, polygon, label, is_ng)

            final = "NG" if ng_ids else "PASS"
            banner_h = max(70, int(round(raw.shape[0] * 0.055)))
            banner_color = (0, 0, 210) if final == "NG" else (0, 150, 0)
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], banner_h), banner_color, -1)
            title = f"MODEL {final} | {source.name} | NG: {','.join(ng_ids) if ng_ids else '-'}"
            cv2.putText(
                canvas,
                title,
                (20, int(banner_h * 0.68)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.8, banner_h / 70.0),
                (255, 255, 255),
                max(2, int(round(banner_h / 32.0))),
                cv2.LINE_AA,
            )

            scenario = rows[0].get("scenario", "unknown") or "unknown"
            dest_dir = overlays / scenario
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{source.stem}.png"
            write_image(dest, canvas)
            cards.append(
                {
                    "name": source.name,
                    "final": final,
                    "ng_rois": ", ".join(ng_ids) if ng_ids else "-",
                    "alignment": alignment.method,
                    "image_rel": dest.relative_to(output).as_posix(),
                }
            )
            print(
                f"[{index}/{len(grouped)}] {source.name:<28} {alignment.method:<24} "
                f"NG={','.join(ng_ids) if ng_ids else '-'}",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            print(f"[{index}/{len(grouped)}] {source.name} -> FAILED: {exc}", flush=True)

    _make_html(output, cards, args.scenario)
    print("", flush=True)
    print(f"Created: {len(cards)}", flush=True)
    print(f"Failed:  {failed}", flush=True)
    print(f"HTML:    {output / 'index.html'}", flush=True)
    print(f"Images:  {overlays}", flush=True)


if __name__ == "__main__":
    main()
