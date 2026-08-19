from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference, make_overlay
from product_align_inspector.io_utils import read_image, write_image
from product_align_inspector.roi import validate_roi

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def images(root: Path):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXT)


def scenario(rel: Path) -> str:
    parts = rel.parts
    if parts and parts[0].lower() == "good":
        return "good"
    if len(parts) >= 2 and parts[0].lower() == "ng":
        return parts[1]
    return parts[0] if parts else "unknown"


def check_config(reference, cfg):
    h, w = reference.shape[:2]
    cw, ch = cfg.get("reference_width"), cfg.get("reference_height")
    if cw is not None and ch is not None and (int(cw), int(ch)) != (w, h):
        raise SystemExit(
            f"CONFIG/REFERENCE SIZE MISMATCH: config={cw}x{ch}, reference={w}x{h}. "
            "Rigid workflow does not remap ROI coordinates."
        )
    for slot in cfg.get("screw_slots", []):
        if slot.get("enabled", True) and not validate_roi(slot.get("roi"), w, h):
            raise SystemExit(f"Invalid ROI {slot.get('id')}: {slot.get('roi')}")


def draw_rois(img, cfg):
    for slot in cfg.get("screw_slots", []):
        if not slot.get("enabled", True):
            continue
        roi = slot.get("roi")
        if roi is None:
            continue
        x, y, w, h = map(int, roi)
        expected = str(slot.get("expected", ""))
        color = (0, 190, 0) if expected == "screw" else (0, 150, 220)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            img,
            f"{slot.get('id')}:{expected}",
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Verify rigid alignment only. Reference must be a real RAW GOOD image "
            "with the same resolution as every input image."
        )
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument(
        "--config",
        help=(
            "Optional ROI JSON. Omit this while validating geometry. "
            "Only use a config created on the SAME raw reference coordinate system."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--foreground-threshold", type=int, default=238)
    parser.add_argument("--ecc-accept", type=float, default=0.70)
    args = parser.parse_args()

    root = Path(args.input_root).resolve()
    out = Path(args.output).resolve()
    reference = read_image(args.reference)

    cfg_json = None
    if args.config:
        cfg_json = json.loads(Path(args.config).read_text(encoding="utf-8"))
        check_config(reference, cfg_json)

    filters = {str(x).lower() for x in (args.scenario or [])}
    selected = []
    for path in images(root):
        rel = path.relative_to(root)
        sc = scenario(rel)
        if not filters or sc.lower() in filters:
            selected.append((path, rel, sc))

    align_cfg = ProductLocatorConfig(
        foreground_threshold=args.foreground_threshold,
        ecc_accept_score=args.ecc_accept,
    )

    rows = []
    print("=== Rigid alignment verification ===")
    print(f"Reference: {Path(args.reference).resolve()}")
    print(f"Reference size: {reference.shape[1]}x{reference.shape[0]}")
    print("Allowed geometry: rotation + X/Y translation only")
    print("Scale/shear/perspective: DISABLED")
    print(f"Images: {len(selected)} | ROI drawing: {'ON' if cfg_json else 'OFF'}")

    for i, (path, rel, sc) in enumerate(selected, 1):
        started = time.perf_counter()
        try:
            image = read_image(path)
            if image.shape[:2] != reference.shape[:2]:
                raise RuntimeError(
                    f"RAW size mismatch: input={image.shape[1]}x{image.shape[0]}, "
                    f"reference={reference.shape[1]}x{reference.shape[0]}"
                )

            result = align_to_reference(image, reference, align_cfg)
            elapsed = time.perf_counter() - started

            aligned_path = out / "aligned" / rel.with_suffix(".png")
            overlay_path = out / "overlays" / rel.with_suffix(".jpg")
            aligned_path.parent.mkdir(parents=True, exist_ok=True)
            overlay_path.parent.mkdir(parents=True, exist_ok=True)

            write_image(aligned_path, result.aligned)

            if cfg_json is None:
                overlay = make_overlay(reference, result.aligned)
            else:
                overlay = result.aligned.copy()
                draw_rois(overlay, cfg_json)
            write_image(overlay_path, overlay)

            tx, ty = result.rigid_translation_xy or ("", "")
            rows.append(
                {
                    "path": rel.as_posix(),
                    "scenario": sc,
                    "status": "ALIGN_OK",
                    "method": result.method,
                    "rotation_deg": result.rigid_rotation_deg,
                    "tx": tx,
                    "ty": ty,
                    "candidate_score": result.candidate_score,
                    "ecc": result.ecc_score,
                    "time_sec": elapsed,
                    "overlay": str(overlay_path),
                    "error": "",
                }
            )
            print(
                f"[{i}/{len(selected)}] {rel.as_posix()} -> ALIGN_OK | "
                f"rot={result.rigid_rotation_deg:.3f} deg | "
                f"tx={float(tx):.1f} ty={float(ty):.1f} | "
                f"ecc={result.ecc_score:.4f} | {elapsed:.3f}s"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "path": rel.as_posix(),
                    "scenario": sc,
                    "status": "RETRY",
                    "method": "",
                    "rotation_deg": "",
                    "tx": "",
                    "ty": "",
                    "candidate_score": "",
                    "ecc": "",
                    "time_sec": elapsed,
                    "overlay": "",
                    "error": str(exc),
                }
            )
            print(f"[{i}/{len(selected)}] {rel.as_posix()} -> RETRY ({elapsed:.3f}s): {exc}")

    out.mkdir(parents=True, exist_ok=True)
    fields = [
        "path",
        "scenario",
        "status",
        "method",
        "rotation_deg",
        "tx",
        "ty",
        "candidate_score",
        "ecc",
        "time_sec",
        "overlay",
        "error",
    ]
    with (out / "alignment_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(row["status"] == "ALIGN_OK" for row in rows)
    summary = {
        "images": len(rows),
        "align_ok": ok,
        "retry": len(rows) - ok,
        "reference": str(Path(args.reference).resolve()),
        "reference_size": [reference.shape[1], reference.shape[0]],
        "geometry": "rotation+translation only",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"ALIGN_OK / RETRY: {ok} / {len(rows) - ok}")


if __name__ == "__main__":
    main()
