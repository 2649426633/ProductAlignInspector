from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.alignment import ProductLocatorConfig, align_to_reference
from product_align_inspector.anomaly.roi_patchcore import collect_anomaly_regions
from product_align_inspector.io_utils import read_image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
WINDOW = "ProductAlignInspector - Test ROI Labels"
TOP_BAR_H = 58


def collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_labels(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    if not path.exists():
        return result
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rel = str(row.get("relative_path", "")).strip().replace("\\", "/")
            raw = str(row.get("defect_rois", "")).strip()
            if rel:
                result[rel] = {v.strip() for v in raw.split("+") if v.strip()}
    return result


def save_labels(path: Path, image_keys: list[str], labels: dict[str, set[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "defect_rois"])
        writer.writeheader()
        for key in image_keys:
            if key in labels:
                writer.writerow({"relative_path": key, "defect_rois": "+".join(sorted(labels[key]))})


def group_color(roi_id: str, selected: bool) -> tuple[int, int, int]:
    if selected:
        return (0, 0, 255)
    rid = roi_id.upper()
    if rid.startswith("SPRING"):
        return (255, 140, 40)
    if rid.startswith("E"):
        return (0, 170, 255)
    if rid.startswith("S"):
        return (60, 210, 60)
    return (200, 200, 200)


def main() -> None:
    p = argparse.ArgumentParser(description="Annotate exact defect ROI IDs per test image without renaming files/folders.")
    p.add_argument("--test-root", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", default="dataset_roi_dino/test/labels.csv")
    p.add_argument("--max-width", type=int, default=1500)
    p.add_argument("--max-height", type=int, default=850)
    p.add_argument("--threshold", type=int, default=238)
    args = p.parse_args()

    test_root = Path(args.test_root).resolve()
    reference = read_image(Path(args.reference).resolve())
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    regions = collect_anomaly_regions(config)
    if not regions:
        raise SystemExit("No enabled ROI regions found in config.")

    # Only annotate NG images. GOOD truth is all-normal by definition.
    ng_root = test_root / "ng"
    images = collect_images(ng_root)
    if not images:
        raise SystemExit(f"No NG images found under: {ng_root}")

    keys = [str(img.relative_to(test_root).as_posix()) for img in images]
    output = Path(args.output).resolve()
    labels = load_labels(output)

    align_cfg = ProductLocatorConfig(foreground_threshold=args.threshold)
    index = 0
    current_aligned: np.ndarray | None = None
    current_base_display: np.ndarray | None = None
    current_scale = 1.0
    current_selected: set[str] = set()
    current_key = ""
    current_error = ""
    dirty = True

    def load_current() -> None:
        nonlocal current_aligned, current_base_display, current_scale
        nonlocal current_selected, current_key, current_error, dirty

        image_path = images[index]
        current_key = keys[index]
        current_selected = set(labels.get(current_key, set()))
        current_error = ""
        try:
            raw = read_image(image_path)
            result = align_to_reference(raw, reference, align_cfg)
            current_aligned = result.aligned
            h, w = current_aligned.shape[:2]
            current_scale = min(args.max_width / w, args.max_height / h, 1.0)
            dw = max(1, int(round(w * current_scale)))
            dh = max(1, int(round(h * current_scale)))

            # Critical performance fix: resize the 3393x2156 aligned image only once
            # when the image changes. The old implementation resized it every UI frame.
            current_base_display = cv2.resize(
                current_aligned,
                (dw, dh),
                interpolation=cv2.INTER_AREA,
            )
            if current_base_display.ndim == 2:
                current_base_display = cv2.cvtColor(current_base_display, cv2.COLOR_GRAY2BGR)

            print(
                f"[{index + 1}/{len(images)}] {current_key} | "
                f"align={result.method}, inliers={result.feature_inlier_ratio:.1%}, ECC={result.ecc_score} | "
                f"loaded labels={sorted(current_selected)}"
            )
        except Exception as exc:
            current_aligned = None
            current_base_display = None
            current_scale = 1.0
            current_error = str(exc)
            print(f"[{index + 1}/{len(images)}] {current_key} -> ALIGN FAILED: {exc}")
        dirty = True

    def save_current() -> None:
        labels[current_key] = set(current_selected)
        save_labels(output, keys, labels)
        print(f"Saved {current_key}: {sorted(current_selected)}")

    def render() -> np.ndarray:
        if current_base_display is None:
            canvas = np.zeros((650, 1200, 3), dtype=np.uint8)
            cv2.putText(
                canvas,
                f"ALIGN FAILED: {current_error}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "N next | B previous | Q quit",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return canvas

        # Cheap copy of the already-resized display image; no full-resolution resize here.
        canvas = current_base_display.copy()
        thickness = 2
        for region in regions:
            x, y, rw, rh = region.roi
            x = int(round(x * current_scale))
            y = int(round(y * current_scale))
            rw = max(1, int(round(rw * current_scale)))
            rh = max(1, int(round(rh * current_scale)))
            selected = region.id in current_selected
            color = group_color(region.id, selected)
            cv2.rectangle(canvas, (x, y), (x + rw, y + rh), color, thickness)
            label = f"{region.id}{'*' if selected else ''}"
            cv2.putText(
                canvas,
                label,
                (x, max(18, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

        canvas = cv2.copyMakeBorder(
            canvas,
            TOP_BAR_H,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(20, 20, 20),
        )
        title = f"{index + 1}/{len(images)}  {current_key}  defects={'+'.join(sorted(current_selected)) or '-'}"
        help_text = "Click ROI=toggle defect | ENTER/N=save+next | B=previous | C=clear | Q=save+quit"
        cv2.putText(canvas, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, help_text, (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        return canvas

    def mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal dirty
        if event != cv2.EVENT_LBUTTONDOWN or current_aligned is None:
            return

        y -= TOP_BAR_H
        if y < 0:
            return
        ox = x / current_scale
        oy = y / current_scale
        hits = []
        for region in regions:
            rx, ry, rw, rh = region.roi
            if rx <= ox <= rx + rw and ry <= oy <= ry + rh:
                hits.append(region)
        if not hits:
            return

        # Prefer the smallest containing ROI if regions overlap.
        region = min(hits, key=lambda r: r.roi[2] * r.roi[3])
        if region.id in current_selected:
            current_selected.remove(region.id)
        else:
            current_selected.add(region.id)
        print(f"Toggle {region.id} -> {'DEFECT' if region.id in current_selected else 'NORMAL'}")
        dirty = True

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, mouse)
    load_current()

    while True:
        # Only redraw when the image or selected labels changed.
        if dirty:
            cv2.imshow(WINDOW, render())
            dirty = False

        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue
        if key in (ord("q"), 27):
            save_current()
            break
        if key in (13, 10, ord("n")):
            save_current()
            if index < len(images) - 1:
                index += 1
                load_current()
            else:
                print("Reached final NG image.")
        elif key == ord("b"):
            save_current()
            if index > 0:
                index -= 1
                load_current()
        elif key == ord("c"):
            current_selected.clear()
            dirty = True
            print("Cleared current defect ROI labels.")

    cv2.destroyAllWindows()
    print(f"Labels CSV: {output}")


if __name__ == "__main__":
    main()
