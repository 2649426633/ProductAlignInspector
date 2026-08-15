from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow direct execution: python tools\annotate_rois.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from product_align_inspector.io_utils import read_image, write_image


WINDOW_NAME = "ProductAlignInspector - ROI Annotation"


class RoiAnnotator:
    def __init__(
        self,
        image: np.ndarray,
        image_path: Path,
        output_path: Path,
        product_name: str,
        max_width: int,
        max_height: int,
    ) -> None:
        self.image = image
        self.image_path = image_path
        self.output_path = output_path
        self.product_name = product_name
        self.h, self.w = image.shape[:2]

        self.scale = min(max_width / self.w, max_height / self.h, 1.0)
        self.display_w = max(1, int(round(self.w * self.scale)))
        self.display_h = max(1, int(round(self.h * self.scale)))

        self.start: tuple[int, int] | None = None
        self.current: tuple[int, int] | None = None
        self.pending_roi: list[int] | None = None

        self.screw_slots: list[dict[str, object]] = []
        self.spring_regions: list[dict[str, object]] = []

        if output_path.exists():
            self._load_existing()

    def _load_existing(self) -> None:
        try:
            data = json.loads(self.output_path.read_text(encoding="utf-8"))
            self.screw_slots = list(data.get("screw_slots", []))
            self.spring_regions = list(data.get("spring_regions", []))
            print(
                f"Loaded existing config: {len(self.screw_slots)} screw slots, "
                f"{len(self.spring_regions)} spring regions"
            )
        except Exception as exc:
            print(f"Warning: could not load existing config: {exc}")

    def _display_to_original(self, x: int, y: int) -> tuple[int, int]:
        ox = int(round(x / self.scale))
        oy = int(round(y / self.scale))
        ox = min(max(0, ox), self.w - 1)
        oy = min(max(0, oy), self.h - 1)
        return ox, oy

    def _original_to_display_roi(self, roi: list[int]) -> tuple[int, int, int, int]:
        x, y, w, h = roi
        return (
            int(round(x * self.scale)),
            int(round(y * self.scale)),
            max(1, int(round(w * self.scale))),
            max(1, int(round(h * self.scale))),
        )

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.current = (x, y)
            self.pending_roi = None
        elif event == cv2.EVENT_MOUSEMOVE and self.start is not None:
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start is not None:
            self.current = (x, y)
            x0, y0 = self.start
            x1, y1 = self.current
            dx0, dx1 = sorted((x0, x1))
            dy0, dy1 = sorted((y0, y1))
            if dx1 - dx0 >= 4 and dy1 - dy0 >= 4:
                ox0, oy0 = self._display_to_original(dx0, dy0)
                ox1, oy1 = self._display_to_original(dx1, dy1)
                self.pending_roi = [ox0, oy0, max(1, ox1 - ox0), max(1, oy1 - oy0)]
                print(f"Pending ROI: {self.pending_roi}")
                print("Press S=screw, E=expected empty, P=spring region, C=cancel")
            self.start = None
            self.current = None

    def _next_id(self, prefix: str, items: list[dict[str, object]]) -> str:
        used = {str(item.get("id", "")) for item in items}
        index = 1
        while f"{prefix}{index:02d}" in used:
            index += 1
        return f"{prefix}{index:02d}"

    def add_screw_slot(self, expected: str) -> None:
        if self.pending_roi is None:
            print("Draw an ROI first.")
            return
        prefix = "S" if expected == "screw" else "E"
        slot_id = self._next_id(prefix, self.screw_slots)
        self.screw_slots.append({
            "id": slot_id,
            "roi": self.pending_roi.copy(),
            "expected": expected,
            "enabled": True,
        })
        print(f"Added {slot_id}: expected={expected}, roi={self.pending_roi}")
        self.pending_roi = None

    def add_spring_region(self) -> None:
        if self.pending_roi is None:
            print("Draw an ROI first.")
            return
        while True:
            raw = input("Expected spring count in this ROI: ").strip()
            try:
                count = int(raw)
                if count < 0:
                    raise ValueError
                break
            except ValueError:
                print("Please enter an integer >= 0.")
        region_id = self._next_id("SPRING", self.spring_regions)
        self.spring_regions.append({
            "id": region_id,
            "roi": self.pending_roi.copy(),
            "expected_count": count,
            "enabled": True,
        })
        print(f"Added {region_id}: expected_count={count}, roi={self.pending_roi}")
        self.pending_roi = None

    def undo(self) -> None:
        if self.pending_roi is not None:
            self.pending_roi = None
            print("Pending ROI cancelled.")
            return
        # Undo the most recently appended group in a simple, predictable way.
        if self.spring_regions:
            removed = self.spring_regions.pop()
            print(f"Removed {removed.get('id')}")
        elif self.screw_slots:
            removed = self.screw_slots.pop()
            print(f"Removed {removed.get('id')}")
        else:
            print("Nothing to undo.")

    def save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": 1,
            "product": self.product_name,
            "coordinate_system": {
                "reference_image": str(self.image_path),
                "image_width": self.w,
                "image_height": self.h,
                "note": "All ROI coordinates use the aligned reference-image coordinate system.",
            },
            "screw_slots": self.screw_slots,
            "spring_regions": self.spring_regions,
        }
        self.output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        preview_path = self.output_path.with_name(self.output_path.stem + "_preview.png")
        preview = self.render(original_resolution=True)
        write_image(preview_path, preview)

        print(f"Saved config:  {self.output_path}")
        print(f"Saved preview: {preview_path}")
        print(f"Screw/empty slots: {len(self.screw_slots)}")
        print(f"Spring regions:    {len(self.spring_regions)}")

    def render(self, original_resolution: bool = False) -> np.ndarray:
        if original_resolution:
            canvas = self.image.copy()
            scale = 1.0
        else:
            canvas = cv2.resize(self.image, (self.display_w, self.display_h), interpolation=cv2.INTER_AREA)
            scale = self.scale

        def convert(roi: list[int]) -> tuple[int, int, int, int]:
            if original_resolution:
                return tuple(map(int, roi))  # type: ignore[return-value]
            return self._original_to_display_roi(roi)

        font_scale = max(0.45, min(1.0, 0.65 * scale if original_resolution else 0.55))
        thickness = 2 if not original_resolution else max(2, int(round(self.w / 2200)))

        for slot in self.screw_slots:
            roi = list(map(int, slot["roi"]))
            x, y, w, h = convert(roi)
            expected = str(slot.get("expected", "screw"))
            # Green-ish for expected screw, orange-ish for expected empty.
            color = (60, 210, 60) if expected == "screw" else (0, 170, 255)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(
                canvas,
                f"{slot.get('id')}:{expected}",
                (x, max(20, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        for region in self.spring_regions:
            roi = list(map(int, region["roi"]))
            x, y, w, h = convert(roi)
            color = (255, 120, 40)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(
                canvas,
                f"{region.get('id')}:count={region.get('expected_count')}",
                (x, max(20, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        if not original_resolution:
            roi = self.pending_roi
            if roi is not None:
                x, y, w, h = convert(roi)
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(
                    canvas,
                    "PENDING: S / E / P / C",
                    (x, max(20, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif self.start is not None and self.current is not None:
                x0, y0 = self.start
                x1, y1 = self.current
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 0, 255), 2)

            help_text = "Drag ROI | S screw | E empty | P spring | C cancel | U undo | W save | Q quit"
            cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1] - 1, 980), 34), (20, 20, 20), -1)
            cv2.putText(canvas, help_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return canvas

    def run(self) -> None:
        print("\nROI annotation controls:")
        print("  Mouse drag : draw ROI")
        print("  S          : expected screw")
        print("  E          : expected empty hole/slot")
        print("  P          : spring region (asks expected count in terminal)")
        print("  C          : cancel current ROI")
        print("  U          : undo")
        print("  W          : save product JSON + preview")
        print("  Q / ESC    : quit")
        print(f"Reference resolution: {self.w} x {self.h}")
        print(f"Display scale: {self.scale:.4f}\n")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self.display_w, self.display_h)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        while True:
            cv2.imshow(WINDOW_NAME, self.render())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                self.add_screw_slot("screw")
            elif key == ord("e"):
                self.add_screw_slot("empty")
            elif key == ord("p"):
                self.add_spring_region()
            elif key == ord("c"):
                self.pending_roi = None
                print("Pending ROI cancelled.")
            elif key == ord("u"):
                self.undo()
            elif key == ord("w"):
                self.save()

        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively annotate screw/empty slots and spring regions.")
    parser.add_argument("--image", required=True, help="Aligned reference image")
    parser.add_argument("--output", required=True, help="Output product JSON")
    parser.add_argument("--product", default="product", help="Product name")
    parser.add_argument("--max-width", type=int, default=1700, help="Maximum display width")
    parser.add_argument("--max-height", type=int, default=950, help="Maximum display height")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    output_path = Path(args.output).resolve()
    image = read_image(image_path)

    annotator = RoiAnnotator(
        image=image,
        image_path=image_path,
        output_path=output_path,
        product_name=args.product,
        max_width=max(600, args.max_width),
        max_height=max(400, args.max_height),
    )
    annotator.run()


if __name__ == "__main__":
    main()
