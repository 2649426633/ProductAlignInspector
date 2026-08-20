from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from product_align_inspector.roi import enabled_slots

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_images(root: Path | None) -> list[Path]:
    if root is None:
        return []
    return sorted(
        p.resolve()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _normal_label(expected: str) -> str:
    expected = str(expected).lower()
    if expected == "screw":
        return "screw"
    if expected == "empty":
        return "empty"
    raise ValueError(f"Unsupported slot expected state: {expected}")


def _append_rows(
    rows: list[dict[str, str]],
    *,
    root: Path | None,
    source: str,
    slots: list[dict],
) -> None:
    for path in collect_images(root):
        row: dict[str, str] = {
            "image": str(path),
            "source": source,
            "split": "",
        }

        for slot in slots:
            slot_id = str(slot["id"])
            expected = str(slot.get("expected", "")).lower()

            if source == "good":
                row[slot_id] = _normal_label(expected)
            elif source == "all_empty":
                row[slot_id] = "empty"
            elif source == "missing_screws":
                row[slot_id] = "empty" if expected == "empty" else "?"
            elif source == "excess_screws":
                row[slot_id] = "screw" if expected == "screw" else "?"
            else:
                raise ValueError(source)

        rows.append(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a wide CSV manifest for the shared screw/empty CNN without moving or "
            "renaming any dataset images. Ambiguous missing/excess positions are written as '?'."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--good-root")
    parser.add_argument("--all-empty-root")
    parser.add_argument("--missing-root")
    parser.add_argument("--excess-root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    roots = {
        "good": None if not args.good_root else Path(args.good_root).resolve(),
        "all_empty": None if not args.all_empty_root else Path(args.all_empty_root).resolve(),
        "missing_screws": None if not args.missing_root else Path(args.missing_root).resolve(),
        "excess_screws": None if not args.excess_root else Path(args.excess_root).resolve(),
    }
    if not any(root is not None for root in roots.values()):
        raise SystemExit("Provide at least one dataset root.")

    config = json.loads(Path(args.config).resolve().read_text(encoding="utf-8"))
    slots = enabled_slots(config)
    slot_ids = [str(slot["id"]) for slot in slots]
    if not slot_ids:
        raise SystemExit("No enabled S/E slots found in config.")

    rows: list[dict[str, str]] = []
    for source, root in roots.items():
        _append_rows(rows, root=root, source=source, slots=slots)

    if not rows:
        raise SystemExit("No images found under the supplied dataset roots.")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image", "source", "split", *slot_ids]
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    unknown_cells = sum(
        1 for row in rows for slot_id in slot_ids if row.get(slot_id) == "?"
    )
    source_counts: dict[str, int] = {}
    for row in rows:
        source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1

    print("=== Semantic S/E manifest created ===")
    print(f"Output:            {output}")
    print(f"Images:            {len(rows)}")
    print(f"Slots:             {slot_ids}")
    print(f"Sources:           {source_counts}")
    print(f"Unknown cells:     {unknown_cells}")
    print()
    print("Label rules:")
    print("  good            -> S=screw, E=empty")
    print("  all_empty       -> every slot=empty")
    print("  missing_screws  -> E=empty; S01/S02='?' until manually labeled")
    print("  excess_screws   -> S=screw; E01..E09='?' until manually labeled")
    print()
    print("Fill '?' with screw or empty before final per-slot threshold calibration.")
    print("The tool never moves, renames, or modifies dataset images.")


if __name__ == "__main__":
    main()
