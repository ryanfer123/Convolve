#!/usr/bin/env python3
"""Convert Label Studio JSON export to YOLO labels.

Expected Label Studio export format:
- JSON list of tasks
- Each task contains `data.image` and `annotations[].result[]`
- Rectangle labels in percent units

Outputs YOLO label files to the specified output directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


LABEL_MAP = {
    "signature": 0,
    "stamp": 1,
}


def rect_to_yolo(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    # Label Studio percent to absolute
    x_abs = x / 100.0 * img_w
    y_abs = y / 100.0 * img_h
    w_abs = w / 100.0 * img_w
    h_abs = h / 100.0 * img_h
    # YOLO normalized center format
    x_c = (x_abs + w_abs / 2) / img_w
    y_c = (y_abs + h_abs / 2) / img_h
    w_n = w_abs / img_w
    h_n = h_abs / img_h
    return x_c, y_c, w_n, h_n


def get_image_size(image_path: Path) -> Tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as img:
        return img.size  # (w, h)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert Label Studio JSON to YOLO labels")
    parser.add_argument("--ls_json", type=Path, required=True, help="Label Studio export JSON")
    parser.add_argument("--images_dir", type=Path, required=True, help="Directory with images")
    parser.add_argument("--labels_dir", type=Path, required=True, help="Output labels directory")
    args = parser.parse_args()

    args.labels_dir.mkdir(parents=True, exist_ok=True)

    tasks = json.loads(args.ls_json.read_text())
    for task in tasks:
        image_url = task.get("data", {}).get("image", "")
        image_name = Path(image_url).name
        image_path = args.images_dir / image_name
        if not image_path.exists():
            continue

        img_w, img_h = get_image_size(image_path)
        label_lines: List[str] = []

        for ann in task.get("annotations", []):
            for res in ann.get("result", []):
                if res.get("type") != "rectanglelabels":
                    continue
                value = res.get("value", {})
                labels = value.get("rectanglelabels", [])
                if not labels:
                    continue
                label = labels[0].lower()
                if label not in LABEL_MAP:
                    continue

                x = float(value.get("x", 0))
                y = float(value.get("y", 0))
                w = float(value.get("width", 0))
                h = float(value.get("height", 0))
                x_c, y_c, w_n, h_n = rect_to_yolo(x, y, w, h, img_w, img_h)
                label_lines.append(
                    f"{LABEL_MAP[label]} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"
                )

        if label_lines:
            label_path = args.labels_dir / f"{image_path.stem}.txt"
            label_path.write_text("\n".join(label_lines))


if __name__ == "__main__":
    main()
