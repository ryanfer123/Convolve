#!/usr/bin/env python3
"""Train YOLO model for signature/stamp detection."""
from __future__ import annotations

from pathlib import Path


def main() -> None:
    from ultralytics import YOLO

    data_yaml = Path("/Users/ryanfernandes/convolve/yolo_data/dataset.yaml")
    model = YOLO("yolov8n.pt")  # start from a small pretrained backbone
    model.train(data=str(data_yaml), epochs=50, imgsz=1024, batch=8, workers=4)


if __name__ == "__main__":
    main()
