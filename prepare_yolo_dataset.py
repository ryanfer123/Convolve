#!/usr/bin/env python3
"""Prepare YOLO dataset structure from raw images.

Copies images from convolve/train into yolo_data/images/{train,val}.
Labels are expected to be created separately.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path


def main() -> None:
    src_dir = Path("/Users/ryanfernandes/convolve/train")
    dst_root = Path("/Users/ryanfernandes/convolve/yolo_data")
    train_dir = dst_root / "images" / "train"
    val_dir = dst_root / "images" / "val"

    images = sorted([p for p in src_dir.glob("*.png") if p.is_file()])
    if not images:
        raise SystemExit("No PNG images found in train folder.")

    random.seed(42)
    random.shuffle(images)

    split_idx = int(len(images) * 0.9)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for p in train_imgs:
        shutil.copy2(p, train_dir / p.name)
    for p in val_imgs:
        shutil.copy2(p, val_dir / p.name)

    print(f"Copied {len(train_imgs)} train images and {len(val_imgs)} val images.")


if __name__ == "__main__":
    main()
