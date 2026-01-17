from __future__ import annotations

from pathlib import Path
from typing import Dict

from utils.vision import render_pdf_to_images


def detect_signature_and_stamp(
    pdf_path: Path,
    max_pages: int | None = None,
    model_path: Path | None = None,
) -> Dict[str, Dict]:
    """Detect signature and stamp using a YOLO model if provided.

    The model is expected to have classes containing "signature" and "stamp".
    BBox coordinates are returned in image pixel space (x1, y1, x2, y2).
    """
    if model_path is None:
        return {
            "signature": {"present": False, "bbox": None},
            "stamp": {"present": False, "bbox": None},
        }

    try:
        from ultralytics import YOLO
    except Exception:
        return {
            "signature": {"present": False, "bbox": None},
            "stamp": {"present": False, "bbox": None},
        }

    model = YOLO(str(model_path))
    images = render_pdf_to_images(pdf_path, max_pages=max_pages)

    best = {
        "signature": {"present": False, "bbox": None, "conf": 0.0},
        "stamp": {"present": False, "bbox": None, "conf": 0.0},
    }

    for image in images:
        results = model.predict(image, verbose=False)
        if not results:
            continue
        result = results[0]
        if result.boxes is None:
            continue
        names = result.names or {}
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            label = str(names.get(cls_id, "")).lower()
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            if "signature" in label and conf > best["signature"]["conf"]:
                best["signature"] = {
                    "present": True,
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "conf": conf,
                }
            if "stamp" in label and conf > best["stamp"]["conf"]:
                best["stamp"] = {
                    "present": True,
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "conf": conf,
                }

    return {
        "signature": {"present": best["signature"]["present"], "bbox": best["signature"]["bbox"]},
        "stamp": {"present": best["stamp"]["present"], "bbox": best["stamp"]["bbox"]},
    }
