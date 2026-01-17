#!/usr/bin/env python3
"""Run signature/stamp detection on images and output result.json."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

from utils.extraction import extract_fields
from utils.io_utils import load_master_list

_PADDLE_OCR = None
_EASY_OCR = None


def detect_on_image(image_path: Path, model_path: Path, conf: float) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    try:
        from ultralytics import YOLO
    except Exception:
        empty = {
            "signature": {"present": False, "bbox": None},
            "stamp": {"present": False, "bbox": None},
        }
        return empty, empty

    model = YOLO(str(model_path))
    results = model.predict(str(image_path), verbose=False, conf=conf)
    if not results:
        empty = {
            "signature": {"present": False, "bbox": None},
            "stamp": {"present": False, "bbox": None},
        }
        return empty, empty

    result = results[0]
    if result.boxes is None:
        empty = {
            "signature": {"present": False, "bbox": None},
            "stamp": {"present": False, "bbox": None},
        }
        return empty, empty

    names = result.names or {}
    boxes = result.boxes

    best = {
        "signature": {"present": False, "bbox": None, "conf": 0.0},
        "stamp": {"present": False, "bbox": None, "conf": 0.0},
    }

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

    simple = {
        "signature": {"present": best["signature"]["present"], "bbox": best["signature"]["bbox"]},
        "stamp": {"present": best["stamp"]["present"], "bbox": best["stamp"]["bbox"]},
    }
    return simple, best


def _get_paddle_ocr():
    global _PADDLE_OCR
    if _PADDLE_OCR is not None:
        return _PADDLE_OCR
    try:
        from paddleocr import PaddleOCR

        _PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        return _PADDLE_OCR
    except Exception:
        return None


def _get_easy_ocr():
    global _EASY_OCR
    if _EASY_OCR is not None:
        return _EASY_OCR
    try:
        import easyocr

        _EASY_OCR = easyocr.Reader(["en"], gpu=False)
        return _EASY_OCR
    except Exception:
        return None


def ocr_image_text(image_path: Path) -> str:
    try:
        from PIL import Image
    except Exception:
        return ""

    ocr = _get_paddle_ocr()
    easy = None
    if ocr is None:
        easy = _get_easy_ocr()
        if easy is None:
            return ""

    try:
        with Image.open(image_path) as img:
            w, h = img.size
            crops = [
                img,  # full
                img.crop((0, 0, w, int(h * 0.2))),  # header
                img.crop((0, int(h * 0.25), w, int(h * 0.75))),  # body/table
            ]
            texts = []
            for crop in crops:
                if ocr is not None:
                    result = ocr.ocr(crop, cls=True)
                    if not result:
                        continue
                    for line in result:
                        if line and isinstance(line, list):
                            for seg in line:
                                if len(seg) >= 2:
                                    texts.append(seg[1][0])
                else:
                    # Upscale for better detection
                    scale = 2
                    resized = crop.resize((crop.width * scale, crop.height * scale))
                    try:
                        import numpy as np

                        img_np = np.array(resized)
                    except Exception:
                        img_np = resized
                    result = easy.readtext(
                        img_np,
                        text_threshold=0.3,
                        low_text=0.2,
                        link_threshold=0.2,
                        contrast_ths=0.1,
                        adjust_contrast=0.5,
                    )
                    for seg in result:
                        if isinstance(seg, str):
                            texts.append(seg)
                        elif len(seg) >= 2:
                            texts.append(seg[1])
            return "\n".join([t for t in texts if t.strip()]).strip()
    except Exception:
        return ""


def draw_boxes(image_path: Path, best: Dict[str, Dict], output_path: Path, fields: Dict[str, object] | None = None) -> None:
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        colors = {"signature": "#00C853", "stamp": "#2962FF"}
        for key in ("signature", "stamp"):
            bbox = best.get(key, {}).get("bbox")
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=colors.get(key, "#FF6D00"), width=3)
            label = key
            if font:
                draw.text((x1 + 4, y1 + 4), label, fill=colors.get(key, "#FF6D00"), font=font)

        if fields:
            lines = [
                f"Dealer: {fields.get('dealer_name') or 'NA'}",
                f"Model: {fields.get('model_name') or 'NA'}",
                f"HP: {fields.get('horse_power') or 'NA'}",
                f"Cost: {fields.get('asset_cost') or 'NA'}",
            ]
            text = "\n".join(lines)
            x, y = 10, 10
            if font:
                try:
                    bbox = draw.multiline_textbbox((x, y), text, font=font)
                    draw.rectangle(bbox, fill="white")
                except Exception:
                    pass
                draw.multiline_text((x, y), text, fill="black", font=font)

        img.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sig/stamp detection on images")
    parser.add_argument("--input_dir", type=Path, required=True, help="Folder with image files")
    parser.add_argument("--output_dir", type=Path, required=True, help="Folder to write outputs")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.1, help="YOLO confidence threshold")
    parser.add_argument("--master_dealers", type=Path, default=None, help="Optional dealer master list (txt/csv)")
    parser.add_argument("--master_models", type=Path, default=None, help="Optional model master list (txt/csv)")
    parser.add_argument("--no_ocr", action="store_true", help="Disable OCR-based text extraction")
    parser.add_argument(
        "--visualize_dir",
        type=Path,
        default=None,
        help="Folder to write annotated images (defaults to <output_dir>/visuals)",
    )
    parser.add_argument(
        "--write_json",
        action="store_true",
        help="Write result.json (off by default)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize_dir is None:
        args.visualize_dir = args.output_dir / "visuals"
    args.visualize_dir.mkdir(parents=True, exist_ok=True)

    dealer_master = load_master_list(args.master_dealers)
    model_master = load_master_list(args.master_models)

    image_paths: List[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(sorted(args.input_dir.glob(ext)))

    results = []
    for image_path in image_paths:
        start = time.time()
        visual_fields, best = detect_on_image(image_path, args.model_path, args.conf)
        text = "" if args.no_ocr else ocr_image_text(image_path)
        fields, confidence = extract_fields(text, dealer_master, model_master)
        duration = round(time.time() - start, 4)

        output_image = args.visualize_dir / image_path.name
        fields["signature"] = visual_fields.get("signature", {"present": False, "bbox": None})
        fields["stamp"] = visual_fields.get("stamp", {"present": False, "bbox": None})
        draw_boxes(image_path, best, output_image, fields=fields)

        results.append(
            {
                "doc_id": image_path.stem,
                "fields": fields,
                "confidence": confidence,
                "processing_time_sec": duration,
                "cost_estimate_usd": 0.0,
            }
        )

    if args.write_json:
        output_path = args.output_dir / "result.json"
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"Wrote {len(results)} results to {output_path}")
    print(f"Wrote {len(image_paths)} annotated images to {args.visualize_dir}")


if __name__ == "__main__":
    main()
