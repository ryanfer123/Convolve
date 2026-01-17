from __future__ import annotations

from pathlib import Path
from typing import List

import pdfplumber

from utils.vision import render_pdf_to_images


def _ocr_image(image) -> str:
    try:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        result = ocr.ocr(image, cls=True)
        texts = []
        if not result:
            return ""
        for line in result:
            if line and isinstance(line, list):
                for seg in line:
                    if len(seg) >= 2:
                        texts.append(seg[1][0])
        return "\n".join([t for t in texts if t.strip()]).strip()
    except Exception:
        pass

    try:
        import easyocr

        reader = easyocr.Reader(["en"], gpu=False)
        result = reader.readtext(image)
        texts = []
        for seg in result:
            if len(seg) >= 2:
                texts.append(seg[1])
        return "\n".join([t for t in texts if t.strip()]).strip()
    except Exception:
        return ""


def extract_text_from_pdf(
    pdf_path: Path,
    max_pages: int | None = None,
    use_ocr: bool = False,
) -> List[str]:
    """Extract text per page using embedded PDF text and optional OCR.

    If OCR is enabled, pages with no embedded text will be rendered and OCR'd.
    """
    texts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages
        if max_pages is not None:
            pages = pages[:max_pages]
        for page in pages:
            texts.append(page.extract_text() or "")

    if not use_ocr:
        return texts

    images = render_pdf_to_images(pdf_path, max_pages=max_pages)
    for idx, image in enumerate(images):
        if idx < len(texts) and texts[idx].strip():
            continue
        ocr_text = _ocr_image(image)
        if idx < len(texts):
            texts[idx] = (texts[idx] + "\n" + ocr_text).strip()
        else:
            texts.append(ocr_text)

    return texts
