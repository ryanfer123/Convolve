from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from utils.extraction import extract_fields
from utils.io_utils import load_master_list
from utils.signature_stamp import detect_signature_and_stamp
from utils.text_extract import extract_text_from_pdf


def process_document(
    pdf_path: Path,
    dealer_master: List[str],
    model_master: List[str],
    max_pages: int | None = None,
    use_ocr: bool = False,
    sigstamp_model: Path | None = None,
) -> Tuple[Dict, float]:
    texts = extract_text_from_pdf(pdf_path, max_pages=max_pages, use_ocr=use_ocr)
    full_text = "\n".join(texts)
    fields, confidence = extract_fields(full_text, dealer_master, model_master)

    visual_fields = detect_signature_and_stamp(
        pdf_path,
        max_pages=max_pages,
        model_path=sigstamp_model,
    )
    fields["signature"] = visual_fields.get("signature", {"present": False, "bbox": None})
    fields["stamp"] = visual_fields.get("stamp", {"present": False, "bbox": None})

    return fields, confidence


__all__ = ["process_document", "load_master_list"]
