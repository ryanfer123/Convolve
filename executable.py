#!/usr/bin/env python3
"""Main entry point for document extraction."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from utils.pipeline import process_document, load_master_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invoice field extraction")
    parser.add_argument("--input_dir", type=Path, required=True, help="Folder containing PDF files")
    parser.add_argument("--output_dir", type=Path, required=True, help="Folder to write result.json")
    parser.add_argument("--master_dealers", type=Path, default=None, help="Optional dealer master list (txt/csv)")
    parser.add_argument("--master_models", type=Path, default=None, help="Optional model master list (txt/csv)")
    parser.add_argument("--max_pages", type=int, default=None, help="Optional limit on pages per PDF")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR for scanned PDFs (requires Tesseract)")
    parser.add_argument(
        "--sigstamp_model",
        type=Path,
        default=None,
        help="Optional YOLO model path for signature/stamp detection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dealer_master = load_master_list(args.master_dealers)
    model_master = load_master_list(args.master_models)

    results = []
    for pdf_path in sorted(args.input_dir.glob("*.pdf")):
        start = time.time()
        fields, confidence = process_document(
            pdf_path,
            dealer_master=dealer_master,
            model_master=model_master,
            max_pages=args.max_pages,
            use_ocr=args.ocr,
            sigstamp_model=args.sigstamp_model,
        )
        duration = round(time.time() - start, 4)

        results.append(
            {
                "doc_id": pdf_path.stem,
                "fields": fields,
                "confidence": round(confidence, 4),
                "processing_time_sec": duration,
                "cost_estimate_usd": 0.0,
            }
        )

    output_path = args.output_dir / "result.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
