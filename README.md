# Document AI Extraction – Submission

## Overview
This repository contains a baseline end-to-end pipeline for extracting invoice fields from PDF documents. It is intentionally lightweight and modular so you can extend it with OCR, layout detection, and signature/stamp detectors.

## Expected Output
One JSON object per document with:
- `dealer_name`
- `model_name`
- `horse_power`
- `asset_cost`
- `signature` (presence + bbox)
- `stamp` (presence + bbox)
- `confidence`, `processing_time_sec`, `cost_estimate_usd`

## Structure
```
submission.zip
│
├── executable.py
├── requirements.txt
├── README.md
├── utils/
│   ├── __init__.py
│   ├── io_utils.py
│   ├── text_extract.py
│   ├── extraction.py
│   └── pipeline.py
└── sample_output/
    └── result.json
```

## How to Run
```
python executable.py --input_dir /path/to/pdfs --output_dir ./sample_output
```
Enable OCR for scanned PDFs (requires Tesseract installed locally):
```
python executable.py --input_dir /path/to/pdfs --output_dir ./sample_output --ocr
```
Enable signature/stamp detection with a YOLO model:
```
python executable.py \
  --input_dir /path/to/pdfs \
  --output_dir ./sample_output \
  --sigstamp_model /path/to/sigstamp_yolo.pt
```
Optional master lists:
```
python executable.py \
  --input_dir /path/to/pdfs \
  --output_dir ./sample_output \
  --master_dealers /path/to/dealers.csv \
  --master_models /path/to/models.csv
```

## Notes
- OCR is optional and triggered with `--ocr`. If Tesseract is not installed, OCR silently falls back.
- Signature/stamp detection runs only if `--sigstamp_model` is provided. Class names must include "signature" and "stamp".
- Bounding boxes are returned in image pixel coordinates (x1, y1, x2, y2).

## Extending the Pipeline
- OCR: integrate `pytesseract` or `PaddleOCR` with rendered page images.
- Layout: use lightweight object detection for key-value regions.
- Signature/Stamp: train a small detector (YOLOv5/YOLOv8) and produce bbox coordinates.

## Labeling Signature/Stamp
Use the labeling workflow in [labeling/README.md](labeling/README.md) to create YOLO labels and train a detector.

## Cost & Latency
- This baseline estimates `$0.00` cost and logs processing time per document.
- Update cost estimation when adding OCR/ML models.
