# Document AI Extraction – Submission

## Overview
This repository contains a modular Document AI pipeline for extracting invoice fields. It supports signature/stamp detection and OCR-based field extraction, with clean separation of stages per the hackathon guidelines.

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
├── run_sigstamp_images.py
└── sample_output/
    └── result.json
```

## Architecture (Aligned to Guidelines)
1. **Document Ingestion**: load PDF/PNG inputs.
2. **Visual & Textual Understanding**: OCR for text; YOLO for signatures/stamps.
3. **Field Detection**: keyword + pattern extraction for dealer/model/HP/cost.
4. **Semantic Structuring**: normalize and fuzzy-match to master lists.
5. **Post-Processing**: confidence aggregation and schema validation.
6. **Output**: structured JSON (and optional visual overlays for PNGs).

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

### PNG Pipeline (Visual + JSON)
For PNG/JPG inputs, use the image pipeline. It writes annotated images by default and JSON with `--write_json`:
```
/path/to/python run_sigstamp_images.py \
  --input_dir /path/to/images \
  --output_dir ./sample_output \
  --model_path /path/to/best.pt \
  --write_json
```
Annotated images are in `sample_output/visuals`.

## Notes
- OCR is optional and triggered with `--ocr`. On macOS, the PNG pipeline uses EasyOCR.
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
