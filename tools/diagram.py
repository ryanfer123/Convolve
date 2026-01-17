#!/usr/bin/env python3
"""Generate a simple Mermaid architecture diagram file."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate architecture diagram (Mermaid)")
    parser.add_argument("--output", type=Path, required=True, help="Output .md file path")
    args = parser.parse_args()

    diagram = """
```mermaid
flowchart TD
  A[Input PDFs/PNGs] --> B[Ingestion]
  B --> C[OCR: EasyOCR/PaddleOCR]
  B --> D[YOLO Signature/Stamp]
  C --> E[Field Extraction]
  E --> F[Fuzzy Match Master Lists]
  D --> G[Visual Fields]
  F --> H[Post-Processing]
  G --> H
  H --> I[JSON Output]
  H --> J[Visual Overlays]
```
""".strip()

    args.output.write_text(diagram)


if __name__ == "__main__":
    main()
