from __future__ import annotations

from pathlib import Path
from typing import List

import pypdfium2 as pdfium


def render_pdf_to_images(pdf_path: Path, max_pages: int | None = None) -> List["object"]:
    """Render PDF pages to PIL images using pypdfium2."""
    images: List[object] = []
    doc = pdfium.PdfDocument(str(pdf_path))
    page_count = len(doc)
    if max_pages is not None:
        page_count = min(page_count, max_pages)

    for i in range(page_count):
        page = doc[i]
        bitmap = page.render(scale=2.0)
        images.append(bitmap.to_pil())

    return images
