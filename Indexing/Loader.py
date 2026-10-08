import pymupdf as fitz
from pathlib import Path


def load_single_document(pdf_path: str) -> list[dict]:
    """Extract text from one PDF, page by page."""
    documents = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                documents.append({
                    "content": text,
                    "source": str(pdf_path),
                    "page": page_num,
                })
    return documents


