# Indexing/diagram_extractor.py
import pymupdf as fitz
from pathlib import Path

def extract_page_images(pdf_path: str, page_numbers: list[int], output_dir: str, dpi: int = 200) -> list[dict]:
    """Render specific pages of a PDF to PNG images for vision-model captioning."""
    doc = fitz.open(pdf_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pdf_name = Path(pdf_path).stem
    extracted = []

    for page_num in page_numbers:
        page = doc[page_num - 1]  # fitz pages are 0-indexed internally
        pix = page.get_pixmap(dpi=dpi)
        image_path = f"{output_dir}/{pdf_name}_page{page_num}.png"
        pix.save(image_path)
        extracted.append({
            "source": pdf_path,
            "page": page_num,
            "image_path": image_path,
        })

    return extracted