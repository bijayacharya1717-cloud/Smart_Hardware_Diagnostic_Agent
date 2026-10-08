# Indexing/diagram_detector.py
import pymupdf as fitz

def find_diagram_pages(pdf_path: str, text_density_threshold: float = 0.15) -> list[int]:
    """Flag pages likely to contain diagrams: low text coverage relative to page area,
    OR pages containing known diagram-indicating keywords nearby."""
    doc = fitz.open(pdf_path)
    diagram_pages = []
    keywords = ["pin configuration", "pinout", "schematic", "block diagram",
                "typical application", "package outline", "mechanical drawing"]

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().lower()
        page_area = page.rect.width * page.rect.height
        text_blocks = page.get_text("blocks")
        text_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in text_blocks)
        density = text_area / page_area if page_area else 1.0

        has_keyword = any(kw in text for kw in keywords)
        is_sparse = density < text_density_threshold

        if has_keyword or is_sparse:
            diagram_pages.append(page_num)

    return diagram_pages

