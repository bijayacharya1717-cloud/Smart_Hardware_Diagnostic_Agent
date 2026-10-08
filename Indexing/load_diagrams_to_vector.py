# Indexing/load_diagrams_to_vector.py
from pathlib import Path

from Indexing.diagram_detector import find_diagram_pages
from Indexing.diagram_extracter import extract_page_images
from Indexing.diagram_captioner import caption_diagram
from Indexing.embed import embed_chunks
from Indexing.vector_store import store_embeddings, create_collection
from google import genai
import os


def index_diagrams(pdf_path: str, output_dir: str = "Data/diagrams"):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), vertexai=False)

    pages = find_diagram_pages(pdf_path)
    images = extract_page_images(pdf_path, pages, output_dir)

    diagram_chunks = []
    for img in images:
        caption = caption_diagram(img["image_path"], client)
        if caption is None:
            continue  # Gemini declined/failed on this page; skip it rather than crash

        diagram_chunks.append({
            "content": caption,
            "source": img["source"],
            "page": img["page"],
            "type": "diagram_caption",          # marks this chunk as diagram-derived
            "image_path": img["image_path"],     # so the real image can be shown later
        })

    if not diagram_chunks:
        print(f"No diagram captions produced for {pdf_path}; nothing to embed/store for this file.")
        return diagram_chunks

    embedded_chunks = embed_chunks(diagram_chunks)   # same embedding model as text chunks
    store_embeddings(embedded_chunks)                # same Qdrant collection, stable IDs
    return diagram_chunks


if __name__ == "__main__":
    create_collection("Hardware_Diagnostic", 384)  # no-op if it already exists

    for pdf_path in Path("Data").rglob("*.pdf"):
        print(f"Indexing diagrams in {pdf_path}...")
        index_diagrams(str(pdf_path))

    print("diagram indexing completed")
