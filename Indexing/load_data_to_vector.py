from pathlib import Path

from Indexing.Loader import load_single_document
from Indexing.chunker import chunk_document
from Indexing.embed import embed_chunks
from Indexing.vector_store import store_embeddings, create_collection


def index_text(pdf_path: str):
    """Index the text content of a single PDF into Qdrant."""
    documents = load_single_document(pdf_path)
    chunks = chunk_document(documents=documents)
    embedded_chunks = embed_chunks(chunks=chunks)
    store_embeddings(embedded_chunks=embedded_chunks)
    return chunks


if __name__ == "__main__":
    create_collection("Hardware_Diagnostic", 384)

    for pdf_path in Path("Data").rglob("*.pdf"):
        print(f"Indexing text in {pdf_path}...")
        index_text(str(pdf_path))

    print("store completed")
