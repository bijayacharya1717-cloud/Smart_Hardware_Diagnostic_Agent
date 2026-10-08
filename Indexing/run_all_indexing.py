# Indexing/run_all_indexing.py
from pathlib import Path

from Indexing.load_data_to_vector import index_text
from Indexing.load_diagrams_to_vector import index_diagrams
from Indexing.vector_store import create_collection

PDF_DIR = Path("Data")

if __name__ == "__main__":
    create_collection("Hardware_Diagnostic", 384)  # no-op if it already exists

    for pdf_path in PDF_DIR.rglob("*.pdf"):
        print(f"\n=== Indexing {pdf_path} ===")

        print("-> text pass")
        index_text(str(pdf_path))

        print("-> diagram pass")
        index_diagrams(str(pdf_path))

    print("\nAll indexing completed.")
