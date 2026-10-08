def chunk_document(
    documents: list[dict],
    chunk_size: int = 800,
    chunking_overlap: int = 100,
) -> list[dict]:
    if chunking_overlap >= chunk_size:
        raise ValueError("chunking_overlap must be smaller than chunk_size")

    chunks = []
    for doc in documents:
        content = doc["content"]
        start = 0
        chunk_index = 0

        while start < len(content):
            end = start + chunk_size
            chunks.append({
                "content": content[start:end],
                "source": doc["source"],
                "page": doc["page"],
                "chunk_index": chunk_index,
            })
            chunk_index += 1

            if end >= len(content):
                break
            start = end - chunking_overlap

    return chunks