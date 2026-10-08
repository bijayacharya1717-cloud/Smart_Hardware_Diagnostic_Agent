from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

model = SentenceTransformer(MODEL_NAME)

def embed_chunks(chunks: list[dict]) -> list[dict]:
    text = [chunk["content"] for chunk in chunks]

    embeddings = model.encode(text)

    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunk = dict(chunk)  # copy every existing field, whatever shape this chunk has
        embedded_chunk["embedding"] = embedding.tolist()
        embedded_chunks.append(embedded_chunk)

    return embedded_chunks
