from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

model = SentenceTransformer(MODEL_NAME)


def embed_query(query: str) -> list[float]:
    embedding = model.encode(query)
    return embedding.tolist()