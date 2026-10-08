import os 
import hashlib

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

client = QdrantClient(
    url = os.getenv("QDRANT_URL"),
    api_key =os.getenv("QDRANT_API_KEY"),
)

def create_collection(collection_name: str, vector_size: int):

    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config =VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )


def store_embeddings(embedded_chunks: list[dict]):
    points= []

    for chunk in embedded_chunks:
        id_source = f"{chunk['source']}_{chunk['page']}_{chunk.get('chunk_index', 0)}"
        point_id = int(hashlib.md5(id_source.encode()).hexdigest()[:8], 16) # sha256 can also be used as a hash model to get better security 
        points.append(
            PointStruct(
                id=point_id,
                vector=chunk["embedding"],
                payload={k: v for k, v in chunk.items() if k != "embedding"},
            )
        )

    client.upsert(collection_name ="Hardware_Diagnostic", points =points)