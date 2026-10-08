"""MCP document search tools for indexed datasheet text and diagram captions."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PayloadSchemaType
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

COLLECTION = os.getenv("QDRANT_COLLECTION", "Hardware_Diagnostic")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
embedder = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(RERANK_MODEL)


def _search(query: str, kind: str, limit: int = 5) -> dict:
    if not query.strip():
        return {"error": "query must not be empty", "evidence": []}
    if limit < 1 or limit > 10:
        return {"error": "limit must be between 1 and 10", "evidence": []}
    query_vector = embedder.encode(query).tolist()
    response = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=max(limit * 4, 12),
        query_filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="diagram_caption"))]) if kind == "schematic" else None,
    )
    points = response.points
    if kind == "datasheet":
        points = [p for p in points if p.payload and p.payload.get("type") != "diagram_caption"]
    if not points:
        return {"query": query, "evidence": []}
    scores = reranker.predict([(query, str(p.payload.get("content", ""))) for p in points])
    ranked = sorted(zip(points, scores), key=lambda item: float(item[1]), reverse=True)[:limit]
    return {
        "query": query,
        "evidence": [
            {
                "evidence_id": f"{p.payload.get('source', 'unknown')}#p{p.payload.get('page', '?')}#{p.id}",
                "content": p.payload.get("content", ""),
                "source": p.payload.get("source"),
                "page": p.payload.get("page"),
                "type": p.payload.get("type", "text"),
                "image_path": p.payload.get("image_path"),
                "rerank_score": round(float(score), 4),
            }
            for p, score in ranked
        ],
    }


datasheet_mcp = FastMCP("datasheet_rag")
schematic_mcp = FastMCP("schematic_rag")


@datasheet_mcp.tool
def search_datasheets(query: str, limit: int = 5) -> dict:
    """Search datasheet text for specifications, limits, operation and fault modes."""
    return _search(query, "datasheet", limit)


@schematic_mcp.tool
def search_schematics(query: str, limit: int = 5) -> dict:
    """Search indexed schematic and pinout captions for physical signal connectivity."""
    return _search(query, "schematic", limit)


if __name__ == "__main__":
    server = os.getenv("RAG_SERVER", "datasheet")
    if server == "schematic":
        # Existing collections may not have an index for this filter field.
        # Create it before serving schematic searches so Qdrant can filter captions.
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="type",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )
    (schematic_mcp if server == "schematic" else datasheet_mcp).run()
