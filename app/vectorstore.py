from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue
)
import uuid
from app.config import settings

client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)

VECTOR_SIZE = 1024 

 # CREATE TO collection
def ensure_collection():
    """collection"""
    existing = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
 
  # SAVE TO documents
def upsert_documents(texts: list[str], embeddings: list[list[float]], metadata: list[dict] = None):
    """embeddings"""
    if metadata is None:
        metadata = [{} for _ in texts]

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={"text": txt, **meta},
        )
        for txt, emb, meta in zip(texts, embeddings, metadata)
    ]

    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
    return len(points)

  # SEARCH
def search_similar(query_vector: list[float], top_k: int = 5) -> list[dict]:
    """similar"""
    results = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {"text": hit.payload.get("text", ""), "score": hit.score, "metadata": hit.payload}
        for hit in results
    ]