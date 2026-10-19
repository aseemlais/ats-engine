from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.config import config

class QdrantVectorDB:
    def __init__(
        self,
        url: Optional[str] = None,
        storage_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        dim: Optional[int] = None
    ):
        self.url = url or config.qdrant_url
        self.storage_path = storage_path or config.qdrant_path
        self.collection_name = collection_name or config.qdrant_collection
        self.dim = dim or config.embedding_dimension
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            if self.url:
                self._client = QdrantClient(url=self.url)
            else:
                self._client = QdrantClient(path=self.storage_path)
            self._ensure_collection()
        return self._client

    def _ensure_collection(self):
        """Ensures the collection exists with Cosine distance and appropriate dimensions."""
        if not self._client.collection_exists(self.collection_name):
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
            )

    def upsert_points(self, points: List[PointStruct], batch_size: int = 256):
        """Upserts a list of PointStruct in batches."""
        for i in range(0, len(points), batch_size):
            chunk = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=chunk
            )

    def search_similar(self, query_vector: np.ndarray, limit: int = 500) -> List[Dict[str, Any]]:
        """Queries Qdrant for nearest neighbors by cosine similarity."""
        vec_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vec_list,
            limit=limit,
            with_payload=True
        )
        results = []
        for point in response.points:
            results.append({
                "candidate_id": point.payload.get("candidate_id"),
                "score": float(point.score),
                "payload": point.payload
            })
        return results

    def count_resumes(self) -> int:
        try:
            res = self.client.get_collection(self.collection_name)
            return res.points_count or 0
        except Exception:
            return 0
