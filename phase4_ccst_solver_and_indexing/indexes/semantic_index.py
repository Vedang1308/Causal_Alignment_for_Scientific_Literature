"""
Dense Semantic ANN Index (I_sem) for CCST Retrieval.
Implements top-k_s semantic neighbor retrieval.
Supports Faiss IndexFlatIP / HNSW with a vectorized NumPy cosine fallback.
"""

from typing import List, Tuple, Optional
import numpy as np


class SemanticIndex:
    """
    Dense vector ANN index for semantic / topical similarity retrieval.
    """
    def __init__(self, dimension: int = 384, use_faiss: bool = True):
        self.dimension = dimension
        self.use_faiss = use_faiss
        self.doc_ids: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self._faiss_index = None

        if self.use_faiss:
            try:
                import faiss
                self._faiss_index = faiss.IndexFlatIP(dimension)
            except (ImportError, Exception):
                self._faiss_index = None

    def add_documents(self, doc_ids: List[str], embeddings: np.ndarray) -> None:
        """
        Adds normalized embeddings to the index.
        embeddings: shape (N, dimension)
        """
        if len(doc_ids) != len(embeddings):
            raise ValueError(f"Mismatch: {len(doc_ids)} IDs vs {len(embeddings)} embeddings")

        # L2-normalize vectors for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = (embeddings / norms).astype(np.float32)

        self.doc_ids.extend(doc_ids)

        if self._faiss_index is not None:
            self._faiss_index.add(normalized)
        else:
            if self.vectors is None:
                self.vectors = normalized
            else:
                self.vectors = np.vstack([self.vectors, normalized])

    def query(self, query_vector: np.ndarray, top_k: int = 15) -> List[Tuple[str, float]]:
        """
        Retrieves top-k_s nearest document IDs by cosine similarity.
        Returns: [(doc_id, score), ...]
        """
        if not self.doc_ids:
            return []

        top_k = min(top_k, len(self.doc_ids))
        q = np.array(query_vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        q = q.reshape(1, -1)

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q, top_k)
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if 0 <= idx < len(self.doc_ids):
                    results.append((self.doc_ids[idx], float(score)))
            return results
        else:
            # Vectorized NumPy cosine similarity
            sims = np.dot(self.vectors, q.T).flatten()
            if top_k >= len(sims):
                top_indices = np.argsort(-sims)
            else:
                part = np.argpartition(-sims, top_k)[:top_k]
                top_indices = part[np.argsort(-sims[part])]

            return [(self.doc_ids[i], float(sims[i])) for i in top_indices]

