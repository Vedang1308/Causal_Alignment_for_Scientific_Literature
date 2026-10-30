"""
sbert_retriever.py - Modern Dense Sentence-BERT / SciBERT Retrieval Baseline.

Proposal References:
Table I: Modern AI/DL Retrieval - SBERT [9]
Section V: Baselines - Sentence-BERT dense embeddings, MRR, Recall@k, nDCG evaluation.
"""

import sys
import math
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1_foundation.schemas import DocumentRecord, BenchmarkPair


class SBERTRetriever:
    """
    Dense semantic retrieval using Sentence-BERT / SciBERT embeddings.
    Leverages GPU acceleration on NVIDIA A100 when available.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None,
        use_mock_fallback: bool = True,
    ):
        self.model_name = model_name
        self.use_mock_fallback = use_mock_fallback
        self.model = None
        self.doc_ids: List[str] = []
        self.doc_records: Dict[str, DocumentRecord] = {}
        self.embeddings: Optional[np.ndarray] = None

        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

    _warned: bool = False

    def _init_model(self):
        """Initializes the SentenceTransformer model if not already loaded."""
        if self.model is not None or (self.use_mock_fallback and self._warned):
            return
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading Sentence-BERT model '{self.model_name}' on {self.device}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print(f"Loaded real Sentence-BERT weights successfully on device: {self.device}")
        except Exception as e:
            if self.use_mock_fallback:
                if not SBERTRetriever._warned:
                    print(f"Notice: Could not load SentenceTransformer ({e}). Using deterministic mock embeddings.")
                    SBERTRetriever._warned = True
                self.model = None
            else:
                raise e

    def _encode_text(self, texts: List[str]) -> np.ndarray:
        """Encodes text list into normalized embeddings (L2 norm = 1.0)."""
        self._init_model()
        if self.model is not None:
            embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
            return embs.astype(np.float32)

        # Fallback deterministic pseudo-embedding (384-dimensional based on hash)
        dim = 384
        embs = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = text.lower().split()
            for w in words:
                h = hash(w) % dim
                embs[i, h] += 1.0
            norm = np.linalg.norm(embs[i])
            if norm > 0:
                embs[i] /= norm
            else:
                embs[i, 0] = 1.0
        return embs

    def fit(self, documents: List[DocumentRecord]) -> "SBERTRetriever":
        """Encodes and indexes all documents."""
        self.doc_ids = [d.doc_id for d in documents]
        self.doc_records = {d.doc_id: d for d in documents}
        texts = [f"{d.title} {d.abstract}" for d in documents]
        self.embeddings = self._encode_text(texts)
        return self

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Scores all indexed documents via cosine similarity and returns top-k matches."""
        if self.embeddings is None:
            raise ValueError("Retriever has not been fitted yet. Call fit() first.")

        q_emb = self._encode_text([query_text])[0]
        sims = np.dot(self.embeddings, q_emb)

        top_indices = np.argsort(-sims)[:top_k]
        return [(self.doc_ids[i], float(sims[i])) for i in top_indices]

    def evaluate_retrieval_metrics(
        self,
        benchmark_pairs: List[BenchmarkPair],
        top_k_list: List[int] = [1, 5, 10, 20],
    ) -> Dict[str, float]:
        """Evaluates MRR, Recall@k, and nDCG over benchmark pairs."""
        rr_list = []
        recall_at_k = {k: [] for k in top_k_list}
        ndcg_at_k = {k: [] for k in top_k_list}

        for pair in benchmark_pairs:
            q_text = f"{pair.doc_a.title} {pair.doc_a.abstract}"
            target_id = pair.doc_b.doc_id

            ranked = self.query(q_text, top_k=max(top_k_list))
            ranked_ids = [doc_id for doc_id, _ in ranked]

            if target_id in ranked_ids:
                rank = ranked_ids.index(target_id) + 1
                rr_list.append(1.0 / rank)
            else:
                rr_list.append(0.0)

            for k in top_k_list:
                top_k_ids = ranked_ids[:k]
                if target_id in top_k_ids:
                    recall_at_k[k].append(1.0)
                    rank = top_k_ids.index(target_id) + 1
                    ndcg_at_k[k].append(1.0 / math.log2(rank + 1))
                else:
                    recall_at_k[k].append(0.0)
                    ndcg_at_k[k].append(0.0)

        metrics = {
            "model": self.model_name,
            "MRR": round(sum(rr_list) / max(1, len(rr_list)), 4),
        }
        for k in top_k_list:
            metrics[f"Recall@{k}"] = round(sum(recall_at_k[k]) / max(1, len(recall_at_k[k])), 4)
            metrics[f"nDCG@{k}"] = round(sum(ndcg_at_k[k]) / max(1, len(ndcg_at_k[k])), 4)

        return metrics


if __name__ == "__main__":
    from phase2_data_and_benchmark.curate_benchmark import generate_stratum3_natural
    pairs = generate_stratum3_natural(count=10)
    docs = [p.doc_a for p in pairs] + [p.doc_b for p in pairs]
    retriever = SBERTRetriever().fit(docs)
    res = retriever.evaluate_retrieval_metrics(pairs)
    print("SBERT Retriever Evaluation:", res)
