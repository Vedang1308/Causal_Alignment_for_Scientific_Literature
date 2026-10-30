"""
bm25_retriever.py - Classical BM25 Retrieval Baseline.

Proposal References:
Table I: Classical DM/ML Retrieval - BM25 [11]
Section V: Baselines - BM25, MRR, Recall@k, nDCG evaluation.
"""

import sys
import re
import math
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1_foundation.schemas import DocumentRecord, BenchmarkPair


def tokenize(text: str) -> List[str]:
    """Simple lowercase regex tokenizer."""
    return re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())


class BM25Retriever:
    """
    Standard BM25Okapi implementation for lexical document retrieval.
    Computes term-frequency / inverse-document-frequency with length normalization.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_ids: List[str] = []
        self.doc_records: Dict[str, DocumentRecord] = {}
        self.doc_lens: List[int] = []
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.term_freqs: List[Dict[str, int]] = []

    def fit(self, documents: List[DocumentRecord]) -> "BM25Retriever":
        """Indexes a list of DocumentRecord instances."""
        self.corpus_size = len(documents)
        self.doc_ids = [d.doc_id for d in documents]
        self.doc_records = {d.doc_id: d for d in documents}
        self.doc_lens = []
        self.term_freqs = []
        self.doc_freqs = {}

        total_tokens = 0
        for doc in documents:
            tokens = tokenize(f"{doc.title} {doc.abstract}")
            self.doc_lens.append(len(tokens))
            total_tokens += len(tokens)

            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)

            for t in tf.keys():
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.avg_doc_len = total_tokens / max(1, self.corpus_size)

        # Compute IDF
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            # Standard Lucene/Okapi smoothed IDF
            self.idf[term] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

        return self

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Scores all indexed documents and returns top-k (doc_id, score) pairs."""
        q_tokens = tokenize(query_text)
        scores = [0.0] * self.corpus_size

        for term in q_tokens:
            if term not in self.idf:
                continue
            idf_val = self.idf[term]

            for idx in range(self.corpus_size):
                tf = self.term_freqs[idx].get(term, 0)
                if tf > 0:
                    doc_len = self.doc_lens[idx]
                    num = tf * (self.k1 + 1.0)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    scores[idx] += idf_val * (num / denom)

        ranked_indices = sorted(range(self.corpus_size), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.doc_ids[i], scores[i]) for i in ranked_indices if scores[i] > 0]

    def evaluate_retrieval_metrics(
        self,
        benchmark_pairs: List[BenchmarkPair],
        top_k_list: List[int] = [1, 5, 10, 20],
    ) -> Dict[str, float]:
        """
        Evaluates retrieval metrics: MRR, Recall@k, and nDCG on benchmark pairs.
        For each pair (A, B), uses A as query and checks if B is retrieved.
        """
        rr_list = []
        recall_at_k = {k: [] for k in top_k_list}
        ndcg_at_k = {k: [] for k in top_k_list}

        for pair in benchmark_pairs:
            q_text = f"{pair.doc_a.title} {pair.doc_a.abstract}"
            target_id = pair.doc_b.doc_id

            ranked = self.query(q_text, top_k=max(top_k_list))
            ranked_ids = [doc_id for doc_id, _ in ranked]

            # Reciprocal Rank (RR)
            if target_id in ranked_ids:
                rank = ranked_ids.index(target_id) + 1
                rr_list.append(1.0 / rank)
            else:
                rr_list.append(0.0)

            # Recall@k and nDCG@k
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
            "MRR": round(sum(rr_list) / max(1, len(rr_list)), 4),
        }
        for k in top_k_list:
            metrics[f"Recall@{k}"] = round(sum(recall_at_k[k]) / max(1, len(recall_at_k[k])), 4)
            metrics[f"nDCG@{k}"] = round(sum(ndcg_at_k[k]) / max(1, len(ndcg_at_k[k])), 4)

        return metrics


if __name__ == "__main__":
    from phase2_data_and_benchmark.curate_benchmark import generate_stratum3_natural
    # Simple test
    pairs = generate_stratum3_natural(count=10)
    docs = [p.doc_a for p in pairs] + [p.doc_b for p in pairs]
    retriever = BM25Retriever().fit(docs)
    res = retriever.query(pairs[0].doc_a.title, top_k=3)
    print(f"Top 3 BM25 matches for '{pairs[0].doc_a.title[:40]}...':")
    for doc_id, score in res:
        print(f"  - {doc_id}: score = {score:.3f}")

