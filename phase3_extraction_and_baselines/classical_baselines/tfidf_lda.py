"""
tfidf_lda.py - Classical TF-IDF and Latent Dirichlet Allocation (LDA) Baselines with K-Means.

Proposal References:
Table I: Classical DM/ML Retrieval & Grouping - TF-IDF/LDA [12], K-Means clustering.
"""

import sys
import math
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1_foundation.schemas import DocumentRecord, BenchmarkPair


class TfidfLdaBaseline:
    """
    Combines TF-IDF term weighting, LDA topic modeling, and K-Means document clustering.
    """

    def __init__(self, n_topics: int = 15, n_clusters: int = 8, random_state: int = 42):
        self.n_topics = n_topics
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.vectorizer = None
        self.lda_model = None
        self.kmeans_model = None
        self.tfidf_matrix = None
        self.topic_distributions = None
        self.cluster_labels = None
        self.doc_ids: List[str] = []

    def fit(self, documents: List[DocumentRecord]) -> "TfidfLdaBaseline":
        """Fits TF-IDF, LDA, and K-Means models over a list of documents."""
        self.doc_ids = [d.doc_id for d in documents]
        texts = [f"{d.title} {d.abstract}" for d in documents]

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import LatentDirichletAllocation
            from sklearn.cluster import KMeans

            # 1. TF-IDF
            min_df_val = 2 if len(documents) >= 20 else 1
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words="english",
                min_df=min_df_val,
                ngram_range=(1, 2),
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)

            # 2. LDA Topic Modeling
            n_components = min(self.n_topics, max(2, len(documents) // 10 if len(documents) >= 20 else min(len(documents), self.n_topics)))
            self.lda_model = LatentDirichletAllocation(
                n_components=n_components,
                random_state=self.random_state,
                learning_method="online",
                max_iter=10,
            )
            self.topic_distributions = self.lda_model.fit_transform(self.tfidf_matrix)

            # 3. K-Means Clustering on Topic Distributions
            k = min(self.n_clusters, max(1, len(documents) // 20 if len(documents) >= 40 else min(len(documents), self.n_clusters)))
            self.kmeans_model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            self.cluster_labels = self.kmeans_model.fit_predict(self.topic_distributions)

        except ImportError:
            # Native fallback for environments where scikit-learn is not installed
            print("Notice: scikit-learn not detected in current environment. Using native TF-IDF fallback.")
            self._fit_native_fallback(texts)

        return self

    def _fit_native_fallback(self, texts: List[str]):
        """Native pure-Python TF-IDF implementation."""
        from collections import Counter
        doc_tokens = [[w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", t)] for t in texts]
        self.vocab = {}
        for tokens in doc_tokens:
            for t in set(tokens):
                self.vocab[t] = self.vocab.get(t, 0) + 1

        # Retain terms with min_df >= 2, or fallback to all terms if vocabulary is small
        filtered_vocab = {t: c for t, c in self.vocab.items() if c >= 2}
        if len(filtered_vocab) < self.n_topics:
            filtered_vocab = self.vocab

        self.term_to_idx = {t: i for i, t in enumerate(filtered_vocab.keys())}
        N = len(texts)
        self.native_idf = {t: math.log((N + 1.0) / (filtered_vocab[t] + 1.0)) + 1.0 for t in filtered_vocab}

        V = max(self.n_topics, len(self.term_to_idx))
        matrix = np.zeros((N, max(1, V)), dtype=np.float32)
        for i, tokens in enumerate(doc_tokens):
            counts = Counter(tokens)
            for t, cnt in counts.items():
                if t in self.term_to_idx:
                    idx = self.term_to_idx[t]
                    matrix[i, idx] = cnt * self.native_idf[t]
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] /= norm

        self.tfidf_matrix = matrix
        self.topic_distributions = matrix[:, :self.n_topics]
        self.cluster_labels = np.zeros(N, dtype=int)

    def query_by_tfidf(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Cosine similarity retrieval on TF-IDF space."""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            if hasattr(self, 'vectorizer') and self.vectorizer is not None:
                q_vec = self.vectorizer.transform([query_text])
                sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
                top_indices = np.argsort(-sims)[:top_k]
                return [(self.doc_ids[i], float(sims[i])) for i in top_indices if sims[i] > 0]
        except ImportError:
            pass

        # Native fallback cosine similarity
        q_tokens = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", query_text)]
        V = getattr(self, 'tfidf_matrix', np.zeros((len(self.doc_ids), 1))).shape[1]
        q_vec = np.zeros(V, dtype=np.float32)
        term_to_idx = getattr(self, 'term_to_idx', {})
        native_idf = getattr(self, 'native_idf', {})

        for t in q_tokens:
            if t in term_to_idx:
                idx = term_to_idx[t]
                q_vec[idx] += native_idf.get(t, 1.0)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm

        sims = np.dot(self.tfidf_matrix, q_vec)
        top_indices = np.argsort(-sims)[:top_k]
        return [(self.doc_ids[i], float(sims[i])) for i in top_indices if sims[i] > 0]

    def query_by_lda_topic(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Jensen-Shannon or Cosine similarity on LDA topic distribution."""
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self.vectorizer.transform([query_text])
        q_topic = self.lda_model.transform(q_vec)
        sims = cosine_similarity(q_topic, self.topic_distributions)[0]

        top_indices = np.argsort(-sims)[:top_k]
        return [(self.doc_ids[i], float(sims[i])) for i in top_indices if sims[i] > 0]

    def evaluate_retrieval_metrics(
        self,
        benchmark_pairs: List[BenchmarkPair],
        mode: str = "tfidf",
        top_k_list: List[int] = [1, 5, 10, 20],
    ) -> Dict[str, float]:
        """Evaluates MRR, Recall@k, and nDCG using TF-IDF or LDA retrieval."""
        query_fn = self.query_by_tfidf if mode == "tfidf" else self.query_by_lda_topic
        rr_list = []
        recall_at_k = {k: [] for k in top_k_list}
        ndcg_at_k = {k: [] for k in top_k_list}

        for pair in benchmark_pairs:
            q_text = f"{pair.doc_a.title} {pair.doc_a.abstract}"
            target_id = pair.doc_b.doc_id

            ranked = query_fn(q_text, top_k=max(top_k_list))
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
            "mode": mode,
            "MRR": round(sum(rr_list) / max(1, len(rr_list)), 4),
        }
        for k in top_k_list:
            metrics[f"Recall@{k}"] = round(sum(recall_at_k[k]) / max(1, len(recall_at_k[k])), 4)
            metrics[f"nDCG@{k}"] = round(sum(ndcg_at_k[k]) / max(1, len(ndcg_at_k[k])), 4)

        return metrics


if __name__ == "__main__":
    from phase2_data_and_benchmark.curate_benchmark import generate_stratum3_natural
    pairs = generate_stratum3_natural(count=15)
    docs = [p.doc_a for p in pairs] + [p.doc_b for p in pairs]
    baseline = TfidfLdaBaseline(n_topics=5, n_clusters=4).fit(docs)
    res_tfidf = baseline.evaluate_retrieval_metrics(pairs, mode="tfidf")
    print(f"TF-IDF Baseline Results: {res_tfidf}")
