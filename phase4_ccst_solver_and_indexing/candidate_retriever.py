"""
Candidate Retriever for CCST Scaling.
Implements Section IV-C:
Combines three sparse ANN indexes:
    C = N_sem(k_s) U N_graph(k_m) U N_ctx(k_c)
Reduces pairwise alignment complexity from O(N^2) to O(N log N + N * k_ccst).
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np

from phase1_foundation.schemas.graph import DocumentGraph
from .indexes.semantic_index import SemanticIndex
from .indexes.mechanism_index import MechanismIndex
from .indexes.context_index import ContextIndex


@dataclass
class RetrievedCandidate:
    """A candidate document retrieved for downstream CCST optimal transport alignment."""
    doc_id: str
    semantic_score: float = 0.0
    mechanism_score: float = 0.0
    context_score: float = 0.0
    retrieved_by_semantic: bool = False
    retrieved_by_mechanism: bool = False
    retrieved_by_context: bool = False


@dataclass
class CandidateRetrievalResult:
    """Summary of candidate retrieval for a query document."""
    query_doc_id: str
    candidates: List[RetrievedCandidate] = field(default_factory=list)
    total_candidates: int = 0
    k_semantic: int = 15
    k_mechanism: int = 15
    k_context: int = 10


def _build_pseudo_embedding(text: str, dim: int = 384) -> np.ndarray:
    """
    Constructs a deterministic dense projection vector from text tokens,
    used when precomputed model weights are not loaded.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().replace("-", " ").split()
    for tok in tokens:
        h = hash(tok)
        idx1 = abs(h) % dim
        idx2 = abs(h // 384) % dim
        vec[idx1] += 1.0
        vec[idx2] += 0.5
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class CandidateRetriever:
    """
    Multi-index candidate retriever combining semantic, structural, and context indexes.
    """
    def __init__(
        self,
        dimension: int = 384,
        k_s: int = 15,
        k_m: int = 15,
        k_c: int = 10,
    ):
        self.k_s = k_s
        self.k_m = k_m
        self.k_c = k_c
        self.semantic_index = SemanticIndex(dimension=dimension)
        self.mechanism_index = MechanismIndex(wl_iterations=2)
        self.context_index = ContextIndex()
        self.indexed_graphs: Dict[str, DocumentGraph] = {}

    def index_corpus(
        self,
        graphs: List[DocumentGraph],
        embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Indexes a list of DocumentGraphs across all three sparse indexes.
        """
        doc_ids = [g.doc_id for g in graphs]
        self.indexed_graphs = {g.doc_id: g for g in graphs}

        # 1. Index dense semantic vectors
        if embeddings is not None:
            emb_matrix = np.array([embeddings[did] for did in doc_ids], dtype=np.float32)
        else:
            emb_matrix = np.array([
                _build_pseudo_embedding(f"{g.title} {g.abstract}") for g in graphs
            ], dtype=np.float32)

        self.semantic_index.add_documents(doc_ids, emb_matrix)

        # 2. Index unsigned mechanism graph topology
        self.mechanism_index.add_graphs(graphs)

        # 3. Index material context conditions
        self.context_index.add_graphs(graphs)

    def retrieve(
        self,
        query_graph: DocumentGraph,
        query_embedding: Optional[np.ndarray] = None,
    ) -> CandidateRetrievalResult:
        """
        Retrieves union of candidate documents: C = N_sem U N_graph U N_ctx.
        """
        if query_embedding is None:
            query_embedding = _build_pseudo_embedding(f"{query_graph.title} {query_graph.abstract}")

        # 1. Semantic candidates
        sem_hits = self.semantic_index.query(query_embedding, top_k=self.k_s)
        
        # 2. Structural mechanism candidates (WL unsigned)
        mech_hits = self.mechanism_index.query(query_graph, top_k=self.k_m)
        
        # 3. Context candidates
        ctx_hits = self.context_index.query(query_graph, top_k=self.k_c)

        # Merge hits into candidates dictionary
        merged: Dict[str, RetrievedCandidate] = {}

        for doc_id, score in sem_hits:
            if doc_id == query_graph.doc_id:
                continue
            if doc_id not in merged:
                merged[doc_id] = RetrievedCandidate(doc_id=doc_id)
            merged[doc_id].semantic_score = score
            merged[doc_id].retrieved_by_semantic = True

        for doc_id, score in mech_hits:
            if doc_id == query_graph.doc_id:
                continue
            if doc_id not in merged:
                merged[doc_id] = RetrievedCandidate(doc_id=doc_id)
            merged[doc_id].mechanism_score = score
            merged[doc_id].retrieved_by_mechanism = True

        for doc_id, score in ctx_hits:
            if doc_id == query_graph.doc_id:
                continue
            if doc_id not in merged:
                merged[doc_id] = RetrievedCandidate(doc_id=doc_id)
            merged[doc_id].context_score = score
            merged[doc_id].retrieved_by_context = True

        candidates_list = list(merged.values())

        return CandidateRetrievalResult(
            query_doc_id=query_graph.doc_id,
            candidates=candidates_list,
            total_candidates=len(candidates_list),
            k_semantic=self.k_s,
            k_mechanism=self.k_m,
            k_context=self.k_c,
        )
