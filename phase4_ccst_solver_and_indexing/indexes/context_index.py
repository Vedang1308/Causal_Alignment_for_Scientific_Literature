"""
Material Context Compatibility Index (I_ctx) for CCST Retrieval.
Implements top-k_c candidate retrieval based on material scope conditions
(species/system, environment, dosage scale, time horizon).
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict
from phase1_foundation.schemas.graph import DocumentGraph
from phase1_foundation.schemas.primitives import ContextDimension


def _tokenize_context(text: str) -> Set[str]:
    """Extracts normalized tokens from a context string."""
    cleaned = text.lower().replace("-", " ").replace("/", " ").replace("(", " ").replace(")", " ")
    return {w for w in cleaned.split() if len(w) > 2 and w not in {"the", "and", "for", "with", "from"}}


class ContextIndex:
    """
    Inverted multi-attribute index over experimental and material scope conditions.
    """
    def __init__(self):
        self.doc_contexts: Dict[str, Dict[str, str]] = {}
        # Inverted index: (dimension, token) -> list of doc_ids
        self.inverted_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    def add_graphs(self, graphs: List[DocumentGraph]) -> None:
        """Indexes context conditions from a list of DocumentGraphs."""
        for g in graphs:
            ctx = g.context_summary
            dim_map: Dict[str, str] = {}
            for dim in ContextDimension:
                val = ctx.get_dimension(dim)
                if val:
                    dim_map[dim.value] = val
                    for token in _tokenize_context(val):
                        self.inverted_index[(dim.value, token)].add(g.doc_id)

            self.doc_contexts[g.doc_id] = dim_map

    def query(self, query_graph: DocumentGraph, top_k: int = 15) -> List[Tuple[str, float]]:
        """
        Retrieves top-k_c candidate documents with matching or overlapping material context.
        """
        ctx = query_graph.context_summary
        scores: Dict[str, float] = defaultdict(float)

        for dim in ContextDimension:
            val = ctx.get_dimension(dim)
            if not val:
                continue
            q_tokens = _tokenize_context(val)
            for token in q_tokens:
                key = (dim.value, token)
                if key in self.inverted_index:
                    for doc_id in self.inverted_index[key]:
                        if doc_id != query_graph.doc_id:
                            # Weight dimension matches
                            scores[doc_id] += 1.0

        if not scores:
            return []

        # Normalize score
        max_score = max(scores.values()) if scores else 1.0
        results = [(doc_id, float(s / max_score)) for doc_id, s in scores.items()]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

