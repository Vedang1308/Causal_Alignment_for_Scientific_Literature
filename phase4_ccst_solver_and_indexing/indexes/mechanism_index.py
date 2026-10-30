"""
Unsigned Mechanism Structural Graph Index (I_graph) for CCST Retrieval.
Implements top-k_m structural candidate retrieval using Weisfeiler-Lehman (WL)
subtree hashing and inverted topological indexing.
Detects cross-domain analogies where vocabulary differs completely but causal
topology is isomorphic.
"""

from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
import hashlib
from phase1_foundation.schemas.graph import DocumentGraph


def _wl_structural_features(graph: DocumentGraph, iterations: int = 2) -> Counter:
    """
    Extracts structural Weisfeiler-Lehman subtree tokens and path motifs from a DocumentGraph.
    Omits textual words and edge polarities to capture purely unsigned causal topology.
    """
    if not graph.nodes:
        return Counter()

    node_id_to_idx = {n.id: i for i, n in enumerate(graph.nodes)}
    n = len(graph.nodes)

    # Initial labels: node functional role and in/out degrees
    in_edges = defaultdict(list)
    out_edges = defaultdict(list)

    for edge in graph.edges:
        if edge.source in node_id_to_idx and edge.target in node_id_to_idx:
            u = node_id_to_idx[edge.source]
            v = node_id_to_idx[edge.target]
            out_edges[u].append(v)
            in_edges[v].append(u)

    # Step 0 labels: role and degree
    current_labels = [
        f"R_{node.role.value}_din{len(in_edges[i])}_dout{len(out_edges[i])}"
        for i, node in enumerate(graph.nodes)
    ]

    tokens = Counter(current_labels)

    # Weisfeiler-Lehman relabeling iterations
    for it in range(1, iterations + 1):
        next_labels = []
        for i in range(n):
            in_neighbors = sorted([current_labels[v] for v in in_edges[i]])
            out_neighbors = sorted([current_labels[w] for w in out_edges[i]])
            
            sig = f"{current_labels[i]}|IN:{','.join(in_neighbors)}|OUT:{','.join(out_neighbors)}"
            # Hash to compact token
            h = hashlib.md5(sig.encode("utf-8")).hexdigest()[:8]
            token = f"WL{it}_{h}"
            next_labels.append(token)
            tokens[token] += 1
        current_labels = next_labels

    # Directed path motifs: length 2 (u -> v -> w)
    for u in range(n):
        for v in out_edges[u]:
            role_u = graph.nodes[u].role.value
            role_v = graph.nodes[v].role.value
            tokens[f"PATH1_{role_u}->{role_v}"] += 1
            for w in out_edges[v]:
                role_w = graph.nodes[w].role.value
                tokens[f"PATH2_{role_u}->{role_v}->{role_w}"] += 1

    return tokens


class MechanismIndex:
    """
    Sparse inverted structural index for unsigned mechanism graph topology.
    """
    def __init__(self, wl_iterations: int = 2):
        self.wl_iterations = wl_iterations
        self.doc_features: Dict[str, Counter] = {}
        # Inverted index: structural_token -> list of (doc_id, count)
        self.inverted_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.doc_norm: Dict[str, float] = {}

    def add_graphs(self, graphs: List[DocumentGraph]) -> None:
        """Indexes a collection of DocumentGraphs."""
        for g in graphs:
            feats = _wl_structural_features(g, iterations=self.wl_iterations)
            self.doc_features[g.doc_id] = feats
            norm_sq = sum(c * c for c in feats.values())
            self.doc_norm[g.doc_id] = (norm_sq ** 0.5) if norm_sq > 0 else 1.0

            for token, count in feats.items():
                self.inverted_index[token].append((g.doc_id, count))

    def query(self, query_graph: DocumentGraph, top_k: int = 15) -> List[Tuple[str, float]]:
        """
        Retrieves top-k_m structurally isomorphic candidates by cosine/Jaccard similarity
        over WL topological features.
        """
        if not self.doc_features:
            return []

        q_feats = _wl_structural_features(query_graph, iterations=self.wl_iterations)
        if not q_feats:
            return []

        q_norm_sq = sum(c * c for c in q_feats.values())
        q_norm = (q_norm_sq ** 0.5) if q_norm_sq > 0 else 1.0

        scores: Dict[str, float] = defaultdict(float)

        for token, q_count in q_feats.items():
            if token in self.inverted_index:
                for doc_id, d_count in self.inverted_index[token]:
                    if doc_id != query_graph.doc_id:
                        scores[doc_id] += float(q_count * d_count)

        if not scores:
            return []

        # Normalize to cosine similarity
        results = []
        for doc_id, dot in scores.items():
            sim = dot / (q_norm * self.doc_norm.get(doc_id, 1.0))
            results.append((doc_id, float(min(1.0, sim))))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

