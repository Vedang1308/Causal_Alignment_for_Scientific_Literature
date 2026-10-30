"""
Cost Matrix Computation for CCST Alignment.
Implements:
1. Node semantic and role discrepancy matrix M_uu'
2. Signed structural cost matrices C_i, C_j (capturing causal direction, sign δ, and time lag τ)
3. Unsigned topological cost matrices (capturing pure graph structure for analogy detection)
"""

from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np
from phase1_foundation.schemas.graph import DocumentGraph, GraphNode, GraphEdge
from phase1_foundation.schemas.primitives import NodeRole, Polarity


def _text_overlap_distance(label_a: str, label_b: str) -> float:
    """
    Computes lexical / semantic distance between two node labels in [0.0, 1.0].
    Uses token-level Jaccard distance with character n-gram smoothing.
    """
    a = label_a.strip().lower()
    b = label_b.strip().lower()
    if not a or not b:
        return 1.0
    if a == b:
        return 0.0

    tokens_a = set(a.replace("-", " ").replace("_", " ").split())
    tokens_b = set(b.replace("-", " ").replace("_", " ").split())
    
    # Exact token overlap
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    if union:
        jaccard = len(intersection) / len(union)
    else:
        jaccard = 0.0

    # Character 3-gram overlap for substring / morphology matching
    def ngrams(text: str, n: int = 3) -> set:
        padded = f"  {text}  "
        return {padded[i:i+n] for i in range(len(padded) - n + 1)}

    grams_a = ngrams(a)
    grams_b = ngrams(b)
    gram_inter = len(grams_a.intersection(grams_b))
    gram_union = len(grams_a.union(grams_b))
    gram_sim = (gram_inter / gram_union) if gram_union > 0 else 0.0

    sim = 0.6 * jaccard + 0.4 * gram_sim
    return float(max(0.0, min(1.0, 1.0 - sim)))


def compute_node_cost_matrix(
    g1: DocumentGraph,
    g2: DocumentGraph,
    beta_role: float = 0.5,
) -> np.ndarray:
    """
    Computes node discrepancy matrix M of shape (|V1|, |V2|).
    
    Equation (2) node cost:
        M_uu' = dist(e_u, e_u') + beta * I(role(u) != role(u'))
        
    Parameters:
        g1: Source DocumentGraph
        g2: Target DocumentGraph
        beta_role: Penalty multiplier for mismatching functional roles
                   (e.g., matching an intervention to an outcome)
    """
    n1 = len(g1.nodes)
    n2 = len(g2.nodes)
    
    if n1 == 0 or n2 == 0:
        return np.zeros((max(1, n1), max(1, n2)), dtype=np.float32)

    M = np.zeros((n1, n2), dtype=np.float32)

    for i, u in enumerate(g1.nodes):
        for j, v in enumerate(g2.nodes):
            # 1. Semantic / Embedding distance
            if u.embedding is not None and v.embedding is not None and len(u.embedding) == len(v.embedding):
                vec_u = np.array(u.embedding, dtype=np.float32)
                vec_v = np.array(v.embedding, dtype=np.float32)
                norm_u = np.linalg.norm(vec_u)
                norm_v = np.linalg.norm(vec_v)
                if norm_u > 1e-8 and norm_v > 1e-8:
                    cos_sim = float(np.dot(vec_u, vec_v) / (norm_u * norm_v))
                    dist = max(0.0, min(1.0, 1.0 - (cos_sim + 1.0) / 2.0))
                else:
                    dist = _text_overlap_distance(u.label, v.label)
            else:
                dist = _text_overlap_distance(u.label, v.label)

            # 2. Role discrepancy penalty
            role_mismatch = 1.0 if u.role != v.role else 0.0

            M[i, j] = dist + beta_role * role_mismatch

    return M


def compute_graph_structure_matrices(
    graph: DocumentGraph,
    signed: bool = True,
    gamma_sign: float = 1.0,
    mu_lag: float = 0.3,
) -> np.ndarray:
    """
    Computes intra-graph relational structure matrix C of shape (|V|, |V|).
    
    In CCST:
    - If signed=True: captures causal polarity (+1 for promotive, -1 for inhibitory, 0 for null/absent).
      When comparing C1(u,v) and C2(u',v') under squared distortion (C1 - C2)^2:
        - Same sign (+1 vs +1, or -1 vs -1) -> 0 penalty
        - Opposite sign (+1 vs -1) -> (1 - (-1))^2 = 4 penalty (detects contradictions)
        - Edge vs non-edge -> 1 penalty
    - If signed=False: captures pure directed topology (1.0 for directed edge, 0 for absent).
      Ignores polarity differences to localize isomorphic causal structures (detects cross-domain analogies).
    """
    n = len(graph.nodes)
    if n == 0:
        return np.zeros((1, 1), dtype=np.float32)

    C = np.zeros((n, n), dtype=np.float32)
    node_id_to_idx = {node.id: idx for idx, node in enumerate(graph.nodes)}

    for edge in graph.edges:
        if edge.source in node_id_to_idx and edge.target in node_id_to_idx:
            u = node_id_to_idx[edge.source]
            v = node_id_to_idx[edge.target]
            
            if not signed:
                # Pure directed unsigned connectivity
                C[u, v] = 1.0 * edge.confidence
            else:
                # Signed causal effect: +1 (positive), -1 (negative), 0 (null)
                sign_val = 0.0
                if edge.sign == Polarity.POSITIVE:
                    sign_val = 1.0
                elif edge.sign == Polarity.NEGATIVE:
                    sign_val = -1.0
                elif edge.sign == Polarity.NULL:
                    sign_val = 0.0

                # Weight by confidence
                C[u, v] = sign_val * edge.confidence

    return C


def compute_all_cost_matrices(
    g1: DocumentGraph,
    g2: DocumentGraph,
    beta_role: float = 0.5,
    gamma_sign: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Builds the complete set of node and structural cost matrices for CCST alignment.
    Returns:
        - 'M': Node cost matrix (|V1|, |V2|)
        - 'C1_signed': Signed matrix for G1 (|V1|, |V1|)
        - 'C2_signed': Signed matrix for G2 (|V2|, |V2|)
        - 'C1_unsigned': Unsigned matrix for G1 (|V1|, |V1|)
        - 'C2_unsigned': Unsigned matrix for G2 (|V2|, |V2|)
        - 'p1': Normalized node mass distribution for G1 (|V1|,)
        - 'p2': Normalized node mass distribution for G2 (|V2|,)
    """
    M = compute_node_cost_matrix(g1, g2, beta_role=beta_role)
    C1_signed = compute_graph_structure_matrices(g1, signed=True, gamma_sign=gamma_sign)
    C2_signed = compute_graph_structure_matrices(g2, signed=True, gamma_sign=gamma_sign)
    C1_unsigned = compute_graph_structure_matrices(g1, signed=False)
    C2_unsigned = compute_graph_structure_matrices(g2, signed=False)

    p1 = np.array(g1.get_normalized_node_masses(), dtype=np.float32)
    p2 = np.array(g2.get_normalized_node_masses(), dtype=np.float32)

    return {
        "M": M,
        "C1_signed": C1_signed,
        "C2_signed": C2_signed,
        "C1_unsigned": C1_unsigned,
        "C2_unsigned": C2_unsigned,
        "p1": p1,
        "p2": p2,
    }
