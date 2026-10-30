"""
Unit Tests for Phase 4: CCST Solver & Sparse ANN Indexing.
Validates:
1. Node cost and signed/unsigned structure matrices
2. Context gate penalty (Phi_gate) and contradiction prevention
3. Partial FGW optimal transport solver convergence and marginal bounds
4. Signed vs unsigned distance gap (Delta_sign) for contradiction detection
5. Cross-domain analogy discovery via Weisfeiler-Lehman mechanism index
6. Multi-index candidate retriever union
"""

import pytest
import numpy as np

from phase1_foundation.schemas.graph import DocumentGraph, GraphNode, GraphEdge
from phase1_foundation.schemas.mechanism import MaterialContext
from phase1_foundation.schemas.primitives import (
    NodeRole,
    EdgeType,
    Polarity,
    ContextDimension,
    ContextCompatibility,
)
from phase4_ccst_solver_and_indexing.cost_matrices import (
    compute_node_cost_matrix,
    compute_graph_structure_matrices,
    compute_all_cost_matrices,
)
from phase4_ccst_solver_and_indexing.context_gate import ContextGate
from phase4_ccst_solver_and_indexing.ot_solver import CCSTOptimalTransportSolver
from phase4_ccst_solver_and_indexing.indexes.semantic_index import SemanticIndex
from phase4_ccst_solver_and_indexing.indexes.mechanism_index import MechanismIndex
from phase4_ccst_solver_and_indexing.indexes.context_index import ContextIndex
from phase4_ccst_solver_and_indexing.candidate_retriever import CandidateRetriever


def _create_sample_graphs():
    """Helper creating a baseline graph and variations for testing."""
    # Graph A: Baseline promotive claim
    ctx_a = MaterialContext()
    ctx_a.set_dimension(ContextDimension.SPECIES_OR_SYSTEM, "Homo sapiens")
    ctx_a.set_dimension(ContextDimension.ENVIRONMENT, "clinical")
    ctx_a.set_dimension(ContextDimension.DOSE_OR_SCALE, "80mg oral")
    ctx_a.set_dimension(ContextDimension.TIME_HORIZON, "24 weeks")

    nodes_a = [
        GraphNode(id="A_pop", role=NodeRole.POPULATION, label="adult patients", mass=1.0),
        GraphNode(id="A_int", role=NodeRole.INTERVENTION, label="atorvastatin", mass=1.0),
        GraphNode(id="A_med", role=NodeRole.MEDIATOR, label="hmg-coa reductase", mass=1.0),
        GraphNode(id="A_out", role=NodeRole.OUTCOME, label="ldl cholesterol", mass=1.0),
    ]
    edges_a = [
        GraphEdge(source="A_pop", target="A_int", sign=Polarity.POSITIVE),
        GraphEdge(source="A_int", target="A_med", sign=Polarity.POSITIVE),
        GraphEdge(source="A_med", target="A_out", sign=Polarity.NEGATIVE),  # lowers LDL
    ]
    g_a = DocumentGraph(
        doc_id="DOC_A",
        title="Atorvastatin efficacy in humans",
        abstract="In human adult patients, atorvastatin inhibits hmg-coa reductase lowering ldl cholesterol.",
        domain="biomedical",
        nodes=nodes_a,
        edges=edges_a,
        context_summary=ctx_a,
    )

    # Graph B: Opposite outcome under matching context (Contradiction Candidate)
    nodes_b = [
        GraphNode(id="B_pop", role=NodeRole.POPULATION, label="adult patients", mass=1.0),
        GraphNode(id="B_int", role=NodeRole.INTERVENTION, label="atorvastatin", mass=1.0),
        GraphNode(id="B_med", role=NodeRole.MEDIATOR, label="hmg-coa reductase", mass=1.0),
        GraphNode(id="B_out", role=NodeRole.OUTCOME, label="ldl cholesterol", mass=1.0),
    ]
    edges_b = [
        GraphEdge(source="B_pop", target="B_int", sign=Polarity.POSITIVE),
        GraphEdge(source="B_int", target="B_med", sign=Polarity.POSITIVE),
        GraphEdge(source="B_med", target="B_out", sign=Polarity.POSITIVE),  # Opposite sign: increases LDL
    ]
    g_b = DocumentGraph(
        doc_id="DOC_B",
        title="Atorvastatin paradoxical elevation",
        abstract="In human adult patients, atorvastatin increases ldl cholesterol unexpectedly.",
        domain="biomedical",
        nodes=nodes_b,
        edges=edges_b,
        context_summary=ctx_a,  # Same context!
    )

    # Graph C: Opposite outcome under DIFFERENT context (Contextual Heterogeneity)
    ctx_c = MaterialContext()
    ctx_c.set_dimension(ContextDimension.SPECIES_OR_SYSTEM, "Mus musculus (mice)")
    ctx_c.set_dimension(ContextDimension.ENVIRONMENT, "in vitro cell culture")
    ctx_c.set_dimension(ContextDimension.DOSE_OR_SCALE, "lethal microdose")
    ctx_c.set_dimension(ContextDimension.TIME_HORIZON, "4 hours")

    g_c = DocumentGraph(
        doc_id="DOC_C",
        title="Murine in vitro study of statins",
        abstract="In murine cell cultures, statins elevated cholesterol biomarkers over 4 hours.",
        domain="biomedical",
        nodes=nodes_b,
        edges=edges_b,
        context_summary=ctx_c,
    )

    # Graph D: Cross-domain analogy (Cybersecurity malware patch propagation)
    ctx_d = MaterialContext()
    ctx_d.set_dimension(ContextDimension.SPECIES_OR_SYSTEM, "enterprise network")
    ctx_d.set_dimension(ContextDimension.ENVIRONMENT, "cloud servers")
    ctx_d.set_dimension(ContextDimension.DOSE_OR_SCALE, "fleetwide update")
    ctx_d.set_dimension(ContextDimension.TIME_HORIZON, "24 hours")

    nodes_d = [
        GraphNode(id="D_pop", role=NodeRole.POPULATION, label="network endpoints", mass=1.0),
        GraphNode(id="D_int", role=NodeRole.INTERVENTION, label="security patch", mass=1.0),
        GraphNode(id="D_med", role=NodeRole.MEDIATOR, label="vulnerability closure", mass=1.0),
        GraphNode(id="D_out", role=NodeRole.OUTCOME, label="compromise rate", mass=1.0),
    ]
    edges_d = [
        GraphEdge(source="D_pop", target="D_int", sign=Polarity.POSITIVE),
        GraphEdge(source="D_int", target="D_med", sign=Polarity.POSITIVE),
        GraphEdge(source="D_med", target="D_out", sign=Polarity.NEGATIVE),  # lowers compromise rate
    ]
    g_d = DocumentGraph(
        doc_id="DOC_D",
        title="Automated patch distribution across enterprise servers",
        abstract="Distributing security patches closes vulnerabilities and reduces compromise rates across servers.",
        domain="computer_security",
        nodes=nodes_d,
        edges=edges_d,
        context_summary=ctx_d,
    )

    return g_a, g_b, g_c, g_d


def test_cost_matrices():
    g_a, g_b, _, _ = _create_sample_graphs()
    
    # 1. Node cost matrix
    M = compute_node_cost_matrix(g_a, g_b, beta_role=0.5)
    assert M.shape == (4, 4)
    # Diagonal should be minimal since labels and roles match exactly
    for i in range(4):
        assert M[i, i] < 0.1

    # 2. Signed structure matrices
    C_signed = compute_graph_structure_matrices(g_a, signed=True)
    assert C_signed.shape == (4, 4)
    # A_med -> A_out has negative sign
    assert C_signed[2, 3] == -1.0

    # 3. Unsigned structure matrices
    C_unsigned = compute_graph_structure_matrices(g_a, signed=False)
    assert C_unsigned.shape == (4, 4)
    assert C_unsigned[2, 3] == 1.0  # purely unsigned


def test_context_gate_evaluation():
    g_a, g_b, g_c, _ = _create_sample_graphs()
    gate = ContextGate()

    # Identical context (A vs B): gating penalty should be 0.0, allows contradiction
    eval_ab = gate.evaluate(g_a, g_b)
    assert eval_ab.gate_penalty == 0.0
    assert eval_ab.allows_contradiction is True
    assert eval_ab.is_context_incompatible is False

    # Conflicting context (A vs C): gating penalty should be high, forbids contradiction
    eval_ac = gate.evaluate(g_a, g_c)
    assert eval_ac.gate_penalty > 0.5
    assert eval_ac.allows_contradiction is False
    assert eval_ac.is_context_incompatible is True
    assert "Contextual Heterogeneity" in eval_ac.invalidation_reason


def test_ot_solver_convergence_and_bounds():
    g_a, g_b, _, _ = _create_sample_graphs()
    solver = CCSTOptimalTransportSolver(alpha=0.4, max_iter=30)

    res = solver.align(g_a, g_b, signed=True)
    assert res.transport_cost >= 0.0
    assert res.total_aligned_mass <= 1.0 + 1e-5
    assert res.total_aligned_mass > 0.1

    # Coupling marginal upper bounds: T1 <= p1, T^T 1 <= p2
    p1 = np.array(g_a.get_normalized_node_masses())
    p2 = np.array(g_b.get_normalized_node_masses())
    row_sums = res.coupling_matrix.sum(axis=1)
    col_sums = res.coupling_matrix.sum(axis=0)
    assert np.all(row_sums <= p1 + 1e-4)
    assert np.all(col_sums <= p2 + 1e-4)


def test_signed_vs_unsigned_gap_on_contradiction():
    g_a, g_b, _, _ = _create_sample_graphs()
    solver = CCSTOptimalTransportSolver()

    pair_align = solver.solve_pair(g_a, g_b)
    # Because edge signs conflict (med -> out is -1 vs +1),
    # signed distance should be strictly greater than unsigned distance!
    assert pair_align.signed_distance > pair_align.unsigned_distance
    assert pair_align.signed_vs_unsigned_gap > 0.0

    # Conflicting sign edge must be detected in sign_flip_edges
    assert len(pair_align.sign_flip_edges) >= 1
    flip = pair_align.sign_flip_edges[0]
    assert flip.is_sign_flip is True


def test_weisfeiler_lehman_cross_domain_analogy_detection():
    g_a, _, _, g_d = _create_sample_graphs()
    
    # G_A (biomedical) and G_D (cybersecurity) have 0 lexical overlap,
    # but their causal topologies (Pop -> Int -> Med -> Out) are 100% isomorphic!
    mech_idx = MechanismIndex(wl_iterations=2)
    mech_idx.add_graphs([g_a, g_d])

    hits = mech_idx.query(g_a, top_k=5)
    assert len(hits) >= 1
    top_doc_id, top_score = hits[0]
    assert top_doc_id == "DOC_D"
    assert top_score > 0.7  # Strong structural isomorphism detected!


def test_candidate_retriever_multi_index_union():
    g_a, g_b, g_c, g_d = _create_sample_graphs()
    retriever = CandidateRetriever(k_s=3, k_m=3, k_c=3)
    retriever.index_corpus([g_a, g_b, g_c, g_d])

    result = retriever.retrieve(g_a)
    assert result.total_candidates >= 2
    candidate_ids = {c.doc_id for c in result.candidates}
    
    # Semantic/context matching retrieves G_B
    assert "DOC_B" in candidate_ids
    # Structural mechanism matching retrieves G_D (analogy)
    assert "DOC_D" in candidate_ids
