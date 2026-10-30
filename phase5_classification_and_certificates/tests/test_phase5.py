"""
Unit Tests for Phase 5: 5-Way Classification, Abstention & Certificates.
Validates:
1. Feature extraction from CCST dual-view alignments (14 features)
2. Calibrated classification across Agreement, Analogy, Contradiction, and Heterogeneity
3. Context-gating protection against false contradictions
4. Selective classification and risk-threshold abstention
5. Disagreement certificate generation, minimal disagreement set, and formatting
"""

import pytest
import numpy as np

from phase1_foundation.schemas.graph import DocumentGraph, GraphNode, GraphEdge
from phase1_foundation.schemas.mechanism import MaterialContext
from phase1_foundation.schemas.primitives import (
    NodeRole,
    Polarity,
    ContextDimension,
    RelationType,
    ProvenanceSpan,
)
from phase4_ccst_solver_and_indexing.ot_solver import CCSTOptimalTransportSolver
from phase5_classification_and_certificates.feature_extractor import CCSTFeatureExtractor, FEATURE_NAMES
from phase5_classification_and_certificates.classifier import CCSTClassifier
from phase5_classification_and_certificates.abstention_gate import AbstentionGate
from phase5_classification_and_certificates.certificate_builder import CertificateBuilder
from phase5_classification_and_certificates.formatter import CertificateFormatter


def _create_sample_graphs():
    """Helper creating sample graphs across test conditions."""
    # Graph A: Baseline promotive claim in humans
    ctx_a = MaterialContext()
    ctx_a.set_dimension(ContextDimension.SPECIES_OR_SYSTEM, "Homo sapiens")
    ctx_a.set_dimension(ContextDimension.ENVIRONMENT, "clinical")
    ctx_a.set_dimension(ContextDimension.DOSE_OR_SCALE, "80mg oral")
    ctx_a.set_dimension(ContextDimension.TIME_HORIZON, "24 weeks")

    span_a = ProvenanceSpan(start_char=0, end_char=50, text="Atorvastatin reduces LDL cholesterol in clinical patients.")
    nodes_a = [
        GraphNode(id="A_pop", role=NodeRole.POPULATION, label="adult patients", mass=1.0, provenance=span_a),
        GraphNode(id="A_int", role=NodeRole.INTERVENTION, label="atorvastatin", mass=1.0, provenance=span_a),
        GraphNode(id="A_med", role=NodeRole.MEDIATOR, label="hmg-coa reductase", mass=1.0, provenance=span_a),
        GraphNode(id="A_out", role=NodeRole.OUTCOME, label="ldl cholesterol", mass=1.0, provenance=span_a),
    ]
    edges_a = [
        GraphEdge(source="A_pop", target="A_int", sign=Polarity.POSITIVE),
        GraphEdge(source="A_int", target="A_med", sign=Polarity.POSITIVE),
        GraphEdge(source="A_med", target="A_out", sign=Polarity.NEGATIVE),  # lowers LDL
    ]
    g_a = DocumentGraph(
        doc_id="DOC_A",
        title="Atorvastatin clinical efficacy in humans",
        abstract="In human adult patients, atorvastatin reduces ldl cholesterol via hmg-coa reductase.",
        domain="biomedical",
        nodes=nodes_a,
        edges=edges_a,
        context_summary=ctx_a,
    )

    # Graph B: Opposite claim under MATCHING context (True Contradiction)
    nodes_b = [
        GraphNode(id="B_pop", role=NodeRole.POPULATION, label="adult patients", mass=1.0, provenance=span_a),
        GraphNode(id="B_int", role=NodeRole.INTERVENTION, label="atorvastatin", mass=1.0, provenance=span_a),
        GraphNode(id="B_med", role=NodeRole.MEDIATOR, label="hmg-coa reductase", mass=1.0, provenance=span_a),
        GraphNode(id="B_out", role=NodeRole.OUTCOME, label="ldl cholesterol", mass=1.0, provenance=span_a),
    ]
    edges_b = [
        GraphEdge(source="B_pop", target="B_int", sign=Polarity.POSITIVE),
        GraphEdge(source="B_int", target="B_med", sign=Polarity.POSITIVE),
        GraphEdge(source="B_med", target="B_out", sign=Polarity.POSITIVE),  # Opposite sign: increases LDL!
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

    # Graph C: Opposite claim under INCOMPATIBLE context (Contextual Heterogeneity)
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
        context_summary=ctx_c,  # Incompatible context!
    )

    # Graph D: Disparate domain, identical topology (Cross-Domain Analogy)
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
        domain="computer_security",  # Disparate domain!
        nodes=nodes_d,
        edges=edges_d,
        context_summary=ctx_d,
    )

    return g_a, g_b, g_c, g_d


def test_feature_extractor():
    g_a, g_b, _, _ = _create_sample_graphs()
    solver = CCSTOptimalTransportSolver()
    alignment = solver.solve_pair(g_a, g_b)

    extractor = CCSTFeatureExtractor()
    feat_dict = extractor.extract_features_dict(g_a, g_b, alignment)
    feat_vec = extractor.extract_features_vector(g_a, g_b, alignment)

    assert len(feat_dict) == 16
    assert feat_vec.shape == (16,)
    assert "signed_vs_unsigned_gap" in feat_dict
    assert feat_dict["num_sign_flips"] >= 1.0


def test_calibrated_classifier_predictions():
    g_a, g_b, g_c, g_d = _create_sample_graphs()
    solver = CCSTOptimalTransportSolver()
    classifier = CCSTClassifier()

    # 1. Agreement: g_a vs g_a
    align_aa = solver.solve_pair(g_a, g_a)
    pred_aa, probs_aa, _, _ = classifier.predict(g_a, g_a, align_aa)
    assert pred_aa == RelationType.AGREEMENT
    assert probs_aa[RelationType.AGREEMENT] > 0.40

    # 2. Contradiction Candidate: g_a vs g_b (same domain, matching context, flipped sign)
    align_ab = solver.solve_pair(g_a, g_b)
    pred_ab, probs_ab, _, _ = classifier.predict(g_a, g_b, align_ab)
    assert pred_ab == RelationType.CONTRADICTION_CANDIDATE
    assert probs_ab[RelationType.CONTRADICTION_CANDIDATE] > 0.35

    # 3. Contextual Heterogeneity: g_a vs g_c (opposite findings BUT conflicting context)
    align_ac = solver.solve_pair(g_a, g_c)
    pred_ac, probs_ac, _, _ = classifier.predict(g_a, g_c, align_ac)
    assert pred_ac == RelationType.CONTEXTUAL_HETEROGENEITY
    # Contradiction MUST NOT be predicted!
    assert pred_ac != RelationType.CONTRADICTION_CANDIDATE

    # 4. Cross-Domain Analogy: g_a vs g_d (disparate domain, isomorphic topology)
    align_ad = solver.solve_pair(g_a, g_d)
    pred_ad, probs_ad, _, _ = classifier.predict(g_a, g_d, align_ad)
    assert pred_ad == RelationType.CROSS_DOMAIN_ANALOGY


def test_abstention_gate_on_high_uncertainty():
    g_a, g_b, _, _ = _create_sample_graphs()
    solver = CCSTOptimalTransportSolver()
    classifier = CCSTClassifier()
    align_ab = solver.solve_pair(g_a, g_b)

    # When extraction uncertainty is high (e.g. 0.45 > 0.40), system must abstain!
    final_pred, _, abstained, reason = classifier.predict(
        g_a, g_b, align_ab, mean_uncertainty=0.45
    )
    assert abstained is True
    assert final_pred == RelationType.INSUFFICIENT_COMPARABILITY
    assert "uncertainty" in reason.lower()


def test_certificate_builder_and_formatter():
    g_a, g_b, _, _ = _create_sample_graphs()
    solver = CCSTOptimalTransportSolver()
    classifier = CCSTClassifier()
    builder = CertificateBuilder()

    align_ab = solver.solve_pair(g_a, g_b)
    pred, probs, abstained, reason = classifier.predict(g_a, g_b, align_ab)

    cert = builder.build_certificate(
        g1=g_a,
        g2=g_b,
        alignment=align_ab,
        predicted_relation=pred,
        probabilities=probs,
        abstained=abstained,
        abstention_reason=reason,
    )

    assert cert.doc_a_id == "DOC_A"
    assert cert.doc_b_id == "DOC_B"
    assert cert.predicted_relation == RelationType.CONTRADICTION_CANDIDATE
    assert len(cert.minimal_disagreement_set) >= 1
    assert len(cert.context_diffs) == 4
    assert cert.gap_explained_fraction > 0.5

    # Check exporters
    md_report = CertificateFormatter.to_markdown(cert)
    assert "# CCST Disagreement Certificate" in md_report
    assert "CONTRADICTION_CANDIDATE" in md_report

    json_str = CertificateFormatter.to_json(cert)
    assert "minimal_disagreement_set" in json_str

    latex_row = CertificateFormatter.to_latex_row(cert)
    assert "DOC_A" in latex_row and r"\\" in latex_row
