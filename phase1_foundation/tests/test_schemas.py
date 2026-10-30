"""
Comprehensive Unit Tests for Phase 1 Schemas and Data Models.
Verifies:
1. Mechanism Claim instantiation and field constraints (Equation 1).
2. Unknown Context handling (missingness marked unknown, not compatible).
3. Document Graph compilation and mass normalization.
4. Minimal Disagreement Certificate generation and markdown rendering.
5. Benchmark Pair serialization and annotation tracking.
"""

import pytest
from phase1_foundation.schemas import (
    RelationType,
    Polarity,
    EpistemicStatus,
    NodeRole,
    EdgeType,
    ContextDimension,
    ContextCompatibility,
    ProvenanceSpan,
    MaterialContext,
    MechanismClaim,
    GraphNode,
    GraphEdge,
    DocumentGraph,
    AlignedNodePair,
    AlignedEdgePair,
    ContextDiff,
    DisagreementCertificate,
    StratumType,
    StratumSubtype,
    AnnotatorVote,
    DocumentRecord,
    BenchmarkPair,
)


def test_polarity_parsing():
    """Verify polarity conversion from string and integers."""
    assert Polarity.from_str("positive") == Polarity.POSITIVE
    assert Polarity.from_str("+1") == Polarity.POSITIVE
    assert Polarity.from_str("decrease") == Polarity.NEGATIVE
    assert Polarity.from_str("-1") == Polarity.NEGATIVE
    assert Polarity.from_str("none") == Polarity.NULL
    assert Polarity.from_str("0") == Polarity.NULL

    with pytest.raises(ValueError):
        Polarity.from_str("invalid_polarity_value")


def test_provenance_span_validation():
    """Verify character offset ordering and validation."""
    span = ProvenanceSpan(start_char=10, end_char=50, text="Atorvastatin reduces LDL-C")
    assert span.start_char == 10
    assert span.end_char == 50
    assert span.text == "Atorvastatin reduces LDL-C"

    # End char before start char should fail
    with pytest.raises(ValueError):
        ProvenanceSpan(start_char=50, end_char=10, text="Invalid Span")


def test_material_context_missingness_handling():
    """
    CRITICAL REQUIREMENT from proposal:
    Missing context is marked unknown rather than assumed compatible.
    """
    ctx_a = MaterialContext()
    ctx_a.set_dimension(ContextDimension.SPECIES_OR_SYSTEM, "Mus musculus")
    ctx_a.set_dimension(ContextDimension.DOSE_OR_SCALE, "10 mg/kg")

    ctx_b = MaterialContext()
    ctx_b.set_dimension(ContextDimension.SPECIES_OR_SYSTEM, "Homo sapiens")
    # Dose is left missing/unspecified in ctx_b

    # Species is incompatible (mouse vs human)
    compat_species = ctx_a.compare_dimension(ctx_b, ContextDimension.SPECIES_OR_SYSTEM)
    assert compat_species == ContextCompatibility.INCOMPATIBLE

    # Missing dose MUST evaluate to UNKNOWN (-1.0), NOT compatible!
    compat_dose = ctx_a.compare_dimension(ctx_b, ContextDimension.DOSE_OR_SCALE)
    assert compat_dose == ContextCompatibility.UNKNOWN

    # Unspecified dimensions on both must also evaluate to UNKNOWN
    compat_env = ctx_a.compare_dimension(ctx_b, ContextDimension.ENVIRONMENT)
    assert compat_env == ContextCompatibility.UNKNOWN


def test_mechanism_claim_representation():
    """Verify MechanismClaim q_im = (C, P, A, M, Y, τ, δ, η, u, p)_im."""
    ctx = MaterialContext()
    ctx.set_dimension(ContextDimension.SPECIES_OR_SYSTEM, "Adult Patients with Hypertension")
    ctx.set_dimension(ContextDimension.ENVIRONMENT, "Double-Blind Randomized Trial")

    prov = ProvenanceSpan(
        start_char=120,
        end_char=235,
        text="Amlodipine 5mg once daily produced significant reductions in mean systolic blood pressure after 4 weeks."
    )

    claim = MechanismClaim(
        mechanism_id="MECH_001",
        context=ctx,
        population="Adult Patients with Hypertension",
        intervention="Amlodipine 5mg",
        mediating_steps=["L-type calcium channel inhibition", "vascular smooth muscle relaxation"],
        outcome="systolic blood pressure reduction",
        time_horizon="4 weeks",
        polarity=Polarity.NEGATIVE,  # Decreases blood pressure
        epistemic_status=EpistemicStatus.OBSERVED,
        uncertainty=0.05,
        provenance=prov,
    )

    assert claim.mechanism_id == "MECH_001"
    assert len(claim.mediating_steps) == 2
    assert claim.polarity == Polarity.NEGATIVE
    assert claim.epistemic_status == EpistemicStatus.OBSERVED
    assert claim.uncertainty == 0.05
    assert "Amlodipine" in claim.summary()


def test_document_graph_compilation_and_mass_normalization():
    """Verify compiling mechanisms into a unified graph and normalizing mass."""
    ctx = MaterialContext()
    ctx.set_dimension(ContextDimension.SPECIES_OR_SYSTEM, "Human Adults")

    claim1 = MechanismClaim(
        mechanism_id="M1",
        context=ctx,
        population="Human Adults",
        intervention="Atorvastatin",
        mediating_steps=["HMG-CoA reductase inhibition"],
        outcome="LDL cholesterol reduction",
        polarity=Polarity.NEGATIVE,
        uncertainty=0.0,
    )

    claim2 = MechanismClaim(
        mechanism_id="M2",
        context=ctx,
        population="Human Adults",
        intervention="Atorvastatin",
        mediating_steps=[],
        outcome="cardiovascular event risk reduction",
        polarity=Polarity.NEGATIVE,
        uncertainty=0.1,
    )

    doc_graph = DocumentGraph.from_mechanisms(
        doc_id="PMID:12345678",
        title="Effects of Atorvastatin on Cardiovascular Risk",
        abstract="We studied atorvastatin in human adults and found significant LDL reductions...",
        domain="biomedical",
        mechanisms=[claim1, claim2],
    )

    assert doc_graph.doc_id == "PMID:12345678"
    assert len(doc_graph.nodes) > 0
    assert len(doc_graph.edges) > 0

    # Node masses should sum to 1.0 when normalized
    masses = doc_graph.get_normalized_node_masses()
    assert len(masses) == len(doc_graph.nodes)
    assert pytest.approx(sum(masses), rel=1e-5) == 1.0


def test_disagreement_certificate_generation():
    """Verify creating a disagreement certificate and rendering markdown output."""
    node_a = AlignedNodePair(
        node_a_id="n_amlodipine",
        node_b_id="n_felodipine",
        label_a="Amlodipine",
        label_b="Felodipine",
        role_a=NodeRole.INTERVENTION,
        role_b=NodeRole.INTERVENTION,
        coupling_weight=0.85,
    )

    edge_a = AlignedEdgePair(
        edge_a=("n_amlodipine", "n_bp"),
        edge_b=("n_felodipine", "n_bp"),
        sign_a=Polarity.NEGATIVE,  # Decreased BP
        sign_b=Polarity.POSITIVE,  # Increased BP (Conflict!)
        is_sign_flip=True,
    )

    cert = DisagreementCertificate(
        certificate_id="CERT-2026-001",
        doc_a_id="DOC_A",
        doc_b_id="DOC_B",
        predicted_relation=RelationType.CONTRADICTION_CANDIDATE,
        relation_probabilities={
            RelationType.CONTRADICTION_CANDIDATE: 0.88,
            RelationType.CONTEXTUAL_HETEROGENEITY: 0.08,
            RelationType.AGREEMENT: 0.02,
            RelationType.CROSS_DOMAIN_ANALOGY: 0.01,
            RelationType.INSUFFICIENT_COMPARABILITY: 0.01,
        },
        aligned_nodes=[node_a],
        aligned_edges=[edge_a],
        matched_context_fields=["species_or_system"],
        incompatible_context_fields=[],
        unknown_context_fields=["dose_or_scale"],
        minimal_disagreement_set=["(n_amlodipine, n_bp)[-] vs (n_felodipine, n_bp)[+]"],
        evidence_spans_a=[
            ProvenanceSpan(start_char=0, end_char=35, text="Amlodipine significantly decreased BP.")
        ],
        evidence_spans_b=[
            ProvenanceSpan(start_char=0, end_char=34, text="Felodipine paradoxically spiked BP.")
        ],
        gap_explained_fraction=0.94,
    )

    assert cert.predicted_relation == RelationType.CONTRADICTION_CANDIDATE
    assert cert.aligned_edges[0].is_sign_flip is True

    md = cert.to_markdown()
    assert "CERT-2026-001" in md
    assert "CONTRADICTION_CANDIDATE" in md
    assert "Amlodipine significantly decreased BP." in md


def test_benchmark_pair_validation():
    """Verify 4-stratum benchmark pair schema and annotator vote aggregation."""
    doc_a = DocumentRecord(
        doc_id="PMID:001",
        title="Drug X in Clinical Trials",
        abstract="Drug X decreases blood pressure in human hypertension patients.",
        domain="biomedical",
    )
    doc_b = DocumentRecord(
        doc_id="PMID:002",
        title="Drug X in Rat Model",
        abstract="Drug X increases blood pressure in hypertensive rats at high toxic dose.",
        domain="biomedical",
    )

    pair = BenchmarkPair(
        pair_id="PAIR-STR3-042",
        stratum=StratumType.STRATUM_3_NATURAL,
        subtype=StratumSubtype.NATURAL_CONTEXTUAL_HETEROGENEITY,
        doc_a=doc_a,
        doc_b=doc_b,
        ground_truth_relation=RelationType.CONTEXTUAL_HETEROGENEITY,
        annotations=[
            AnnotatorVote(annotator_id="A1", assigned_relation=RelationType.CONTEXTUAL_HETEROGENEITY),
            AnnotatorVote(annotator_id="A2", assigned_relation=RelationType.CONTEXTUAL_HETEROGENEITY),
            AnnotatorVote(annotator_id="A3", assigned_relation=RelationType.CONTRADICTION_CANDIDATE),
        ],
        expected_sign_flip=True,
        expected_context_compatible=False,  # Difference in species (human vs rat)
        notes="Species mismatch explains polarity divergence.",
    )

    assert pair.pair_id == "PAIR-STR3-042"
    assert pair.stratum == StratumType.STRATUM_3_NATURAL
    assert len(pair.annotations) == 3
    assert pair.expected_context_compatible is False
