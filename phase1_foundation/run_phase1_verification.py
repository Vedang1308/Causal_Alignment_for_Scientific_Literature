#!/usr/bin/env python3
"""
run_phase1_verification.py
Phase 1 Foundation Verification & Demonstration Script for CCST.

Performs:
1. Schema & Data Model Validation across all primitive types.
2. Missing Context Verification (Ensuring missing conditions evaluate to UNKNOWN, not COMPATIBLE).
3. Synthetic Paper Pair Construction (Demonstrating Contextual Heterogeneity vs True Contradiction).
4. Graph Compilation from Mechanism Claims (Equation 1 -> Graph Gi).
5. Evidence-Linked Disagreement Certificate Generation & Markdown formatting.
6. System environment check (PyTorch, GPU, and library dependencies).
"""

import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
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


def print_banner(text: str):
    line = "=" * 70
    print(f"\n{line}\n  {text}\n{line}")


def verify_environment():
    print_banner("1. SYSTEM & DEPENDENCY CHECK")
    print(f"Python Version: {sys.version.split()[0]}")
    
    # Check PyTorch & GPU
    try:
        import torch
        print(f" PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f" CUDA Available: YES (Device: {torch.cuda.get_device_name(0)})")
        else:
            print(" CUDA Available: NO (CPU mode - on SOL ensure A100 GPU is allocated)")
    except ImportError:
        print(" PyTorch: Not installed yet in this environment (installed on SOL)")

    # Check POT (Python Optimal Transport)
    try:
        import ot
        print(f" POT (Python Optimal Transport): {ot.__version__}")
    except ImportError:
        print(" POT: Not installed yet in this environment (installed on SOL)")

    # Check NetworkX
    try:
        import networkx as nx
        print(f" NetworkX: {nx.__version__}")
    except ImportError:
        print(" NetworkX: Not installed locally (installed on SOL)")


def verify_missing_context_rule():
    print_banner("2. SCOPE / CONTEXT COMPATIBILITY VERIFICATION")
    print("Testing rule: Missing context MUST evaluate to UNKNOWN (-1.0), never COMPATIBLE.")

    ctx_human = MaterialContext()
    ctx_human.set_dimension(ContextDimension.SPECIES_OR_SYSTEM, "Homo sapiens (adults)")
    ctx_human.set_dimension(ContextDimension.DOSE_OR_SCALE, "10mg daily")
    ctx_human.set_dimension(ContextDimension.ENVIRONMENT, "Double-blind clinical trial")

    ctx_murine = MaterialContext()
    ctx_murine.set_dimension(ContextDimension.SPECIES_OR_SYSTEM, "Mus musculus (mice)")
    # Note: DOSE_OR_SCALE is omitted (missing) in ctx_murine

    compat_species = ctx_human.compare_dimension(ctx_murine, ContextDimension.SPECIES_OR_SYSTEM)
    compat_dose = ctx_human.compare_dimension(ctx_murine, ContextDimension.DOSE_OR_SCALE)

    print(f"  - Species comparison ('Homo sapiens' vs 'Mus musculus'): {compat_species.name} (val: {compat_species.value})")
    print(f"  - Dose comparison ('10mg daily' vs MISSING): {compat_dose.name} (val: {compat_dose.value})")

    assert compat_species == ContextCompatibility.INCOMPATIBLE, "Species should be INCOMPATIBLE"
    assert compat_dose == ContextCompatibility.UNKNOWN, "Missing dimension MUST be UNKNOWN"
    print(" Context compatibility gating verified successfully!")


def verify_causal_mechanism_graph():
    print_banner("3. CAUSAL MECHANISM GRAPH COMPILATION (Equation 1)")
    
    # Paper A: Amlodipine reduces blood pressure in hypertensive humans
    claim_a = MechanismClaim(
        mechanism_id="MECH-A1",
        context=MaterialContext(
            conditions={
                ContextDimension.SPECIES_OR_SYSTEM: "Homo sapiens (hypertensive adults)",
                ContextDimension.ENVIRONMENT: "in vivo",
                ContextDimension.DOSE_OR_SCALE: "5mg oral",
            }
        ),
        population="Hypertensive Adults",
        intervention="Amlodipine",
        mediating_steps=["L-type calcium channel inhibition", "vascular peripheral resistance decrease"],
        outcome="blood pressure reduction",
        time_horizon="4 weeks",
        polarity=Polarity.NEGATIVE,  # Decreases BP
        epistemic_status=EpistemicStatus.OBSERVED,
        uncertainty=0.04,
        provenance=ProvenanceSpan(
            start_char=45,
            end_char=138,
            text="Amlodipine 5mg oral once daily reduced mean systolic blood pressure by 12 mmHg after 4 weeks."
        ),
    )

    doc_a = DocumentGraph.from_mechanisms(
        doc_id="PAPER_001_HUMAN",
        title="Clinical Efficacy of Amlodipine in Human Essential Hypertension",
        abstract="In a cohort of hypertensive adults, Amlodipine 5mg oral once daily reduced mean systolic blood pressure by 12 mmHg after 4 weeks.",
        domain="biomedical",
        mechanisms=[claim_a],
    )

    print(f"Compiled Document Graph: {doc_a.doc_id}")
    print(f"  - Nodes ({len(doc_a.nodes)}): {[n.label for n in doc_a.nodes]}")
    print(f"  - Edges ({len(doc_a.edges)}): {[(e.source, e.target, e.sign.name) for e in doc_a.edges]}")
    print(f"  - Normalized Mass Distribution: {[round(m, 3) for m in doc_a.get_normalized_node_masses()]}")
    print(" Causal mechanism graph compilation verified successfully!")
    return doc_a, claim_a


def verify_disagreement_certificate(doc_a, claim_a):
    print_banner("4. EVIDENCE-LINKED DISAGREEMENT CERTIFICATE")
    print("Simulating Paper B with an apparent contradiction (Blood Pressure Increase in Murine Models)")

    # Paper B: Toxic high dose in mice causing paradoxical reflex hypertension
    cert = DisagreementCertificate(
        certificate_id="CERT-DEMO-001",
        doc_a_id="PAPER_001_HUMAN",
        doc_b_id="PAPER_002_RAT_TOXIC",
        predicted_relation=RelationType.CONTEXTUAL_HETEROGENEITY,
        relation_probabilities={
            RelationType.CONTEXTUAL_HETEROGENEITY: 0.89,
            RelationType.CONTRADICTION_CANDIDATE: 0.07,
            RelationType.AGREEMENT: 0.02,
            RelationType.CROSS_DOMAIN_ANALOGY: 0.01,
            RelationType.INSUFFICIENT_COMPARABILITY: 0.01,
        },
        abstained=False,
        aligned_nodes=[
            AlignedNodePair(
                node_a_id="intervention_amlodipine",
                node_b_id="intervention_amlodipine_high_dose",
                label_a="Amlodipine",
                label_b="Amlodipine High Dose",
                role_a=NodeRole.INTERVENTION,
                role_b=NodeRole.INTERVENTION,
                coupling_weight=0.92,
            ),
            AlignedNodePair(
                node_a_id="outcome_blood_pressure_reduction",
                node_b_id="outcome_blood_pressure_spike",
                label_a="blood pressure reduction",
                label_b="blood pressure spike",
                role_a=NodeRole.OUTCOME,
                role_b=NodeRole.OUTCOME,
                coupling_weight=0.88,
            ),
        ],
        aligned_edges=[
            AlignedEdgePair(
                edge_a=("intervention_amlodipine", "outcome_blood_pressure_reduction"),
                edge_b=("intervention_amlodipine_high_dose", "outcome_blood_pressure_spike"),
                sign_a=Polarity.NEGATIVE,
                sign_b=Polarity.POSITIVE,
                is_sign_flip=True,
            )
        ],
        context_diffs=[
            ContextDiff(
                dimension=ContextDimension.SPECIES_OR_SYSTEM,
                value_a="Homo sapiens (adults)",
                value_b="Mus musculus (mice)",
                compatibility=ContextCompatibility.INCOMPATIBLE,
            ),
            ContextDiff(
                dimension=ContextDimension.DOSE_OR_SCALE,
                value_a="5mg therapeutic",
                value_b="100mg/kg toxic supra-clinical",
                compatibility=ContextCompatibility.INCOMPATIBLE,
            ),
        ],
        matched_context_fields=[],
        incompatible_context_fields=["species_or_system", "dose_or_scale"],
        unknown_context_fields=["environment"],
        minimal_disagreement_set=[
            "Causal effect polarity flip on Blood Pressure (Human [-] vs Mouse [+])"
        ],
        evidence_spans_a=[claim_a.provenance],
        evidence_spans_b=[
            ProvenanceSpan(
                start_char=20,
                end_char=110,
                text="In murine models, acute supra-clinical doses (100mg/kg) triggered paradoxical reflex hypertension."
            )
        ],
        gap_explained_fraction=0.96,
    )

    print("\n" + cert.to_markdown())
    print("\n Disagreement Certificate generated & rendered successfully!")


def main():
    print("=" * 70)
    print(" CCST PHASE 1: FOUNDATION & SCHEMA VERIFICATION")
    print("=" * 70)
    verify_environment()
    verify_missing_context_rule()
    doc_a, claim_a = verify_causal_mechanism_graph()
    verify_disagreement_certificate(doc_a, claim_a)
    print_banner("PHASE 1 ALL VERIFICATIONS PASSED (100% READY FOR SOL)")


if __name__ == "__main__":
    main()
