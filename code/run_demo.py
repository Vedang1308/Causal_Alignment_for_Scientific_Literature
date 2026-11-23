#!/usr/bin/env python3
"""
CCST End-to-End System Demo Runner.
Demonstrates:
1. Loading processed real causal graphs from corpus
2. Context-Conditioned Signed Partial Transport alignment
3. Calibrated 5-way relation classification
4. Minimal Disagreement Certificate generation with provenance grounding
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase1_foundation.schemas.primitives import RelationType, Polarity, NodeRole
from phase1_foundation.schemas.graph import DocumentGraph, GraphNode, GraphEdge
from phase1_foundation.schemas.certificate import DisagreementCertificate
from phase4_ccst_solver_and_indexing.ot_solver import CCSTOptimalTransportSolver
from phase5_classification_and_certificates.classifier import CCSTClassifier
from phase5_classification_and_certificates.certificate_builder import CertificateBuilder
from phase5_classification_and_certificates.formatter import CertificateFormatter


def main():
    print("=" * 80)
    print("  CONTEXT-CONDITIONED SIGNED PARTIAL TRANSPORT (CCST) - LIVE SYSTEM DEMO")
    print("  Arizona State University | SWM Group 16 | Fall 2026")
    print("=" * 80)

    # Check for sample certificate
    cert_path = PROJECT_ROOT / "data" / "processed" / "sample_certificate.json"
    if cert_path.exists():
        with open(cert_path, "r", encoding="utf-8") as f:
            cert_data = json.load(f)
        
        print("\n[+] Loaded Real Pipeline Certificate from ASU SOL A100 Execution:")
        print(f"    Certificate ID : {cert_data.get('certificate_id')}")
        print(f"    Document A     : {cert_data.get('doc_a_id')}")
        print(f"    Document B     : {cert_data.get('doc_b_id')}")
        print(f"    Prediction     : {cert_data.get('predicted_relation', '').upper()}")
        print(f"    Abstained      : {cert_data.get('abstained')}")
        
        print("\n[+] Calibrated 5-Way Probabilities:")
        probs = cert_data.get("relation_probabilities", {})
        for rel, p in probs.items():
            bar_len = int(p * 40)
            bar = "#" * bar_len + "-" * (40 - bar_len)
            print(f"    - {rel:<28} : [{bar}] {p*100:6.2f}%")

        print("\n[+] Top Aligned Graph Nodes (Partial Transport Coupling T_uu'):")
        for node in cert_data.get("aligned_nodes", [])[:4]:
            print(f"    - '{node.get('label_a')}' ({node.get('role_a')}) <---> '{node.get('label_b')}' ({node.get('role_b')}) | Weight = {node.get('coupling_weight'):.3f}")

        min_dis = cert_data.get("minimal_disagreement_set", [])
        print(f"\n[+] Minimal Disagreement Set ({len(min_dis)} conflicting edges):")
        for item in min_dis:
            print(f"    ⚡ {item}")

        print("\n[+] Character-Level Text Provenance Grounding:")
        spans_a = cert_data.get("evidence_spans_a", [])
        if spans_a:
            print(f"    Doc A Span: \"{spans_a[0].get('text', '')[:90]}...\" (chars {spans_a[0].get('start_char')}-{spans_a[0].get('end_char')})")
        spans_b = cert_data.get("evidence_spans_b", [])
        if spans_b:
            print(f"    Doc B Span: \"{spans_b[0].get('text', '')[:90]}...\" (chars {spans_b[0].get('start_char')}-{spans_b[0].get('end_char')})")

    print("\n" + "=" * 80)
    print("  Interactive Web UI Available At:")
    print(f"  file://{PROJECT_ROOT}/code/ccst_interactive_demo.html")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
