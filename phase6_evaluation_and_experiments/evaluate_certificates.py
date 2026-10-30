"""
Certificate Faithfulness and Minimality Evaluator.
Evaluates:
- Deletion faithfulness: fraction of signed-vs-unsigned score gap explained
- Minimality of the disagreement set |D*|
- Provenance grounding coverage and character length
"""

from typing import List, Dict, Any
import numpy as np

from phase1_foundation.schemas.certificate import DisagreementCertificate
from phase1_foundation.schemas.primitives import RelationType


def evaluate_certificate_quality(
    certificates: List[DisagreementCertificate],
) -> Dict[str, float]:
    """
    Computes summary quality and faithfulness metrics across a collection of certificates.
    """
    if not certificates:
        return {}

    n = len(certificates)
    gap_explained = [c.gap_explained_fraction for c in certificates]
    disagreement_sizes = [len(c.minimal_disagreement_set) for c in certificates]
    aligned_nodes_counts = [len(c.aligned_nodes) for c in certificates]
    aligned_edges_counts = [len(c.aligned_edges) for c in certificates]

    # Grounding coverage: certificates with both evidence spans present
    grounded_both = sum(
        1 for c in certificates if len(c.evidence_spans_a) > 0 and len(c.evidence_spans_b) > 0
    )

    # Faithfulness on contradiction candidates specifically
    contra_certs = [c for c in certificates if c.predicted_relation == RelationType.CONTRADICTION_CANDIDATE]
    contra_gap_explained = [c.gap_explained_fraction for c in contra_certs] if contra_certs else [1.0]

    return {
        "mean_gap_explained_all": float(np.mean(gap_explained)),
        "mean_gap_explained_contra": float(np.mean(contra_gap_explained)),
        "mean_minimal_disagreement_size": float(np.mean(disagreement_sizes)),
        "mean_aligned_nodes": float(np.mean(aligned_nodes_counts)),
        "mean_aligned_edges": float(np.mean(aligned_edges_counts)),
        "provenance_grounding_coverage": float(grounded_both / n * 100.0),
        "total_certificates_audited": float(n),
    }

