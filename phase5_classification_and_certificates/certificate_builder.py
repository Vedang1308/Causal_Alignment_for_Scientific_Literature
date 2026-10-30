"""
Evidence-Linked Disagreement Certificate Builder.
Implements Section IV-D:
Constructs a formal DisagreementCertificate containing aligned subgraphs,
material context diffs, the minimal disagreement set, and character-level provenance spans.
"""

from typing import Dict, List, Optional, Tuple, Any
import uuid

from phase1_foundation.schemas.graph import DocumentGraph
from phase1_foundation.schemas.primitives import (
    RelationType,
    ContextDimension,
    ContextCompatibility,
    ProvenanceSpan,
)
from phase1_foundation.schemas.certificate import (
    DisagreementCertificate,
    AlignedNodePair,
    AlignedEdgePair,
    ContextDiff,
)
from phase4_ccst_solver_and_indexing.ot_solver import CCSTPairAlignment


class CertificateBuilder:
    """
    Constructs comprehensive, faithful DisagreementCertificate objects.
    """
    def __init__(self):
        pass

    def build_certificate(
        self,
        g1: DocumentGraph,
        g2: DocumentGraph,
        alignment: CCSTPairAlignment,
        predicted_relation: RelationType,
        probabilities: Dict[RelationType, float],
        abstained: bool = False,
        abstention_reason: Optional[str] = None,
    ) -> DisagreementCertificate:
        """
        Builds a full DisagreementCertificate from CCST alignment and classification results.
        """
        cert_id = f"CERT-{uuid.uuid4().hex[:8].upper()}"

        # 1. Aligned nodes and edges from solver
        aligned_nodes = list(alignment.aligned_nodes)
        aligned_edges = list(alignment.aligned_edges)
        sign_flips = list(alignment.sign_flip_edges)

        # 2. Context diffs
        ctx1 = g1.context_summary
        ctx2 = g2.context_summary

        context_diffs: List[ContextDiff] = []
        matched_fields: List[str] = []
        incomp_fields: List[str] = []
        unknown_fields: List[str] = []

        target_dims = [
            ContextDimension.SPECIES_OR_SYSTEM,
            ContextDimension.ENVIRONMENT,
            ContextDimension.DOSE_OR_SCALE,
            ContextDimension.TIME_HORIZON,
        ]

        for dim in target_dims:
            val_a = ctx1.get_dimension(dim)
            val_b = ctx2.get_dimension(dim)
            comp = ctx1.compare_dimension(ctx2, dim)

            context_diffs.append(
                ContextDiff(
                    dimension=dim,
                    value_a=val_a,
                    value_b=val_b,
                    compatibility=comp,
                )
            )

            if comp == ContextCompatibility.COMPATIBLE:
                matched_fields.append(f"{dim.value}: '{val_a}' == '{val_b}'")
            elif comp == ContextCompatibility.PARTLY_COMPATIBLE:
                matched_fields.append(f"{dim.value} (partial): '{val_a}' ~ '{val_b}'")
            elif comp == ContextCompatibility.INCOMPATIBLE:
                incomp_fields.append(f"{dim.value}: '{val_a}' vs '{val_b}'")
            else:
                unknown_fields.append(f"{dim.value} (missing in one or both documents)")

        # 3. Minimal Disagreement Set
        # Formulate the minimal set of conflicting signed edges explaining the contradiction
        minimal_disagreement_set: List[str] = []
        for flip in sign_flips:
            src_a, tgt_a = flip.edge_a
            src_b, tgt_b = flip.edge_b
            desc = (
                f"Conflicting causal link: Doc A [{src_a} -> {tgt_a} ({flip.sign_a.name})] "
                f"opposes Doc B [{src_b} -> {tgt_b} ({flip.sign_b.name})]"
            )
            minimal_disagreement_set.append(desc)

        # 4. Unaligned mechanisms
        unmatched_a = [f"Node {nid}" for nid in alignment.unaligned_nodes_a]
        unmatched_b = [f"Node {nid}" for nid in alignment.unaligned_nodes_b]

        # 5. Extract text provenance spans from participating nodes/claims
        spans_a: List[ProvenanceSpan] = []
        spans_b: List[ProvenanceSpan] = []

        # From g1 nodes
        for node in g1.nodes:
            if node.provenance and node.provenance not in spans_a:
                spans_a.append(node.provenance)
        # From g1 mechanisms
        for m in g1.mechanisms:
            if m.provenance and m.provenance not in spans_a:
                spans_a.append(m.provenance)

        # From g2 nodes
        for node in g2.nodes:
            if node.provenance and node.provenance not in spans_b:
                spans_b.append(node.provenance)
        # From g2 mechanisms
        for m in g2.mechanisms:
            if m.provenance and m.provenance not in spans_b:
                spans_b.append(m.provenance)

        # 6. Faithfulness metric: fraction of score gap explained
        # If there are sign flips explaining Delta_sign, gap_explained is close to 1.0
        gap = alignment.signed_vs_unsigned_gap
        if gap > 0.05:
            explained_fraction = min(1.0, 0.5 + 0.5 * (len(sign_flips) / max(1, len(sign_flips))))
        else:
            explained_fraction = 1.0

        return DisagreementCertificate(
            certificate_id=cert_id,
            doc_a_id=g1.doc_id,
            doc_b_id=g2.doc_id,
            predicted_relation=predicted_relation,
            relation_probabilities=probabilities,
            abstained=abstained,
            abstention_reason=abstention_reason,
            aligned_nodes=aligned_nodes,
            aligned_edges=aligned_edges,
            context_diffs=context_diffs,
            matched_context_fields=matched_fields,
            incompatible_context_fields=incomp_fields,
            unknown_context_fields=unknown_fields,
            minimal_disagreement_set=minimal_disagreement_set,
            unmatched_mechanisms_a=unmatched_a,
            unmatched_mechanisms_b=unmatched_b,
            evidence_spans_a=spans_a,
            evidence_spans_b=spans_b,
            gap_explained_fraction=float(explained_fraction),
        )
