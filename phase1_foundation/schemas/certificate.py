"""
Minimal Evidence-Linked Disagreement Certificate representation
defined in Section IV-D of the CCST proposal.

Contains:
- Aligned nodes and edges with coupling weights
- Matched, incompatible, and unknown context fields
- Smallest high-confidence disagreement set (sign-flip edges)
- Unmatched mechanisms (unaligned mass)
- Supporting text spans from both papers
- Explained fraction of signed-versus-unsigned score gap
"""

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from .primitives import (
    RelationType,
    Polarity,
    ContextDimension,
    ContextCompatibility,
    ProvenanceSpan,
    NodeRole,
)


class AlignedNodePair(BaseModel):
    """Pair of aligned nodes between Document A and Document B."""
    node_a_id: str
    node_b_id: str
    label_a: str
    label_b: str
    role_a: NodeRole
    role_b: NodeRole
    coupling_weight: float = Field(ge=0.0, description="Transport mass T_uu' assigned by CCST")


class AlignedEdgePair(BaseModel):
    """Pair of aligned edges between Document A and Document B."""
    edge_a: Tuple[str, str] = Field(description="(source_a, target_a)")
    edge_b: Tuple[str, str] = Field(description="(source_b, target_b)")
    sign_a: Polarity
    sign_b: Polarity
    is_sign_flip: bool = Field(description="True if sign_a != sign_b and neither is null")
    edge_type_a: str = "causes"
    edge_type_b: str = "causes"


class ContextDiff(BaseModel):
    """Detailed comparison across a specific material context dimension."""
    dimension: ContextDimension
    value_a: Optional[str] = None
    value_b: Optional[str] = None
    compatibility: ContextCompatibility


class DisagreementCertificate(BaseModel):
    """
    Evidence-grounded minimal disagreement certificate produced for evaluated pairs.
    """
    certificate_id: str = Field(description="Unique certificate identifier")
    doc_a_id: str = Field(description="First document ID")
    doc_b_id: str = Field(description="Second document ID")
    predicted_relation: RelationType = Field(description="Final predicted 5-way relation")
    relation_probabilities: Dict[RelationType, float] = Field(
        default_factory=dict,
        description="Calibrated probability distribution across all 5 classes"
    )
    abstained: bool = Field(default=False, description="True if system abstained due to high risk/uncertainty")
    abstention_reason: Optional[str] = Field(default=None, description="Reason for abstaining if abstained=True")

    # Aligned components
    aligned_nodes: List[AlignedNodePair] = Field(default_factory=list)
    aligned_edges: List[AlignedEdgePair] = Field(default_factory=list)

    # Scope / Context reconciliation
    context_diffs: List[ContextDiff] = Field(default_factory=list)
    matched_context_fields: List[str] = Field(default_factory=list)
    incompatible_context_fields: List[str] = Field(default_factory=list)
    unknown_context_fields: List[str] = Field(default_factory=list)

    # Disagreement & Partial Unmatched sets
    minimal_disagreement_set: List[str] = Field(
        default_factory=list,
        description="Smallest set of conflicting signed edges explaining the contradiction"
    )
    unmatched_mechanisms_a: List[str] = Field(default_factory=list, description="Mechanisms in A left unaligned")
    unmatched_mechanisms_b: List[str] = Field(default_factory=list, description="Mechanisms in B left unaligned")

    # Provenance grounding
    evidence_spans_a: List[ProvenanceSpan] = Field(default_factory=list)
    evidence_spans_b: List[ProvenanceSpan] = Field(default_factory=list)

    # Faithfulness metric
    gap_explained_fraction: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of the signed-vs-unsigned score gap explained by certificate elements"
    )

    def to_markdown(self) -> str:
        """Generates a clean markdown audit report for human verification."""
        lines = [
            f"# Disagreement Certificate: `{self.certificate_id}`",
            f"**Doc A:** `{self.doc_a_id}` | **Doc B:** `{self.doc_b_id}`",
            f"**Assigned Relation:** `{self.predicted_relation.value.upper()}` "
            f"(Abstained: {'YES - ' + str(self.abstention_reason) if self.abstained else 'NO'})",
            "",
            "### Calibrated Probabilities",
        ]
        for rel, prob in sorted(self.relation_probabilities.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- **{rel.value}**: `{prob:.3f}`")

        lines.extend([
            "",
            "### Context Compatibility",
            f"- **Matched Dimensions:** {', '.join(self.matched_context_fields) or 'None'}",
            f"- **Incompatible Dimensions:** {', '.join(self.incompatible_context_fields) or 'None'}",
            f"- **Unknown Dimensions:** {', '.join(self.unknown_context_fields) or 'None'}",
            "",
            "### Minimal Disagreement Set (Sign Flips)",
        ])
        if self.minimal_disagreement_set:
            for item in self.minimal_disagreement_set:
                lines.append(f"- ⚠️ `{item}`")
        else:
            lines.append("- *(No direct sign contradiction detected)*")

        lines.extend([
            "",
            "### Key Aligned Entities",
        ])
        for p in self.aligned_nodes[:5]:
            lines.append(f"- `{p.label_a}` ({p.role_a.value}) ⟷ `{p.label_b}` ({p.role_b.value}) [mass: {p.coupling_weight:.2f}]")

        lines.extend([
            "",
            "### Text Provenance Evidence",
            "**Document A Evidence:**",
        ])
        for s in self.evidence_spans_a:
            lines.append(f"> \"{s.text}\" (chars {s.start_char}-{s.end_char})")

        lines.append("\n**Document B Evidence:**")
        for s in self.evidence_spans_b:
            lines.append(f"> \"{s.text}\" (chars {s.start_char}-{s.end_char})")

        return "\n".join(lines)

