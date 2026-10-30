"""
Benchmark Data Schemas for the 240 Evaluated Pairs across 4 Strata,
defined in Section II of the CCST proposal.

Strata:
1. Controlled mechanism graphs with synthetic changes (vocab, sign, context, noise)
2. Natural cross-domain analogies
3. Scientific contradiction, agreement, and contextual heterogeneity
4. Adversarial hard negatives (quoted-not-endorsed, sim-vs-obs, lexical distractors)
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from .primitives import RelationType


class StratumType(str, Enum):
    """The four benchmark strata defined in Section II."""
    STRATUM_1_CONTROLLED = "stratum_1_controlled"
    STRATUM_2_ANALOGY = "stratum_2_analogy"
    STRATUM_3_NATURAL = "stratum_3_natural"
    STRATUM_4_ADVERSARIAL = "stratum_4_adversarial"


class StratumSubtype(str, Enum):
    """Specific perturbation or generation family."""
    # Stratum 1: Controlled
    CONTROLLED_VOCAB_MUTATION = "controlled_vocab_mutation"
    CONTROLLED_SIGN_FLIP = "controlled_sign_flip"
    CONTROLLED_CONTEXT_SHIFT = "controlled_context_shift"
    CONTROLLED_NOISE_INJECTION = "controlled_noise_injection"

    # Stratum 2: Analogies
    NATURAL_CROSS_DOMAIN_ANALOGY = "natural_cross_domain_analogy"

    # Stratum 3: Natural Science
    NATURAL_AGREEMENT = "natural_agreement"
    NATURAL_CONTRADICTION = "natural_contradiction"
    NATURAL_CONTEXTUAL_HETEROGENEITY = "natural_contextual_heterogeneity"

    # Stratum 4: Adversarial Hard Negatives
    ADVERSARIAL_QUOTED_NOT_ENDORSED = "adversarial_quoted_not_endorsed"
    ADVERSARIAL_SIM_VS_OBSERVATION = "adversarial_sim_vs_observation"
    ADVERSARIAL_LEXICAL_DISTRACTOR = "adversarial_lexical_distractor"
    ADVERSARIAL_MISSING_CONTEXT = "adversarial_missing_context"


class AnnotatorVote(BaseModel):
    """Individual annotator judgment used for Krippendorff's alpha calculation."""
    annotator_id: str = Field(description="Identifier of the annotator (e.g. 'annotator_1')")
    assigned_relation: RelationType = Field(description="The chosen relation category")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Annotator self-rated confidence")
    notes: Optional[str] = Field(default=None, description="Qualitative justification")


class DocumentRecord(BaseModel):
    """Cleaned document metadata for a paper in a benchmark pair."""
    doc_id: str
    title: str
    abstract: str
    domain: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkPair(BaseModel):
    """
    Evaluated paper pair in the 240-pair benchmark dataset.
    """
    pair_id: str = Field(description="Unique pair identifier (e.g. 'PAIR-STR1-001')")
    stratum: StratumType = Field(description="Which stratum this pair belongs to")
    subtype: StratumSubtype = Field(description="Detailed subtype / generation family")
    doc_a: DocumentRecord = Field(description="First document in the pair")
    doc_b: DocumentRecord = Field(description="Second document in the pair")
    ground_truth_relation: RelationType = Field(description="Adjudicated true relation label")

    # Annotations for inter-annotator agreement
    annotations: List[AnnotatorVote] = Field(
        default_factory=list,
        description="Independent annotations (minimum 3 annotators for natural pairs)"
    )
    adjudicated: bool = Field(default=True, description="True once consensus label is finalized")

    # Structural ground truths for alignment evaluation
    expected_aligned_nodes: Optional[Dict[str, str]] = Field(
        default=None,
        description="Expected mapping from doc_a node labels to doc_b node labels"
    )
    expected_sign_flip: bool = Field(default=False, description="True if the pair features an inverted causal sign")
    expected_context_compatible: Optional[bool] = Field(
        default=None,
        description="True if context conditions are materially compatible; False if shifted; None if unknown"
    )
    notes: Optional[str] = Field(default=None, description="Curator notes or rationale")

