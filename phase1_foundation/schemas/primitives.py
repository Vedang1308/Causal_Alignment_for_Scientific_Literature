"""
Primitive types, enumerations, and constants for CCST.
Defines the 5-way relation taxonomy, polarities, node roles,
epistemic statuses, and context dimensions.
"""

from enum import Enum, IntEnum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class RelationType(str, Enum):
    """
    The 5-way relation taxonomy defined in Section I of the CCST proposal:
    1. AGREEMENT: Same mechanism, same polarity, compatible material conditions.
    2. CROSS_DOMAIN_ANALOGY: Isomorphic causal structure across disparate domain vocabularies.
    3. CONTRADICTION_CANDIDATE: Same mechanism, opposing polarity, verified compatible conditions.
    4. CONTEXTUAL_HETEROGENEITY: Opposing or divergent findings explained by scope/context shift.
    5. INSUFFICIENT_COMPARABILITY: High uncertainty, missing critical context, or unaligned mechanism.
    """
    AGREEMENT = "agreement"
    CROSS_DOMAIN_ANALOGY = "cross_domain_analogy"
    CONTRADICTION_CANDIDATE = "contradiction_candidate"
    CONTEXTUAL_HETEROGENEITY = "contextual_heterogeneity"
    INSUFFICIENT_COMPARABILITY = "insufficient_comparability"


class Polarity(IntEnum):
    """
    Causal relation sign / polarity (\\delta in Equation 1):
    +1: Positive effect (promotes, increases, upregulates, activates)
    -1: Negative effect (inhibits, suppresses, decreases, blocks)
     0: Null / neutral / unobserved effect
    """
    POSITIVE = 1
    NEGATIVE = -1
    NULL = 0

    @classmethod
    def from_str(cls, val: str) -> "Polarity":
        val_clean = str(val).strip().lower()
        if val_clean in {"positive", "+1", "+", "increase", "upregulate", "promotes", "causes"}:
            return cls.POSITIVE
        elif val_clean in {"negative", "-1", "-", "decrease", "downregulate", "inhibits", "suppresses"}:
            return cls.NEGATIVE
        elif val_clean in {"null", "0", "neutral", "none", "no_effect", "unspecified"}:
            return cls.NULL
        raise ValueError(f"Unknown polarity string: {val}")


class EpistemicStatus(str, Enum):
    """
    Epistemic status of a scientific claim (\\eta in Equation 1):
    - OBSERVED: Direct experimental finding or empirical measurement.
    - HYPOTHESIZED: Theoretical conjecture, future work, or speculative hypothesis.
    - SIMULATED: Computational model, synthetic in-silico result.
    - QUOTED_NOT_ENDORSED: Quoting a prior author's claim without endorsing it (adversarial negative).
    """
    OBSERVED = "observed"
    HYPOTHESIZED = "hypothesized"
    SIMULATED = "simulated"
    QUOTED_NOT_ENDORSED = "quoted_not_endorsed"


class NodeRole(str, Enum):
    """
    Functional role of a node in a causal mechanism graph:
    - POPULATION: The target entity, species, or system (P in Eq. 1).
    - INTERVENTION: The action, treatment, trigger, or independent variable (A in Eq. 1).
    - MEDIATOR: An intermediate step or pathway in the causal chain (M in Eq. 1).
    - OUTCOME: The end result, dependent variable, or measured effect (Y in Eq. 1).
    - CONTEXT: Material environmental condition or constraint (C in Eq. 1).
    """
    POPULATION = "population"
    INTERVENTION = "intervention"
    MEDIATOR = "mediator"
    OUTCOME = "outcome"
    CONTEXT = "context"


class EdgeType(str, Enum):
    """
    Typed edge relationship in the mechanism graph.
    """
    CAUSES = "causes"
    INHIBITS = "inhibits"
    MEDIATES = "mediates"
    REGULATES = "regulates"
    CORRELATES_WITH = "correlates_with"


class ContextDimension(str, Enum):
    """
    Material context dimensions h used to compute compatibility c_h(i, j) in [0, 1].
    Missing context is strictly recorded as UNKNOWN rather than assumed compatible.
    """
    SPECIES_OR_SYSTEM = "species_or_system"  # e.g., "human adult", "mus musculus", "linux_kernel"
    DOSE_OR_SCALE = "dose_or_scale"          # e.g., "50mg/kg", "10Gbps bandwidth"
    ENVIRONMENT = "environment"              # e.g., "in vitro", "in vivo", "data-center cloud"
    PROTOCOL_OR_ASSAY = "protocol_or_assay"  # e.g., "double-blind RCT", "packet trace capture"
    TIME_HORIZON = "time_horizon"            # e.g., "24 hours", "sub-millisecond", "chronic"
    OTHER = "other"


class ContextCompatibility(float, Enum):
    """
    Material compatibility value c_h(i, j) in [0, 1] for context dimension h:
    1.0: Fully compatible conditions
    0.5: Partly compatible / adjacent conditions
    0.0: Incompatible conditions (e.g. human vs mouse cell culture)
    -1.0: Unknown / Missing condition (signals uncertainty, not compatibility!)
    """
    COMPATIBLE = 1.0
    PARTLY_COMPATIBLE = 0.5
    INCOMPATIBLE = 0.0
    UNKNOWN = -1.0


class ProvenanceSpan(BaseModel):
    """
    Source text provenance (p in Equation 1) linking extracted claims
    directly to character offsets and raw passage text in the original abstract.
    """
    start_char: int = Field(ge=0, description="Start character offset in source text")
    end_char: int = Field(ge=0, description="End character offset in source text")
    text: str = Field(min_length=1, description="Exact substring matched from the source")
    section: Optional[str] = Field(default=None, description="Section heading if available (e.g., Abstract, Results)")

    @field_validator("end_char")
    @classmethod
    def validate_span_bounds(cls, v: int, info) -> int:
        start = info.data.get("start_char", 0)
        if v < start:
            raise ValueError(f"end_char ({v}) cannot be less than start_char ({start})")
        return v
