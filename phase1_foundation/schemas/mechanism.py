"""
Mechanism Claim representation based on Equation (1) of the CCST proposal:
    q_im = (C, P, A, M, Y, τ, δ, η, u, p)_im

Preserves:
- C: Material context conditions
- P: Population or target system
- A: Intervention or trigger
- M: One or more mediating steps (causal pathway)
- Y: Outcome
- τ: Time horizon or temporal lag
- δ: Effect direction / polarity (+1, -1, 0)
- η: Epistemic status (observed, hypothesized, simulated, quoted_not_endorsed)
- u: Extraction uncertainty in [0.0, 1.0]
- p: Source provenance character span
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from .primitives import (
    ContextDimension,
    ContextCompatibility,
    Polarity,
    EpistemicStatus,
    ProvenanceSpan,
)


class MaterialContext(BaseModel):
    """
    Structured container for scope conditions (C in Eq. 1).
    Missing context is explicitly tracked rather than assumed compatible.
    """
    conditions: Dict[ContextDimension, str] = Field(
        default_factory=dict,
        description="Map of context dimension to descriptive string (e.g. {SPECIES_OR_SYSTEM: 'Mus musculus'})"
    )
    unspecified_dimensions: List[ContextDimension] = Field(
        default_factory=list,
        description="Explicitly recorded dimensions that were absent from the source text"
    )

    def get_dimension(self, dim: ContextDimension) -> Optional[str]:
        """Returns the value for a dimension, or None if unknown/missing."""
        return self.conditions.get(dim)

    def is_known(self, dim: ContextDimension) -> bool:
        """Returns True if the dimension was explicitly specified."""
        return dim in self.conditions and dim not in self.unspecified_dimensions

    def set_dimension(self, dim: ContextDimension, value: Optional[str]) -> None:
        """Sets a context dimension or marks it as unspecified if None/empty."""
        if value and value.strip():
            self.conditions[dim] = value.strip()
            if dim in self.unspecified_dimensions:
                self.unspecified_dimensions.remove(dim)
        else:
            self.conditions.pop(dim, None)
            if dim not in self.unspecified_dimensions:
                self.unspecified_dimensions.append(dim)

    def compare_dimension(
        self, other: "MaterialContext", dim: ContextDimension
    ) -> ContextCompatibility:
        """
        Computes material compatibility c_h(i, j) for a given dimension h:
        - If either condition is missing/unknown -> UNKNOWN (-1.0)
        - If values match normalized string -> COMPATIBLE (1.0)
        - If values share substrings or known overlap -> PARTLY_COMPATIBLE (0.5)
        - Otherwise -> INCOMPATIBLE (0.0)
        """
        val_a = self.get_dimension(dim)
        val_b = other.get_dimension(dim)

        if val_a is None or val_b is None:
            return ContextCompatibility.UNKNOWN

        a_norm = val_a.strip().lower()
        b_norm = val_b.strip().lower()

        if a_norm == b_norm:
            return ContextCompatibility.COMPATIBLE

        # Heuristic partial overlap (e.g., "adult human patients" vs "human patients")
        if a_norm in b_norm or b_norm in a_norm:
            return ContextCompatibility.PARTLY_COMPATIBLE

        return ContextCompatibility.INCOMPATIBLE


class MechanismClaim(BaseModel):
    """
    Representation of a single reported causal mechanism:
    q_im = (C, P, A, M, Y, τ, δ, η, u, p)_im
    """
    mechanism_id: str = Field(description="Unique identifier for the mechanism claim within the document")
    context: MaterialContext = Field(default_factory=MaterialContext, description="Scope and experimental context (C)")
    population: str = Field(description="Target population, organism, or system (P)")
    intervention: str = Field(description="Intervention, perturbation, or trigger (A)")
    mediating_steps: List[str] = Field(default_factory=list, description="Ordered intermediate steps/pathway (M)")
    outcome: str = Field(description="Observed outcome or dependent variable (Y)")
    time_horizon: Optional[str] = Field(default=None, description="Temporal duration or lag (τ)")
    polarity: Polarity = Field(default=Polarity.POSITIVE, description="Direction of causal effect δ (+1, -1, 0)")
    epistemic_status: EpistemicStatus = Field(
        default=EpistemicStatus.OBSERVED,
        description="Epistemic status η (observed, hypothesized, simulated, quoted_not_endorsed)"
    )
    uncertainty: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Extraction uncertainty u in [0.0, 1.0]"
    )
    provenance: Optional[ProvenanceSpan] = Field(
        default=None,
        description="Character span and source text passage for the claim (p)"
    )

    @field_validator("uncertainty")
    @classmethod
    def validate_uncertainty(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Uncertainty must be in [0.0, 1.0], got {v}")
        return float(v)

    def summary(self) -> str:
        """Returns a human-readable one-line summary of the causal claim."""
        sign_str = "+" if self.polarity == Polarity.POSITIVE else ("-" if self.polarity == Polarity.NEGATIVE else "0")
        meds = f" via [{', '.join(self.mediating_steps)}]" if self.mediating_steps else ""
        return (
            f"[{self.population}] {self.intervention} --({sign_str})--> "
            f"{self.outcome}{meds} (status: {self.epistemic_status.value})"
        )

