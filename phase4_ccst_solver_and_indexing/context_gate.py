"""
Context Gating Penalty (Phi_gate) for CCST Alignment.
Implements Section IV-B:
Prevents pairs with incompatible or unknown material scope conditions
from being classified as contradictions based solely on low structural graph distance.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from phase1_foundation.schemas.graph import DocumentGraph
from phase1_foundation.schemas.mechanism import MaterialContext
from phase1_foundation.schemas.primitives import ContextDimension, ContextCompatibility


@dataclass
class ContextGateEvaluation:
    """
    Detailed audit of material context compatibility and gating penalty.
    """
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    dimension_compatibilities: Dict[str, str] = field(default_factory=dict)
    
    mean_known_compatibility: float = 1.0
    num_compatible: int = 0
    num_partly_compatible: int = 0
    num_incompatible: int = 0
    num_unknown: int = 0
    
    is_context_incompatible: bool = False
    is_context_unknown: bool = False
    
    # Gating penalty Phi_gate normalized in [0.0, 1.0]
    gate_penalty: float = 0.0
    
    # Core scientific rule flag: can this pair validly be deemed a true contradiction?
    allows_contradiction: bool = True
    invalidation_reason: Optional[str] = None


class ContextGate:
    """
    Computes context compatibility and gating penalty between two DocumentGraphs.
    """
    def __init__(
        self,
        kappa_unknown: float = 0.8,
        incompatible_penalty: float = 1.0,
        partly_compatible_penalty: float = 0.4,
    ):
        """
        Parameters:
            kappa_unknown: Penalty psi(-1) for missing/unspecified context.
                           Default 0.8 ensures unverified context cannot slip through.
            incompatible_penalty: Penalty psi(0.0) for directly conflicting context.
            partly_compatible_penalty: Penalty psi(0.5) for partially matching context.
        """
        self.kappa_unknown = kappa_unknown
        self.incompatible_penalty = incompatible_penalty
        self.partly_compatible_penalty = partly_compatible_penalty
        self.target_dimensions = [
            ContextDimension.SPECIES_OR_SYSTEM,
            ContextDimension.ENVIRONMENT,
            ContextDimension.DOSE_OR_SCALE,
            ContextDimension.TIME_HORIZON,
        ]

    def evaluate(self, g1: DocumentGraph, g2: DocumentGraph) -> ContextGateEvaluation:
        """
        Evaluates material conditions between two document graphs.
        Returns a ContextGateEvaluation object.
        """
        ctx1 = g1.context_summary
        ctx2 = g2.context_summary

        dim_scores: Dict[str, float] = {}
        dim_comp: Dict[str, str] = {}
        
        penalties: List[float] = []
        known_scores: List[float] = []
        
        n_comp = 0
        n_part = 0
        n_incomp = 0
        n_unk = 0

        for dim in self.target_dimensions:
            comp = ctx1.compare_dimension(ctx2, dim)
            dim_comp[dim.value] = comp.name

            if comp == ContextCompatibility.COMPATIBLE:
                dim_scores[dim.value] = 1.0
                penalties.append(0.0)
                known_scores.append(1.0)
                n_comp += 1
            elif comp == ContextCompatibility.PARTLY_COMPATIBLE:
                dim_scores[dim.value] = 0.5
                penalties.append(self.partly_compatible_penalty)
                known_scores.append(0.5)
                n_part += 1
            elif comp == ContextCompatibility.INCOMPATIBLE:
                dim_scores[dim.value] = 0.0
                penalties.append(self.incompatible_penalty)
                known_scores.append(0.0)
                n_incomp += 1
            else:  # UNKNOWN
                dim_scores[dim.value] = -1.0
                penalties.append(self.kappa_unknown)
                n_unk += 1

        mean_known = (sum(known_scores) / len(known_scores)) if known_scores else 0.5
        overall_penalty = (sum(penalties) / len(penalties)) if penalties else 0.0

        is_incompatible = (n_incomp > 0)
        is_unknown = (n_unk > 0)

        # Enforce CCST core rule:
        # Contradictions require verified, compatible experimental conditions.
        allows_contradiction = True
        reason = None

        if is_incompatible:
            allows_contradiction = False
            reason = f"Material context conflict detected ({n_incomp} conflicting condition(s)). Findings constitute Contextual Heterogeneity."
        elif is_unknown and mean_known < 0.7:
            allows_contradiction = False
            reason = f"Material context is insufficiently specified ({n_unk} missing condition(s)). Comparability is unverified."

        return ContextGateEvaluation(
            dimension_scores=dim_scores,
            dimension_compatibilities=dim_comp,
            mean_known_compatibility=mean_known,
            num_compatible=n_comp,
            num_partly_compatible=n_part,
            num_incompatible=n_incomp,
            num_unknown=n_unk,
            is_context_incompatible=is_incompatible,
            is_context_unknown=is_unknown,
            gate_penalty=overall_penalty,
            allows_contradiction=allows_contradiction,
            invalidation_reason=reason,
        )
