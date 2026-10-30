"""
Abstention Gate for Selective Classification.
Implements Section IV-D:
Evaluates confidence margin, prediction entropy, and context missingness risk.
Prevents high-risk misclassifications by selectively abstaining to INSUFFICIENT_COMPARABILITY.
"""

from typing import Dict, Tuple, Optional
import math
from phase1_foundation.schemas.primitives import RelationType
from phase4_ccst_solver_and_indexing.context_gate import ContextGateEvaluation


class AbstentionGate:
    """
    Evaluates whether the system should abstain from committing to a brittle prediction.
    """
    def __init__(
        self,
        confidence_threshold: float = 0.45,
        entropy_threshold: float = 1.35,
        max_uncertainty_threshold: float = 0.40,
    ):
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.max_uncertainty_threshold = max_uncertainty_threshold

    def evaluate_entropy(self, probabilities: Dict[RelationType, float]) -> float:
        """Computes Shannon entropy H(p) = -sum p_k * log(p_k)."""
        entropy = 0.0
        for p in probabilities.values():
            if p > 1e-12:
                entropy -= p * math.log(p)
        return float(entropy)

    def evaluate(
        self,
        predicted_relation: RelationType,
        probabilities: Dict[RelationType, float],
        context_eval: ContextGateEvaluation,
        mean_extraction_uncertainty: float = 0.05,
    ) -> Tuple[bool, Optional[str], RelationType]:
        """
        Determines whether to abstain.
        Returns:
            - should_abstain: bool
            - abstention_reason: Optional[str]
            - final_relation: RelationType (falls back to INSUFFICIENT_COMPARABILITY if abstained)
        """
        max_prob = max(probabilities.values()) if probabilities else 0.0
        entropy = self.evaluate_entropy(probabilities)

        # Rule 1: High extraction uncertainty
        if mean_extraction_uncertainty > self.max_uncertainty_threshold:
            reason = f"High extraction uncertainty ({mean_extraction_uncertainty:.2f} > {self.max_uncertainty_threshold:.2f}) from raw text."
            return True, reason, RelationType.INSUFFICIENT_COMPARABILITY

        # Rule 2: Ambiguous classification probability (low confidence)
        if max_prob < self.confidence_threshold:
            reason = f"Low prediction confidence ({max_prob:.3f} < {self.confidence_threshold:.3f}); entropy = {entropy:.2f}."
            return True, reason, RelationType.INSUFFICIENT_COMPARABILITY

        # Rule 3: High distribution entropy
        if entropy > self.entropy_threshold:
            reason = f"High prediction entropy ({entropy:.3f} > {self.entropy_threshold:.3f}) indicating conflicting signals."
            return True, reason, RelationType.INSUFFICIENT_COMPARABILITY

        # Rule 4: Context Gating violation on Contradiction Candidates
        if predicted_relation == RelationType.CONTRADICTION_CANDIDATE:
            if not context_eval.allows_contradiction:
                # If context is incompatible, it is Contextual Heterogeneity, not abstention
                if context_eval.is_context_incompatible:
                    return False, None, RelationType.CONTEXTUAL_HETEROGENEITY
                # If context is unknown/unverified, abstain from confirming contradiction
                reason = f"Contradiction candidate unverified: material conditions missing/unknown ({context_eval.num_unknown} unspecified field(s))."
                return True, reason, RelationType.INSUFFICIENT_COMPARABILITY

        return False, None, predicted_relation

