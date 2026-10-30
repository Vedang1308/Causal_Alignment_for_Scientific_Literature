"""
Calibrated 5-Way Relation Classifier for CCST.
Implements Section IV-D:
Predicts probability distribution across:
    1. AGREEMENT
    2. CROSS_DOMAIN_ANALOGY
    3. CONTRADICTION_CANDIDATE
    4. CONTEXTUAL_HETEROGENEITY
    5. INSUFFICIENT_COMPARABILITY
Supports supervised fitting with scikit-learn LogisticRegression / CalibratedClassifierCV
and a deterministic, theory-calibrated Bayesian scoring fallback.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from phase1_foundation.schemas.graph import DocumentGraph
from phase1_foundation.schemas.primitives import RelationType
from phase4_ccst_solver_and_indexing.ot_solver import CCSTPairAlignment
from .feature_extractor import CCSTFeatureExtractor, FEATURE_NAMES
from .abstention_gate import AbstentionGate


CLASSES = [
    RelationType.AGREEMENT,
    RelationType.CROSS_DOMAIN_ANALOGY,
    RelationType.CONTRADICTION_CANDIDATE,
    RelationType.CONTEXTUAL_HETEROGENEITY,
    RelationType.INSUFFICIENT_COMPARABILITY,
]


class CCSTClassifier:
    """
    Calibrated 5-way classifier with selective abstention.
    """
    def __init__(
        self,
        temperature: float = 1.0,
        enable_abstention: bool = True,
    ):
        self.classes = CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        self.idx_to_class = {i: c for i, c in enumerate(CLASSES)}
        self.temperature = temperature
        self.enable_abstention = enable_abstention

        self.feature_extractor = CCSTFeatureExtractor()
        self.abstention_gate = AbstentionGate()
        self.sklearn_model = None

    def _theory_calibrated_logits(self, feats: Dict[str, float]) -> np.ndarray:
        """
        Computes calibrated class logits directly from CCST causal alignment theory.
        """
        d_signed = feats["signed_distance"]
        d_unsigned = feats["unsigned_distance"]
        gap = feats["signed_vs_unsigned_gap"]
        aligned_mass = feats["total_aligned_mass"]
        u_aligned_mass = feats.get("unsigned_aligned_mass", aligned_mass)
        max_aligned_mass = max(aligned_mass, u_aligned_mass)
        unaligned_mass = feats["unaligned_mass"]
        flips = feats["num_sign_flips"]
        gate_pen = feats["gate_penalty"]
        has_incomp = feats["has_context_incompatibility"]
        same_dom = feats["same_domain"]
        struct_iso = feats["structural_isomorphism"]
        n_unk = feats["num_unknown_context"]
        u_gw = feats.get("unsigned_gromov_cost", 0.0)

        logits = np.zeros(5, dtype=np.float32)

        # 0: AGREEMENT
        # High aligned mass, low signed distance, 0 sign flips, compatible context
        logits[0] = (
            3.0 * (1.0 - d_signed)
            + 2.0 * aligned_mass
            - 3.5 * flips
            - 2.5 * gate_pen
            + 1.5 * same_dom
        )

        # 1: CROSS_DOMAIN_ANALOGY
        # Different domains, high unsigned aligned mass, near-zero unsigned GW distortion
        cross_dom_signal = (1.0 - same_dom) * 6.0
        logits[1] = (
            cross_dom_signal
            + 4.0 * u_aligned_mass
            + 3.0 * (1.0 - min(1.0, u_gw * 10.0))
            - 1.5 * flips
            - 5.0 * same_dom
        )

        # 2: CONTRADICTION_CANDIDATE
        # Same domain, large signed-vs-unsigned gap, sign flips, low gate penalty (verified matching context!)
        logits[2] = (
            4.0 * gap
            + 3.0 * (1.0 if flips > 0 else 0.0)
            - 4.0 * gate_pen
            - 5.0 * has_incomp
            + 1.5 * same_dom
            + 1.5 * (1.0 - d_unsigned)
        )

        # 3: CONTEXTUAL_HETEROGENEITY
        # Same domain with divergent findings (sign gap or flips) BUT context is incompatible (has_incomp = 1) or high gate penalty
        logits[3] = (
            (4.5 * has_incomp + 3.5 * gate_pen + 2.0 * gap + 1.0 * flips) * same_dom
            - 6.0 * (1.0 - same_dom)
        )

        # 4: INSUFFICIENT_COMPARABILITY
        # Low aligned mass in BOTH views, high missing context
        logits[4] = (
            3.5 * (1.0 - max_aligned_mass)
            + 1.5 * (n_unk / 4.0)
            - 3.0 * (1.0 - same_dom) * u_aligned_mass
        )

        return logits

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax with temperature scaling."""
        scaled = logits / max(1e-4, self.temperature)
        exp_z = np.exp(scaled - np.max(scaled))
        return exp_z / np.sum(exp_z)

    def predict_probabilities(
        self,
        g1: DocumentGraph,
        g2: DocumentGraph,
        alignment: CCSTPairAlignment,
    ) -> Dict[RelationType, float]:
        """
        Computes calibrated class probability distribution.
        """
        feats = self.feature_extractor.extract_features_dict(g1, g2, alignment)

        if self.sklearn_model is not None:
            vec = np.array([[feats[n] for n in FEATURE_NAMES]], dtype=np.float32)
            probs = self.sklearn_model.predict_proba(vec)[0]
        else:
            logits = self._theory_calibrated_logits(feats)
            probs = self._softmax(logits)

        return {cls: float(probs[i]) for i, cls in enumerate(self.classes)}

    def predict(
        self,
        g1: DocumentGraph,
        g2: DocumentGraph,
        alignment: CCSTPairAlignment,
        mean_uncertainty: float = 0.05,
    ) -> Tuple[RelationType, Dict[RelationType, float], bool, Optional[str]]:
        """
        Predicts 5-way relation with selective abstention.
        Returns:
            - final_relation: RelationType
            - probabilities: Dict[RelationType, float]
            - abstained: bool
            - abstention_reason: Optional[str]
        """
        probs = self.predict_probabilities(g1, g2, alignment)
        raw_pred = max(probs.keys(), key=lambda k: probs[k])

        if not self.enable_abstention:
            return raw_pred, probs, False, None

        abstained, reason, final_pred = self.abstention_gate.evaluate(
            raw_pred,
            probs,
            alignment.context_gate,
            mean_extraction_uncertainty=mean_uncertainty,
        )

        return final_pred, probs, abstained, reason

    def fit(
        self,
        pairs: List[Tuple[DocumentGraph, DocumentGraph, CCSTPairAlignment, RelationType]],
    ) -> "CCSTClassifier":
        """
        Optionally fits a scikit-learn CalibratedClassifierCV or LogisticRegression model
        on a labeled set of benchmark pairs.
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.calibration import CalibratedClassifierCV

            X = []
            y = []
            for g1, g2, align, label in pairs:
                vec = self.feature_extractor.extract_features_vector(g1, g2, align)
                X.append(vec)
                y.append(self.class_to_idx[label])

            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=int)

            if len(np.unique(y)) > 1:
                base_lr = LogisticRegression(max_iter=500, random_state=42)
                calibrated = CalibratedClassifierCV(estimator=base_lr, cv=3)
                calibrated.fit(X, y)
                self.sklearn_model = calibrated

        except (ImportError, Exception):
            pass

        return self
