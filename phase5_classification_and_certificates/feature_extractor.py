"""
Feature Extractor for CCST 5-Way Classification.
Converts CCST dual-view alignments and DocumentGraph pairs into a standardized,
calibrated feature vector x_ij in R^14.
"""

from typing import Dict, List, Any, Optional
import numpy as np

from phase1_foundation.schemas.graph import DocumentGraph
from phase4_ccst_solver_and_indexing.ot_solver import CCSTPairAlignment


FEATURE_NAMES = [
    "signed_distance",             # D_signed
    "unsigned_distance",           # D_unsigned
    "signed_vs_unsigned_gap",      # Delta_sign = D_signed - D_unsigned
    "total_aligned_mass",          # sum(T_signed)
    "unsigned_aligned_mass",       # sum(T_unsigned)
    "unaligned_mass",              # U(T) = 2 - 2*sum(T_signed)
    "semantic_cost",               # <M, T>
    "gromov_cost",                 # E_GW(T_signed)
    "unsigned_gromov_cost",        # E_GW(T_unsigned)
    "num_sign_flips",              # count of sign-inverted edges
    "gate_penalty",                # Phi_gate
    "mean_context_comp",           # Mean compatibility of known context
    "has_context_incompatibility", # 1 if any condition is incompatible, else 0
    "num_unknown_context",         # Count of unspecified context dimensions
    "same_domain",                 # 1 if domains match, else 0
    "structural_isomorphism",      # 1 - D_unsigned clamped to [0, 1]
]


class CCSTFeatureExtractor:
    """
    Extracts numerical feature representations from CCST alignment results.
    """
    def __init__(self):
        self.feature_names = FEATURE_NAMES

    def extract_features_dict(
        self,
        g1: DocumentGraph,
        g2: DocumentGraph,
        alignment: CCSTPairAlignment,
    ) -> Dict[str, float]:
        """
        Extracts a dictionary of named alignment features.
        """
        s_res = alignment.signed_result
        u_res = alignment.unsigned_result
        gate = alignment.context_gate

        num_flips = len(alignment.sign_flip_edges)
        same_dom = 1.0 if g1.domain == g2.domain else 0.0
        struct_iso = float(max(0.0, min(1.0, 1.0 - alignment.unsigned_distance)))

        return {
            "signed_distance": float(alignment.signed_distance),
            "unsigned_distance": float(alignment.unsigned_distance),
            "signed_vs_unsigned_gap": float(alignment.signed_vs_unsigned_gap),
            "total_aligned_mass": float(s_res.total_aligned_mass),
            "unsigned_aligned_mass": float(u_res.total_aligned_mass),
            "unaligned_mass": float(s_res.unaligned_mass),
            "semantic_cost": float(s_res.semantic_cost),
            "gromov_cost": float(s_res.gromov_cost),
            "unsigned_gromov_cost": float(u_res.gromov_cost),
            "num_sign_flips": float(num_flips),
            "gate_penalty": float(gate.gate_penalty),
            "mean_context_comp": float(gate.mean_known_compatibility),
            "has_context_incompatibility": 1.0 if gate.is_context_incompatible else 0.0,
            "num_unknown_context": float(gate.num_unknown),
            "same_domain": same_dom,
            "structural_isomorphism": struct_iso,
        }

    def extract_features_vector(
        self,
        g1: DocumentGraph,
        g2: DocumentGraph,
        alignment: CCSTPairAlignment,
    ) -> np.ndarray:
        """
        Extracts an ordered 1D NumPy feature vector of length 14.
        """
        feat_dict = self.extract_features_dict(g1, g2, alignment)
        return np.array([feat_dict[name] for name in self.feature_names], dtype=np.float32)
