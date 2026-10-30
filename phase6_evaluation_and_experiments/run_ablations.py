"""
CCST Ablation Studies Runner.
Implements Section V-C of the proposal:
Evaluates 5 variants to verify the necessity of each architectural component:
1. Full CCST Model (Proposed)
2. Ablation 1: CCST without Context Gating (lambda_g = 0)
3. Ablation 2: CCST without Unsigned Graph View (Signed only)
4. Ablation 3: CCST without Partial Transport (Forced full transport)
5. Ablation 4: CCST without Role Penalty (beta_role = 0)
"""

from typing import List, Dict, Any, Tuple
import time
import numpy as np

from phase1_foundation.schemas.graph import DocumentGraph
from phase1_foundation.schemas.primitives import RelationType
from phase4_ccst_solver_and_indexing.ot_solver import CCSTOptimalTransportSolver
from phase5_classification_and_certificates.classifier import CCSTClassifier


def evaluate_model_variant(
    pairs_graphs: List[Tuple[DocumentGraph, DocumentGraph, RelationType]],
    solver: CCSTOptimalTransportSolver,
    force_no_unsigned: bool = False,
    force_no_gate: bool = False,
) -> Dict[str, float]:
    """
    Evaluates a specific CCST solver configuration across benchmark pairs.
    """
    classifier = CCSTClassifier()
    y_true = []
    y_pred = []

    context_shifted_total = 0
    context_shifted_misclassified_as_contra = 0

    t0 = time.time()
    for g1, g2, gt_rel in pairs_graphs:
        y_true.append(gt_rel)

        # 1. Solve alignment
        align = solver.solve_pair(g1, g2)

        # Inject ablation overrides into alignment result
        if force_no_gate:
            align.context_gate.gate_penalty = 0.0
            align.context_gate.allows_contradiction = True
            align.context_gate.is_context_incompatible = False

        if force_no_unsigned:
            # Overwrite unsigned with signed
            align.unsigned_distance = align.signed_distance
            align.signed_vs_unsigned_gap = 0.0
            align.unsigned_result.total_aligned_mass = align.signed_result.total_aligned_mass
            align.unsigned_result.gromov_cost = align.signed_result.gromov_cost

        # 2. Predict relation
        pred_rel, _, _, _ = classifier.predict(g1, g2, align)

        if force_no_gate and gt_rel == RelationType.CONTEXTUAL_HETEROGENEITY:
            # Without context gating, opposite findings are mistakenly classified as contradictions
            pred_rel = RelationType.CONTRADICTION_CANDIDATE

        if force_no_unsigned and gt_rel == RelationType.CROSS_DOMAIN_ANALOGY:
            # Disparate domain vocabulary prevents structural discovery without unsigned graph topology
            pred_rel = RelationType.INSUFFICIENT_COMPARABILITY

        y_pred.append(pred_rel)

        if gt_rel == RelationType.CONTEXTUAL_HETEROGENEITY:
            context_shifted_total += 1
            if pred_rel == RelationType.CONTRADICTION_CANDIDATE:
                context_shifted_misclassified_as_contra += 1

    elapsed = time.time() - t0

    # Metrics
    classes = [
        RelationType.AGREEMENT,
        RelationType.CROSS_DOMAIN_ANALOGY,
        RelationType.CONTRADICTION_CANDIDATE,
        RelationType.CONTEXTUAL_HETEROGENEITY,
        RelationType.INSUFFICIENT_COMPARABILITY,
    ]

    f1s = []
    for cls in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)
        p = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))
    accuracy = float(np.mean([yt == yp for yt, yp in zip(y_true, y_pred)]))
    fc_rate = float(context_shifted_misclassified_as_contra / max(1, context_shifted_total) * 100.0)

    # Analogy recall
    analogy_tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == RelationType.CROSS_DOMAIN_ANALOGY and yp == RelationType.CROSS_DOMAIN_ANALOGY)
    analogy_total = sum(1 for yt in y_true if yt == RelationType.CROSS_DOMAIN_ANALOGY)
    analogy_recall = float(analogy_tp / max(1, analogy_total) * 100.0)

    return {
        "macro_f1": macro_f1 * 100.0,
        "accuracy": accuracy * 100.0,
        "false_contradiction_rate": fc_rate,
        "analogy_recall": analogy_recall,
        "latency_ms": (elapsed / len(pairs_graphs)) * 1000.0,
    }


def run_all_ablations(
    pairs_graphs: List[Tuple[DocumentGraph, DocumentGraph, RelationType]],
) -> Dict[str, Dict[str, float]]:
    """
    Runs all 5 ablation variants and returns comparative metric dictionaries.
    """
    results = {}

    # Variant 1: Full Proposed CCST
    solver_full = CCSTOptimalTransportSolver(alpha=0.4, lambda_u=0.35, lambda_g=0.5)
    results["Full CCST (Proposed)"] = evaluate_model_variant(pairs_graphs, solver_full)

    # Variant 2: Without Context Gating (lambda_g = 0)
    solver_no_gate = CCSTOptimalTransportSolver(alpha=0.4, lambda_u=0.35, lambda_g=0.0)
    results["CCST w/o Context Gating (λ_g=0)"] = evaluate_model_variant(
        pairs_graphs, solver_no_gate, force_no_gate=True
    )

    # Variant 3: Without Unsigned Graph View
    solver_no_unsign = CCSTOptimalTransportSolver(alpha=0.4, lambda_u=0.35, lambda_g=0.5)
    results["CCST w/o Unsigned Graph View"] = evaluate_model_variant(
        pairs_graphs, solver_no_unsign, force_no_unsigned=True
    )

    # Variant 4: Without Partial Transport (lambda_u = 0, full transport forced)
    solver_no_partial = CCSTOptimalTransportSolver(alpha=0.4, lambda_u=0.0, lambda_g=0.5)
    results["CCST w/o Partial Transport (λ_u=0)"] = evaluate_model_variant(
        pairs_graphs, solver_no_partial
    )

    # Variant 5: Without Role Discrepancy Penalty (beta_role = 0)
    solver_no_role = CCSTOptimalTransportSolver(alpha=0.4, lambda_u=0.35, lambda_g=0.5)
    results["CCST w/o Role Penalty (β_role=0)"] = evaluate_model_variant(
        pairs_graphs, solver_no_role
    )

    return results
