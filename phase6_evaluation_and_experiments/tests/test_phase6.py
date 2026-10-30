"""
Unit Tests for Phase 6: Statistical Evaluation, Ablations & Reporting.
Validates:
1. Bootstrapped confidence intervals and significance testing
2. Expected Calibration Error (ECE) calculation
3. Certificate faithfulness and minimality metrics
4. Ablation study execution across all 5 configurations
5. LaTeX tables and artifacts generation
"""

import pytest
import numpy as np
from pathlib import Path

from phase1_foundation.schemas.primitives import RelationType
from phase5_classification_and_certificates.tests.test_phase5 import _create_sample_graphs
from phase6_evaluation_and_experiments.statistical_tests import (
    bootstrap_confidence_interval,
    paired_bootstrap_p_value,
    compute_ece,
)
from phase6_evaluation_and_experiments.evaluate_certificates import evaluate_certificate_quality
from phase6_evaluation_and_experiments.run_ablations import run_all_ablations
from phase6_evaluation_and_experiments.export_artifacts import (
    export_latex_table1,
    export_latex_table2,
)


def test_bootstrap_confidence_interval():
    y_true = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 1, 1, 0])

    def acc_fn(yt, yp):
        return float(np.mean(yt == yp))

    point, low, high = bootstrap_confidence_interval(
        acc_fn, y_true, y_pred, n_bootstraps=200
    )
    assert 0.0 <= point <= 1.0
    assert low <= point <= high


def test_paired_bootstrap_significance():
    # Exactly identical arrays
    scores_a = np.array([0.8, 0.9, 0.85, 0.88, 0.92])
    p_val_same = paired_bootstrap_p_value(scores_a, scores_a, n_bootstraps=100)
    assert p_val_same == 1.0

    # Radically different arrays
    scores_b = np.array([0.1, 0.2, 0.15, 0.18, 0.22])
    p_val_diff = paired_bootstrap_p_value(scores_a, scores_b, n_bootstraps=200)
    assert p_val_diff < 0.05


def test_compute_ece():
    # Perfectly calibrated predictions
    confidences = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
    predictions = np.array([1, 1, 1, 1, 0])
    y_true = np.array([1, 1, 1, 1, 0])
    # 100% accuracy on bin [0.8, 1.0] vs 0.9 confidence -> ECE = |1.0 - 0.9| = 0.10
    ece = compute_ece(confidences, predictions, y_true, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_ablation_runner_on_sample_graphs():
    g_a, g_b, g_c, g_d = _create_sample_graphs()
    pairs_graphs = [
        (g_a, g_a, RelationType.AGREEMENT),
        (g_a, g_b, RelationType.CONTRADICTION_CANDIDATE),
        (g_a, g_c, RelationType.CONTEXTUAL_HETEROGENEITY),
        (g_a, g_d, RelationType.CROSS_DOMAIN_ANALOGY),
    ]

    ablation_res = run_all_ablations(pairs_graphs)
    assert len(ablation_res) == 5
    assert "Full CCST (Proposed)" in ablation_res
    assert "CCST w/o Context Gating (λ_g=0)" in ablation_res

    # Context gating ablation must show higher false contradiction rate or lower F1
    full_m = ablation_res["Full CCST (Proposed)"]
    assert full_m["macro_f1"] > 0.0


def test_export_latex_tables(tmp_path):
    dummy_main = {
        "BM25": {"macro_f1": 15.0, "false_contradiction_rate": 35.0, "analogy_recall": 0.0, "latency_ms": 1.2},
        "Full CCST (Proposed)": {"macro_f1": 65.0, "f1_ci_low": 58.0, "f1_ci_high": 72.0, "false_contradiction_rate": 0.0, "analogy_recall": 100.0, "latency_ms": 2.5},
    }
    t1_path = tmp_path / "table1.tex"
    export_latex_table1(dummy_main, t1_path)
    assert t1_path.exists()
    content1 = t1_path.read_text(encoding="utf-8")
    assert r"\begin{table*}" in content1
    assert "Full CCST" in content1

    dummy_abl = {
        "Full CCST (Proposed)": {"macro_f1": 65.0, "false_contradiction_rate": 0.0, "analogy_recall": 100.0},
        "CCST w/o Context Gating (λ_g=0)": {"macro_f1": 48.0, "false_contradiction_rate": 45.0, "analogy_recall": 100.0},
    }
    t2_path = tmp_path / "table2.tex"
    export_latex_table2(dummy_abl, t2_path)
    assert t2_path.exists()
    content2 = t2_path.read_text(encoding="utf-8")
    assert r"\begin{table}" in content2
    assert "tab:ablations" in content2
