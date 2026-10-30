"""
calculate_alpha.py - Inter-Annotator Agreement Calculator (Krippendorff's Alpha).

Section II Specification:
Natural pairs (Strata 2, 3, 4) are independently labeled by at least three annotators
using a common schema; agreement is reported with Krippendorff's alpha before adjudication.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1_foundation.schemas import BenchmarkPair, RelationType, StratumType


def compute_krippendorff_alpha_nominal(reliability_data: np.ndarray) -> float:
    """
    Computes nominal Krippendorff's alpha on a matrix of shape (num_annotators, num_items).
    Missing values can be represented as np.nan.
    Uses the official 'krippendorff' package if installed, with a self-contained fallback.
    """
    try:
        import krippendorff
        alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement="nominal")
        return float(alpha)
    except ImportError:
        pass

    # Self-contained implementation for nominal Krippendorff's alpha
    units = reliability_data.shape[1]
    coders = reliability_data.shape[0]

    # Collect unique categories
    valid_mask = ~np.isnan(reliability_data)
    unique_vals = np.unique(reliability_data[valid_mask])
    val_to_idx = {val: i for i, val in enumerate(unique_vals)}
    C = len(unique_vals)

    if C <= 1:
        return 1.0

    # Coincidence matrix
    coincidence = np.zeros((C, C))
    for u in range(units):
        col = reliability_data[:, u]
        valid_coders = col[~np.isnan(col)]
        m_u = len(valid_coders)
        if m_u < 2:
            continue
        for c1 in range(m_u):
            for c2 in range(m_u):
                if c1 != c2:
                    i = val_to_idx[valid_coders[c1]]
                    j = val_to_idx[valid_coders[c2]]
                    coincidence[i, j] += 1.0 / (m_u - 1)

    total_pairs = coincidence.sum()
    if total_pairs == 0:
        return 0.0

    # Observed disagreement
    D_o = total_pairs - np.trace(coincidence)

    # Expected disagreement
    n_k = coincidence.sum(axis=1)
    D_e = 0.0
    for k in range(C):
        for l in range(C):
            if k != l:
                D_e += n_k[k] * n_k[l]
    D_e = D_e / (total_pairs - 1) if (total_pairs - 1) > 0 else 1.0

    if D_e == 0:
        return 1.0
    return float(1.0 - (D_o / D_e))


def evaluate_benchmark_agreement(benchmark_file: Path) -> Dict[str, Any]:
    """Loads benchmark pairs, extracts annotator votes, and computes Krippendorff's alpha."""
    if not benchmark_file.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")

    pairs: List[BenchmarkPair] = []
    with open(benchmark_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(BenchmarkPair.model_validate_json(line.strip()))

    # Map RelationType to integer indices
    rel_to_int = {r: i for i, r in enumerate(RelationType)}
    int_to_rel = {i: r.value for i, r in enumerate(RelationType)}

    # Filter to pairs that have multi-annotator votes (Strata 2, 3, 4)
    annotated_pairs = [p for p in pairs if len(p.annotations) >= 2]

    if not annotated_pairs:
        return {"error": "No multi-annotator pairs found in benchmark."}

    # Identify all unique annotators
    all_annotators = sorted(list({vote.annotator_id for p in annotated_pairs for vote in p.annotations}))
    coder_to_idx = {coder: i for i, coder in enumerate(all_annotators)}

    # Build reliability matrix: shape (num_coders, num_pairs)
    matrix = np.full((len(all_annotators), len(annotated_pairs)), np.nan)

    for unit_idx, p in enumerate(annotated_pairs):
        for vote in p.annotations:
            c_idx = coder_to_idx[vote.annotator_id]
            matrix[c_idx, unit_idx] = rel_to_int[vote.assigned_relation]

    # Global Alpha across all annotated pairs
    overall_alpha = compute_krippendorff_alpha_nominal(matrix)

    # Stratum-specific alphas
    strata_alphas = {}
    for stratum in [StratumType.STRATUM_2_ANALOGY, StratumType.STRATUM_3_NATURAL, StratumType.STRATUM_4_ADVERSARIAL]:
        sub_pairs = [p for p in annotated_pairs if p.stratum == stratum]
        if sub_pairs:
            sub_matrix = np.full((len(all_annotators), len(sub_pairs)), np.nan)
            for u_idx, p in enumerate(sub_pairs):
                for vote in p.annotations:
                    c_idx = coder_to_idx[vote.annotator_id]
                    sub_matrix[c_idx, u_idx] = rel_to_int[vote.assigned_relation]
            strata_alphas[stratum.value] = round(compute_krippendorff_alpha_nominal(sub_matrix), 4)

    results = {
        "total_evaluated_pairs": len(pairs),
        "multi_annotated_pairs": len(annotated_pairs),
        "annotator_ids": all_annotators,
        "krippendorff_alpha_overall": round(overall_alpha, 4),
        "krippendorff_alpha_by_stratum": strata_alphas,
        "agreement_standard": "Nominal metric, threshold alpha >= 0.70 satisfies substantial reliability",
    }
    return results


if __name__ == "__main__":
    b_file = PROJECT_ROOT / "data" / "benchmark" / "benchmark_240.jsonl"
    if b_file.exists():
        res = evaluate_benchmark_agreement(b_file)
        print("Inter-Annotator Agreement Report:")
        print(json.dumps(res, indent=2))
    else:
        print(f"Benchmark file does not exist yet at {b_file}. Run curate_benchmark.py first.")
