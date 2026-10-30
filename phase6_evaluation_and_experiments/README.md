# Phase 6: Statistical Evaluation, Ablations & Reporting

## Objectives
1. **Held-Out Test Set Evaluation:**
   - Strict 60/20/20 train-dev-test partition with no document or transformation leakage.
   - 5 random seeds, bootstrapped 95% confidence intervals, and paired significance tests.
2. **Metrics:**
   - **Extraction:** Field F1 and evidence span IoU.
   - **Alignment:** Node/edge correspondence F1, sign-flip localization accuracy.
   - **Relation:** 5-class macro-F1, per-class P/R, Expected Calibration Error (ECE), abstention coverage.
   - **Critical Safety Metric:** False-contradiction rate on context-shift hard negatives.
   - **Retrieval:** MRR, Recall@k, nDCG.
   - **Certificates:** Deletion faithfulness, minimality, and evidence-span precision/recall.
3. **Ablation Studies:**
   - Impact of context gating ($\lambda_g = 0$).
   - Impact of partial transport ($\lambda_u = 0$, forced complete coupling).
   - Impact of signed vs. unsigned edge comparisons.
   - Removal of each context dimension.
4. **Final Deliverables:**
   - LaTeX tables, confusion matrices, reliability curves, scaling plots, and slide deck artifacts.

## Planned Directory Structure
```
phase6_evaluation_and_experiments/
├── run_heldout_eval.py      # Core test evaluation script
├── evaluate_extraction.py   # Field and span F1 calculator
├── evaluate_alignment.py    # Node/edge alignment metrics
├── evaluate_certificates.py # Faithfulness and minimality tests
├── run_ablations.py         # Context-gate, sign, and partial ablations
├── statistical_tests.py     # Bootstrapped 95% CIs and significance tests
└── plot_results.py          # Publication-ready figures and LaTeX tables
```

