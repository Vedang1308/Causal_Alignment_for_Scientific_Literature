# Phase 6: Statistical Evaluation, Ablations & Final Reporting

This directory contains the held-out experimental evaluation, statistical confidence intervals, comprehensive ablation studies, and publication-ready LaTeX tables/artifacts as specified in **Section V** of the CCST proposal.

---

## 1. Experimental Setup & Protocol

- **Dataset Partition:** Strict 60/20/20 stratified split across the 240 benchmark pairs:
  - **Train:** 144 pairs
  - **Dev:** 48 pairs
  - **Test (Held-out):** 48 pairs
- **Statistical Significance:** 1,000-sample bootstrap 95% confidence intervals and paired bootstrap hypothesis tests ($p < 0.05$).
- **Calibration Metric:** Expected Calibration Error (ECE) across confidence bins.

---

## 2. Main Experimental Results (Table I)

| System / Architecture | Macro F1 (%) | 95% Bootstrap CI | False-Contradiction (%) | Analogy Recall (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25 Retrieval [11]** | 14.8 | - | 38.0 | 0.0 | 1.2 |
| **TF-IDF + LDA [12]** | 18.2 | - | 34.0 | 5.0 | 2.1 |
| **Sentence-BERT / SciBERT [9]** | 24.5 | - | 42.0 | 0.0 | 4.5 |
| **Contriever (Unsupervised) [10]** | 22.1 | - | 44.0 | 0.0 | 4.2 |
| **Direct LLM Pairwise Judge** | 21.4 | - | 40.0 | 30.0 | 12.0 |
| **Full CCST (Proposed)** | **43.9** | **[34.3, 51.8]** | **0.0** | **100.0** | **2.2** |

> **Key Finding:** While standard semantic and LLM judges suffer an alarming **40%–44% false contradiction rate** on context-shifted literature, CCST eliminates **100% of false contradictions** ($0.0\%$ error rate) via context gating ($\Phi_{\text{gate}}$), while discovering cross-domain analogies with **100% recall**.

---

## 3. Ablation Studies (Table II)

| Ablation Configuration | Macro F1 (%) | False-Contradiction (%) | Analogy Recall (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **Full CCST (Proposed)** | **47.9** | **0.0** | **100.0** | 1.9 |
| **CCST w/o Context Gating ($\lambda_g = 0$)** | 36.9 | **100.0** | 100.0 | 2.0 |
| **CCST w/o Unsigned Graph View** | 32.0 | 0.0 | **0.0** | 1.9 |
| **CCST w/o Partial Transport ($\lambda_u = 0$)** | 44.1 | 0.0 | 100.0 | 1.9 |
| **CCST w/o Role Penalty ($\beta_{\text{role}} = 0$)** | 47.9 | 0.0 | 100.0 | 1.9 |

> **Ablation Takeaways:**
> 1. **Context Gating ($\Phi_{\text{gate}}$):** Without context gating, false-contradictions spike to **100%**, proving that optimal transport without context awareness cannot distinguish genuine contradictions from contextual heterogeneity.
> 2. **Unsigned Topological View:** Without the unsigned graph view, analogy recall collapses to **0.0%**, confirming that lexical/signed approaches fail to discover cross-domain mechanisms.

---

## 4. Generated Publication Artifacts

- **LaTeX Table I (Main Results):** [`data/processed/table1_main_results.tex`](../data/processed/table1_main_results.tex)
- **LaTeX Table II (Ablations):** [`data/processed/table2_ablations.tex`](../data/processed/table2_ablations.tex)
- **Evaluation Metrics JSON:** [`data/processed/final_evaluation_summary.json`](../data/processed/final_evaluation_summary.json)
- **Sample Disagreement Certificate:** [`data/processed/sample_certificate.json`](../data/processed/sample_certificate.json)

---

## 5. Directory Layout

```
phase6_evaluation_and_experiments/
├── statistical_tests.py     # Bootstrapped 95% CIs, p-values, ECE calculation
├── evaluate_certificates.py # Faithfulness and minimality metrics
├── run_ablations.py         # 5-variant CCST ablation studies runner
├── run_heldout_eval.py      # 60/20/20 train/dev/test evaluation harness
├── export_artifacts.py      # LaTeX Table I, Table II, and JSON exporters
├── run_phase6_verification.py # Master Phase 6 evaluation runner
└── tests/
    └── test_phase6.py       # 5 unit tests validating stats, CIs, ablations, LaTeX
```

---

## 6. Verification Instructions

```bash
# 1. Run the Phase 6 verification harness (Held-out eval + Ablations + LaTeX generation)
python phase6_evaluation_and_experiments/run_phase6_verification.py

# 2. Run the complete pytest suite across the entire project (32 tests across all 6 phases)
pytest -v
```
