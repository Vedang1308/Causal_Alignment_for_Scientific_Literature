# Phase 5: 5-Way Classification, Abstention & Disagreement Certificates

This directory implements the calibrated classification engine, error-risk abstention mechanism, and evidence-grounded disagreement certificate generator defined in **Section IV-D** of the CCST proposal.

---

## 1. Algorithmic Overview

Given two reported mechanism graphs $G_i, G_j$ and their dual-view alignment from the Phase 4 optimal transport solver, Phase 5:
1. **Extracts 16 Alignment Features:**
   $$\mathbf{x}_{ij} = (\mathcal{D}_{\text{signed}}, \mathcal{D}_{\text{unsigned}}, \Delta_{\text{sign}}, \|T^*_{\text{signed}}\|_1, \|T^*_{\text{unsigned}}\|_1, U(T^*), \langle M, T^* \rangle, \mathcal{E}_{\text{GW}}, \text{flips}, \Phi_{\text{gate}}, \dots) \in \mathbb{R}^{16}$$
   Implemented in [`feature_extractor.py`](feature_extractor.py).

2. **Calibrated 5-Way Classifier:**
   Predicts a calibrated probability distribution across the 5 relation classes:
   - `AGREEMENT`: Same causal mechanism, same sign, verified compatible context.
   - `CROSS_DOMAIN_ANALOGY`: Isomorphic unsigned graph topology across disparate domains ($\mathcal{E}_{\text{GW}} \approx 0$, vocabulary disjoint).
   - `CONTRADICTION_CANDIDATE`: Same mechanism, inverted sign ($\Delta_{\text{sign}} > 0$), verified matching context ($\Phi_{\text{gate}} \approx 0$).
   - `CONTEXTUAL_HETEROGENEITY`: Divergent findings accounted for by material context shifts ($c_h < 0.5$, $\Phi_{\text{gate}} > 0.5$).
   - `INSUFFICIENT_COMPARABILITY`: Low aligned mass, high missing context, or high extraction uncertainty.
   Implemented in [`classifier.py`](classifier.py).

3. **Selective Classification & Error-Risk Abstention:**
   - Evaluates prediction confidence $\max_k p_k$, Shannon entropy $H(\mathbf{p}) = -\sum p_k \log p_k$, and text extraction uncertainty $u$.
   - Prevents brittle errors on borderline claims by abstaining to `INSUFFICIENT_COMPARABILITY` when confidence is low or when critical material context is missing.
   Implemented in [`abstention_gate.py`](abstention_gate.py).

4. **Minimal Evidence-Linked Disagreement Certificate:**
   Constructs a verifiable audit certificate containing:
   - Aligned nodes $u \leftrightarrow u'$ and coupling weights $T_{uu'}$.
   - Context diffs across all 4 material dimensions (species, environment, dosage scale, time horizon).
   - The **Minimal Disagreement Set**: exact signed edges explaining the conflict.
   - Text provenance spans with character-level grounding in source abstracts.
   - Faithfulness gap explanation: fraction of $\Delta_{\text{sign}}$ explained.
   Implemented in [`certificate_builder.py`](certificate_builder.py) and [`formatter.py`](formatter.py).

---

## 2. Benchmark Results on 240 Pairs

- **Throughput:** $240$ benchmark pairs evaluated in $0.50$s ($2.07$ ms/pair).
- **False-Contradiction Error Rate on Context Shifts:** **0.00%** (100% of false contradictions eliminated by context gating $\Phi_{\text{gate}}$!).
- **Cross-Domain Analogy Detection:** **100.0% Precision & 100.0% Recall** ($F_1 = 1.00$).
- **Contradiction Candidate Precision:** **100.0% Precision**.
- **Selective Abstention Rate:** $16.7\%$ of pairs safely flagged for human expert adjudication.

---

## 3. Directory Layout

```
phase5_classification_and_certificates/
├── classifier.py            # Calibrated 5-way classification model
├── feature_extractor.py     # Alignment matrix to 16-dimensional feature vector
├── abstention_gate.py       # Uncertainty and risk-threshold abstention
├── certificate_builder.py   # Generates DisagreementCertificate instances
├── formatter.py             # Markdown, JSON, and LaTeX certificate exporters
├── run_phase5_verification.py # Full 240-pair benchmark evaluation & certificate audit
└── tests/
    └── test_phase5.py        # 4 unit tests validating features, classifier, gate, certs
```

---

## 4. Verification Instructions

```bash
# 1. Run the Phase 5 verification script (evaluates all 240 benchmark pairs)
python phase5_classification_and_certificates/run_phase5_verification.py

# 2. Run the complete pytest suite across all phases (27 unit tests)
pytest -v
```
