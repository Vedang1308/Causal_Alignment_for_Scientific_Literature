# Phase 5: 5-Way Classification, Abstention & Certificates

## Objectives
1. **Calibrated 5-Way Classifier:**
   - Input: alignment features ($\mathcal{D}_{\text{CCST}}$, unsigned cost, signed cost, unaligned mass $U(T)$, context compatibility vector $c_h$, domain divergence).
   - Classes:
     1. `AGREEMENT`
     2. `CROSS_DOMAIN_ANALOGY`
     3. `CONTRADICTION_CANDIDATE`
     4. `CONTEXTUAL_HETEROGENEITY`
     5. `INSUFFICIENT_COMPARABILITY`
   - Calibration: Platt scaling or Isotonic Regression via `CalibratedClassifierCV`.
   - Abstention: abstains when max class probability is below confidence threshold or extraction uncertainty is high.
2. **Minimal Disagreement Certificate Generator:**
   - Aligned subgraphs (nodes, edges, coupling weights $T_{uu'}$).
   - Context diffs (matched, incompatible, unknown).
   - Minimal high-confidence disagreement set (the exact signed edges explaining the conflict).
   - Evidence span highlights grounding each claim in original abstract text.
   - Fraction of signed-vs-unsigned score gap explained.

## Planned Directory Structure
```
phase5_classification_and_certificates/
├── classifier.py            # Calibrated 5-way classification model
├── feature_extractor.py     # Alignment matrix to feature vector
├── abstention_gate.py       # Uncertainty and risk-threshold abstention
├── certificate_builder.py   # Generates human/machine readable certificate
└── formatter.py             # Markdown, JSON, and LaTeX certificate exporters
```
