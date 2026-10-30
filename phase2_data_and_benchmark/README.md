# Phase 2: Data Acquisition & Benchmark Curation

This directory contains the automated data acquisition scrapers, the corpus builder, the 4-stratum ground-truth benchmark generator, and the inter-annotator agreement evaluation suite for **CCST**.

---

## 1. Phase 2 Deliverables & Specifications

### A. Frozen Public Scientific Corpus (`data/processed/corpus.jsonl`)
* **Total Documents:** Exactly 2,000 title-abstract records
* **Disciplines:**
  * Biomedical: 524 documents (26.2%)
  * Computer Security: 486 documents (24.3%)
  * Control Systems: 484 documents (24.2%)
  * Networking: 506 documents (25.3%)
* **Strict Quality Filters:**
  * Retractions and expressions of concern excluded.
  * Duplicate records removed via normalized fingerprinting.
  * English language verified ($\ge 85\%$ ASCII and validated stopword profile).
  * Minimum abstract word count $\ge 50$ words (mean: 66.5 words).

### B. Four-Stratum Benchmark Dataset (`data/benchmark/benchmark_240.jsonl`)
* **Total Evaluated Pairs:** Exactly 240 pairs (balanced 60 pairs per stratum):
  1. **Stratum 1 (Controlled Mutations - 60 pairs):**
     * 15 Vocabulary mutations (synonym mechanisms $\to$ Agreement)
     * 15 Sign flips ($\pm \to \mp$ under identical context $\to$ Contradiction Candidate)
     * 15 Context shifts (different population/assay $\to$ Contextual Heterogeneity)
     * 15 Noise injections (extraneous mechanisms $\to$ Agreement via Partial Transport)
  2. **Stratum 2 (Cross-Domain Analogies - 60 pairs):**
     * Isomorphic causal mechanisms bridging biomedicine, security, control, and networks.
  3. **Stratum 3 (Natural Science Pairs - 60 pairs):**
     * 20 Agreement, 20 Contradiction Candidates, 20 Contextual Heterogeneity.
  4. **Stratum 4 (Adversarial Hard Negatives - 60 pairs):**
     * 15 Quoted-not-endorsed claims
     * 15 Simulation vs. physical observation
     * 15 Missing context cases (triggering mandatory abstention)
     * 15 Lexical distractors (shared words, reversed/unrelated causal graph)

### C. Pre-Adjudication Inter-Annotator Reliability
* **Multi-Annotated Pairs:** 180 natural pairs (Strata 2, 3, 4) reviewed by 3 independent annotators.
* **Krippendorff's $\alpha$ (Nominal):** $\alpha = 0.7667$ ($\ge 0.70$ satisfies substantial agreement standard).

---

## 2. Directory Contents

```
phase2_data_and_benchmark/
├── fetch_arxiv.py             # Automated arXiv API scraper for cs.CR, cs.NI, cs.SY
├── fetch_pubmed.py            # Automated NCBI PubMed E-Utilities API scraper
├── build_corpus.py            # Corpus builder, deduplicator, and quality filter
├── curate_benchmark.py        # 4-Stratum 240-pair benchmark builder
├── calculate_alpha.py         # Krippendorff's alpha inter-annotator agreement tool
├── annotation_guide.md        # Official annotation rubric and decision trees
├── run_phase2_verification.py # Executable verification and audit runner
└── tests/
    └── test_phase2.py         # Pytest suite verifying corpus, benchmark, and alpha
```

---

## 3. How to Run Phase 2

### Run the Phase 2 End-to-End Verification Runner
```bash
python phase2_data_and_benchmark/run_phase2_verification.py
```

### Run the Full Pytest Test Suite
```bash
pytest -v
```

### Recompute Krippendorff's $\alpha$
```bash
python phase2_data_and_benchmark/calculate_alpha.py
```
