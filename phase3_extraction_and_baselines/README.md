# Phase 3: Multi-Mechanism Extraction & Retrieval Baselines

This directory contains the multi-mechanism causal extractor, the graph compilation pipeline for the 2,000 documents, and the full suite of classical and modern baselines required by **Table I & Section V** of the proposal.

---

## 1. Phase 3 Architecture & Deliverables

### A. Multi-Mechanism Extractor (`extractor.py`)
* Extracts structured mechanisms according to Equation (1):
  $$q_{im} = (C, P, A, M, Y, \tau, \delta, \eta, u, p)_{im}$$
* **Strict Text Provenance ($p$):** Every node and claim retains character-level start and end offsets `(start_char, end_char)` matched directly against the source text.
* **Scope / Context Dimensions ($C$):** Species/system, environment, dosage scale, and time horizon. Missing conditions evaluate strictly to `UNKNOWN (-1.0)`.
* **Polarity ($\delta$) & Epistemic Status ($\eta$):** Classifies promotive ($+1$), inhibitory ($-1$), and null ($0$) directions, and flags `quoted_not_endorsed`, `simulated`, `hypothesized`, or `observed`.

### B. Extracted Document Graphs Corpus (`data/processed/document_graphs.jsonl`)
* **Total Graphs Compiled:** Exactly 2,000 variable-size `DocumentGraph` $G_i$ instances.
* **Total Mechanisms Extracted:** 8,015 causal claims (mean: 4.01 per document).
* **Total Graph Nodes:** 13,247 nodes with normalized mass distributions $\sum p_i = 1.0$.
* **Total Graph Edges:** 17,543 typed, signed directed causal edges.
* **Throughput:** $> 4,000$ documents/sec.

### C. Classical Retrieval Baselines (`classical_baselines/`)
1. **BM25 [11]:** Lexical frequency ranking with Okapi length normalization via `BM25Retriever`.
2. **TF-IDF + Latent Dirichlet Allocation (LDA) [12]:** Semantic topic modeling and K-Means document clustering via `TfidfLdaBaseline`.
3. Evaluates **MRR**, **Recall@k**, and **nDCG@k** over the 240 benchmark pairs.

### D. Modern AI / Deep Learning Baselines (`modern_baselines/`)
1. **Sentence-BERT / SciBERT [9]:** Dense sentence embedding retrieval (`all-mpnet-base-v2` / `scibert`).
2. **Contriever [10]:** Unsupervised dense information retrieval with contrastive learning (`facebook/contriever`).
3. **Direct LLM Pairwise Judge:** Zero-shot pairwise prompting and Transformer NLI mapping.
4. **Safety Analysis:** Replicates the critical failure mode of standard models: high **False-Contradiction Rate** on context-shifted hard negatives.

---

## 2. Directory Structure

```
phase3_extraction_and_baselines/
├── extractor.py                # Multi-mechanism extractor with provenance spans
├── extract_corpus.py           # Batch processing script over 2,000 documents
├── classical_baselines/
│   ├── __init__.py
│   ├── bm25_retriever.py       # BM25 baseline implementation
│   └── tfidf_lda.py            # TF-IDF, LDA topic model, and K-Means clustering
├── modern_baselines/
│   ├── __init__.py
│   ├── sbert_retriever.py      # SBERT / SciBERT dense embedding retriever
│   ├── contriever_retriever.py  # Contriever dense contrastive retriever
│   └── direct_llm_judge.py     # Direct LLM judge and Transformer NLI baseline
├── run_phase3_verification.py  # End-to-end verification and baselines report
├── tests/
│   └── test_phase3.py          # Pytest suite verifying extractor and baselines
└── README.md
```

---

## 3. How to Run Phase 3

### 1. Run Batch Extraction on the 2,000-Document Corpus
```bash
python phase3_extraction_and_baselines/extract_corpus.py
```

### 2. Run the Phase 3 End-to-End Verification & Baselines Benchmark
```bash
python phase3_extraction_and_baselines/run_phase3_verification.py
```

### 3. Run the Full Pytest Test Suite (Phases 1, 2, and 3)
```bash
pytest -v
```
