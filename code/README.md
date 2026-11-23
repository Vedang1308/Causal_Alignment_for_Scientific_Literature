# CCST Software Implementation: `code/`

This directory houses the core algorithms, models, and execution scripts for **Context-Conditioned Signed Partial Transport (CCST)**.

---

## 1. Quickstart: Live Demo
Run the end-to-end alignment demo in under 1 second:
```bash
python3 code/run_demo.py
```

To explore the interactive visual UI with bipartite mechanism graph alignments, material context diffs, and minimal disagreement certificates:
Open `code/ccst_interactive_demo.html` in any web browser.

---

## 2. Six-Phase Architecture Overview
The codebase is structured into 6 modular, decoupled phases:

| Module / Phase | Description | Key Files |
| :--- | :--- | :--- |
| **`phase1_foundation/`** | Pydantic v2 schemas for mechanisms, primitives, graphs, and certificates | `schemas/primitives.py`, `schemas/graph.py`, `schemas/certificate.py` |
| **`phase2_data_and_benchmark/`** | Data ingestion from NCBI PubMed & arXiv, benchmark curation, and inter-annotator agreement | `fetch_real_corpus.py`, `curate_benchmark.py`, `calculate_alpha.py` |
| **`phase3_extraction_and_baselines/`** | Multi-mechanism causal graph extraction & classical/modern baselines (BM25, TF-IDF/LDA, SBERT, Contriever, NLI) | `extract_corpus.py`, `extractor.py`, `classical_baselines/`, `modern_baselines/` |
| **`phase4_ccst_solver_and_indexing/`** | Context-Conditioned Signed Partial Transport solver & tri-index candidate retrieval | `ot_solver.py`, `context_gate.py`, `candidate_retriever.py`, `indexes/` |
| **`phase5_classification_and_certificates/`** | Calibrated 5-way relation classifier, selective abstention gate, and evidence certificate builder | `classifier.py`, `abstention_gate.py`, `certificate_builder.py`, `formatter.py` |
| **`phase6_evaluation_and_experiments/`** | Held-out 60/20/20 evaluation, 1,000 bootstrap significance testing, and ablation matrices | `run_heldout_eval.py`, `run_ablations.py`, `evaluate_certificates.py` |

---

## 3. Running Verification & Unit Tests
Run the entire 32-test regression suite:
```bash
pytest -v
```
All 32 tests pass in 0.29s.
