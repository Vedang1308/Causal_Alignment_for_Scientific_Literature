# CCST Scientific Datasets: `data/`

This directory houses the real published scientific corpora, 4-stratum evaluation benchmarks, document graphs, and neural embeddings for **CCST**.

---

## 1. Directory Structure

```
data/
├── raw/                      # Raw API responses and temporary download caches
├── benchmark/
│   ├── benchmark_240.jsonl   # 240 expert-annotated pairs across 4 balanced strata (α = 0.767)
│   └── inter_annotator_matrix.csv # Full multi-annotator agreement matrix
└── processed/
    ├── corpus.jsonl          # 2,054 real scientific papers from PubMed and arXiv
    ├── corpus_doc_ids.json   # Deterministic document indexing dictionary
    ├── corpus_embeddings.npy # (2054, 384) float32 dense Sentence-BERT embedding matrix
    ├── document_graphs.jsonl # 2,054 structured causal mechanism graphs (16,638 mechanisms)
    ├── extraction_metadata.json # Provenance extraction metadata
    ├── final_evaluation_summary.json # Benchmark metrics and CI scores
    ├── sample_certificate.json # Full minimal disagreement certificate artifact
    ├── table1_main_results.tex # LaTeX Table I (Baselines vs CCST)
    └── table2_ablations.tex  # LaTeX Table II (Component Ablation Study)
```

---

## 2. Dataset Statistics

- **Total Published Papers:** 2,054 real full-text scientific papers
  - **PubMed (Biomedical & Clinical):** 1,027 articles fetched via live NCBI E-Utilities API
  - **arXiv (Distributed Systems & AI):** 1,027 papers (`cs.CR`, `cs.NI`, `cs.SY`, `eess.SY`) fetched via live arXiv API
- **Quality Controls:**
  - Length filter: $\ge 50$ words
  - Language: English only
  - Retraction filter: Zero retracted or discontinued papers
  - Deduplication: Exact MD5 abstract hash matching
- **Extracted Causal Graphs:**
  - Mechanisms extracted: **16,638**
  - Graph nodes: **11,945**
  - Causal edges: **35,057**
  - Text Provenance: **100.0%** character-span verification against raw paper text
- **Evaluation Benchmark:**
  - **Stratum 1 (Replication / Agreement):** 60 pairs
  - **Stratum 2 (Cross-Domain Analogies):** 60 pairs
  - **Stratum 3 (Contextual Heterogeneity):** 60 pairs
  - **Stratum 4 (True Contradiction Candidates):** 60 pairs
  - **Inter-Annotator Reliability:** Krippendorff's $\alpha = 0.767$ across 3 independent expert reviewers
