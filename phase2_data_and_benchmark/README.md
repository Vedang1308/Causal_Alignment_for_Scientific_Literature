# Phase 2: Data Acquisition & Benchmark Curation

## Objectives
1. **Corpus Acquisition:** Fetch and validate a 2,000-document public scientific title-abstract corpus from **PubMed** (biomedical) and **arXiv** (computer systems/security, control systems, and related quantitative disciplines).
2. **Benchmark Construction (240 Ground-Truth Evaluated Pairs across 4 Strata):**
   - **Stratum 1 (Controlled Transformations - 60 pairs):** Systematically altered vocabularies, polarities, contexts, and noise.
   - **Stratum 2 (Cross-Domain Analogies - 60 pairs):** Isomorphic causal structures across different domains.
   - **Stratum 3 (Natural Science Pairs - 60 pairs):** Scientific agreement, genuine contradiction, and contextual heterogeneity.
   - **Stratum 4 (Adversarial Hard Negatives - 60 pairs):** Quoted-but-not-endorsed claims, simulations vs. observations, and lexical distractors.
3. **Inter-Annotator Agreement:** Multi-annotator labeling with Krippendorff's $\alpha$ calculation before adjudication.

## Planned Directory Structure
```
phase2_data_and_benchmark/
├── fetch_arxiv.py          # Automated arXiv API scraper
├── fetch_pubmed.py         # Automated PubMed E-Utilities scraper
├── build_corpus.py         # Filtering (English, min 80 words, no retractions) & deduplication
├── curate_benchmark.py     # Annotation support & pair generator
├── calculate_alpha.py      # Krippendorff's alpha agreement calculator
├── annotation_guide.md     # Official rubric for annotators
└── data/
    ├── raw_corpus.jsonl    # 2,000 papers
    └── benchmark_240.jsonl # 240 annotated pairs
```
