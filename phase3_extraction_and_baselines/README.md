# Phase 3: Multi-Mechanism Extraction & Retrieval Baselines

## Objectives
1. **Provenance-Linked Multi-Mechanism Extraction:**
   - Extract up to $M_{\max}$ causal mechanism graphs per paper using LLM/Transformer zero/few-shot prompting with strict JSON schema adherence.
   - Extract nodes, roles, edges, polarity ($\delta \in \{+1, -1, 0\}$), time lag ($\tau$), epistemic status ($\eta$), uncertainty ($u$), and character-level source span provenance ($p$).
2. **Classical Retrieval & Machine Learning Baselines:**
   - BM25 (lexical frequency ranking).
   - TF-IDF + Latent Dirichlet Allocation (LDA) with K-Means clustering.
   - Graph Edit Distance and role-constrained matching.
3. **Modern AI Baselines:**
   - Dense Sentence Embeddings: SBERT (`all-mpnet-base-v2` / `scibert`).
   - Contriever dense retrieval.
   - Direct LLM Zero-Shot Pair Prompting ("Do these two papers agree or contradict?").
   - Transformer NLI baseline.

## Planned Directory Structure
```
phase3_extraction_and_baselines/
├── extractor.py           # LLM mechanism extractor with provenance offsets
├── prompt_templates.py    # Few-shot prompts for structured extraction
├── classical_baselines/
│   ├── bm25_retriever.py  # BM25 baseline implementation
│   └── tfidf_lda.py       # TF-IDF, LDA & clustering baseline
├── modern_baselines/
│   ├── sbert_retriever.py # SBERT / SciBERT embedding retriever
│   ├── direct_llm_nli.py  # Zero-shot LLM pairwise judge
│   └── contriever_eval.py # Contriever dense retrieval
└── extract_corpus.py      # Batch extraction script over 2,000 papers
```

