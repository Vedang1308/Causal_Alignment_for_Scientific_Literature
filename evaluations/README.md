# CCST Empirical Evaluations & Experimental Results: `evaluations/`

This directory houses the evaluation pipelines, benchmark tables, ablation logs, and certificate audit summaries for **CCST**.

---

## 1. Quick Reproduction
To re-run the full held-out evaluation on the 240-pair benchmark:
```bash
python3 evaluations/run_evaluations.py
```

---

## 2. Benchmark Evaluation Summary (Table I)

Evaluation across a 60/20/20 train/dev/test split with **1,000 bootstrap resamples** (95% Confidence Intervals):

| Model / Framework | Macro F1 (%) | False Contradiction Rate (%) | Analogy Recall (%) | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **BM25 Lexical Retrieval** | 14.8 | 38.0 | 0.0 | 1.20 |
| **TF-IDF + LDA Topic Model** | 18.2 | 34.0 | 5.0 | 2.10 |
| **Sentence-BERT / SciBERT** | 24.5 | 42.0 | 0.0 | 4.50 |
| **Contriever (Unsupervised)** | 22.1 | 44.0 | 0.0 | 4.20 |
| **Direct LLM Pairwise Judge** | 21.4 | 40.0 | 30.0 | 12.00 |
| **CCST (Proposed, NVIDIA A100)** | **43.9 [34.3, 51.8]** | **0.00** | **100.0** | **5.77** |

---

## 3. Component Ablation Study (Table II)

| Ablation Variant | Macro F1 (%) | Accuracy (%) | False Contradiction Rate (%) | Analogy Recall (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Full CCST System** | **47.9** | **57.1** | **0.0** | **100.0** |
| w/o Context Gating ($\lambda_g = 0$) | 36.9 | 50.0 | **100.0** *(Critical Failure)* | 100.0 |
| w/o Unsigned Graph View | 32.0 | 38.3 | 0.0 | **0.0** *(Analogy Blindness)* |
| w/o Partial Transport ($\lambda_u = 0$) | 44.1 | 56.2 | 0.0 | 100.0 |
| w/o Role Penalty ($\beta_{\text{role}} = 0$) | 47.9 | 57.1 | 0.0 | 100.0 |

---

## 4. Key Findings
1. **Zero False Contradictions:** Standard vector baselines mistake context shifts for causal contradictions 38%–44% of the time. CCST eliminates false contradictions completely (**0.00%**).
2. **100% Cross-Domain Analogy Recall:** Because CCST aligns topological structure through an unsigned partial transport graph view, it successfully identifies causal analogies across completely disjoint vocabularies (e.g., Computer Systems vs Virology).
3. **Sub-6ms Inference on A100 GPU:** Full optimal transport alignment runs in **5.77 ms** per document pair.
4. **Verifiable Disagreement Certificates:** 100% of contradiction predictions are accompanied by exact sign-inverted causal edges and source text character-level provenance.
