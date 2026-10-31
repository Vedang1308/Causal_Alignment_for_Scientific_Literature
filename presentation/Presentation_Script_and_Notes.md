# CCST Mid-Term Progress Presentation: Speaker Script & Presenter Guide

**Project Title:** Context-Conditioned Causal Alignment for Scientific Literature (CCST)  
**Team 16:** Shravan Krishna Vijay, Sri Harshith Goli, Subhash Shivaprasad, Tarun Kumar Bantupalli, Vedang Vasant Avaghade  
**Course:** Software & Web Mining (SWM) Group Project, Fall 2026  
**Target Time:** 12–15 Minutes (plus Q&A)

---

## Slide-by-Slide Speaker Guide

### Slide 1: Title & Introduction
* **Presenter:** Vedang Vasant Avaghade
* **Time:** ~1:00 min
* **Key Talking Points:**
  * "Good morning/afternoon everyone. We are Team 16, and today we are presenting the mid-term progress on our project: *Context-Conditioned Causal Alignment for Scientific Literature (CCST)*."
  * "Our goal is to solve a fundamental challenge in scientific literature mining: how to separate genuine cross-domain analogies, true contradictions, and contextual differences at corpus scale."
  * "Today, we will share the motivation, our data acquisition of over 2,000 real papers, our annotated benchmark, the baseline vulnerabilities we uncovered, and our initial prototype solver."

---

### Slide 2: The Two Critical Blind Spots in Scientific Search
* **Presenter:** Shravan Krishna Vijay
* **Time:** ~1:30 min
* **Key Talking Points:**
  * "Why is this problem hard? Current search engines and AI models (like SBERT or ChatGPT) evaluate literature using surface vocabulary or direct text matching. This leads to two major blind spots:"
  * **Blind Spot 1 (Vocabulary Siloing):** "An epidemiology paper modeling viral transmission using SEIR networks shares the exact same mathematical graph dynamics as a cybersecurity paper modeling worm malware propagation across switch nodes. But because their vocabularies are completely disjoint, dense embedding models score them as completely unrelated. They achieve 0% recall on structural analogies."
  * **Blind Spot 2 (Contextual Conflation):** "Paper A in human clinical trials finds that a drug decreases blood pressure. Paper B in mouse toxicology finds the drug triggers reflex sympathetic shock under toxic overdose. Standard NLI models see 'decreases' vs. 'increases' and flag this as a direct contradiction! In reality, both papers are right—they just tested different material contexts (species and dosage). This produces a 40%+ false contradiction rate in existing tools."

---

### Slide 3: Our Proposed 5-Way Scientific Relation Taxonomy
* **Presenter:** Subhash Shivaprasad
* **Time:** ~1:00 min
* **Key Talking Points:**
  * "To solve this, we cannot rely on simple binary 'agree vs. disagree'. We formulate a formal 5-way relation taxonomy:"
  * **1. Agreement:** Congruent causal direction under compatible physical context.
  * **2. Cross-Domain Analogy:** Isomorphic causal mechanism structure expressed in different disciplinary vocabularies.
  * **3. Contradiction Candidate:** Opposing causal polarity under *verified identical material scope*. The word 'Candidate' is deliberate—we compare reported claims without pretending an AI model knows absolute causal truth.
  * **4. Contextual Heterogeneity:** Opposing results explained by divergent species, dose, environment, or time horizon.
  * **5. Insufficient Comparability:** Incompatible interventions or missing evidence.

---

### Slide 4: Milestone 1 — 2,054 Real Scientific Documents Ingested
* **Presenter:** Shravan Krishna Vijay
* **Time:** ~1:15 min
* **Key Talking Points:**
  * "Our first major milestone was building a completely real, public scientific corpus."
  * "We implemented automated API ingest engines querying NCBI PubMed E-Utilities and the live arXiv API across four core domains: Biomedical (500 papers), Computer Security (496 papers), Networking (480 papers), and Control Systems (478 papers)."
  * "We enforced all proposal quality filters: minimum 50-word abstracts, strict English ASCII validation, zero retractions, and zero duplicates via MD5 title hashing. Every paper retains its real PMID, arXiv ID, DOI, and authors."

---

### Slide 5: Milestone 2 — Four-Stratum Ground-Truth Benchmark
* **Presenter:** Vedang Vasant Avaghade
* **Time:** ~1:30 min
* **Key Talking Points:**
  * "To rigorously test our system, we designed a balanced 240-pair benchmark divided into four distinct strata of 60 pairs each:"
  * "Stratum 1 provides controlled mutations (vocabulary shifts, sign flips, context shifts, noise) giving objective labels by construction."
  * "Stratum 2 tests cross-domain structural analogies across disjoint disciplines."
  * "Stratum 3 tests natural literature pairs from PubMed and arXiv."
  * "Stratum 4 tests adversarial hard negatives like quoted-not-endorsed claims and simulation vs. observation."
  * **Inter-Annotator Agreement:** "Natural pairs were independently annotated by three reviewers using our standardized schema. We achieved a **Krippendorff's alpha of 0.767**, establishing high pre-adjudication inter-annotator reliability."

---

### Slide 6: Milestone 3 — Multi-Mechanism Extraction with Text Provenance
* **Presenter:** Subhash Shivaprasad
* **Time:** ~1:15 min
* **Key Talking Points:**
  * "Our third milestone was building the extraction engine based on Equation (1): representing every claim as a 10-tuple containing context, population, intervention, mediators, outcome, time, polarity, epistemic status, uncertainty, and character provenance."
  * "We ran batch extraction across the entire 2,054 real documents, extracting **16,638 mechanisms**, **11,945 graph nodes**, and **35,057 directed edges** at a throughput of 965 documents per second."
  * "Most importantly, every extracted mechanism maintains 100% character span provenance back to the original abstract text."

---

### Slide 7: Milestone 4 — Baseline Evaluations Exposing the Core Flaw
* **Presenter:** Shravan Krishna Vijay / Tarun Kumar Bantupalli
* **Time:** ~1:30 min
* **Key Talking Points:**
  * "To understand where current state-of-the-art tools stand, we implemented and evaluated five baseline systems:"
  * "Classical baselines: Okapi BM25 and TF-IDF with LDA topic modeling."
  * "Modern deep-learning baselines: Sentence-BERT, Contriever, and RoBERTa-large-MNLI cross-encoder."
  * **The Critical Empirical Finding:**
    * "First, Sentence-BERT, Contriever, and BM25 achieve **0.0% analogy recall**—completely blind to cross-domain analogies due to vocabulary mismatch."
    * "Second, all baseline models suffer a **38% to 44% false-contradiction error rate** on context-shifted pairs. They flag papers as contradictions simply because words like 'decreases' and 'increases' appear, completely ignoring that the species or dosage differed."
  * "This proves empirically that a context-gated graph alignment method is strictly necessary."

---

### Slide 8: Proposed Solution — CCST Mathematical Formulation
* **Presenter:** Sri Harshith Goli
* **Time:** ~1:30 min
* **Key Talking Points:**
  * "To solve these flaws, we formulate Equation (2): Context-Conditioned Signed Partial Optimal Transport."
  * "It optimizes a transport plan $T$ matching mechanisms between graphs $G_i$ and $G_j$ based on three core mathematical innovations:"
  1. **Partial Transport ($\lambda_u U(T)$):** "Scientific papers contain peripheral mechanisms. Standard optimal transport forces all nodes to match; partial transport allows irrelevant mechanisms to remain unmatched without corrupting the core alignment."
  2. **Signed vs. Unsigned Dual-View ($\Delta_{\text{sign}}$):** "We evaluate graphs under a signed view (sensitive to $+1$ vs $-1$ polarity) and an unsigned view (evaluating purely structural topology). The difference, $\Delta_{\text{sign}}$, isolates true directional opposition from structural isomorphism."
  3. **Context Gating ($\Phi_{\text{gate}}$):** "Evaluates the compatibility of species, dosage, environment, and time horizon. If contexts diverge, $\Phi_{\text{gate}}$ penalizes declaring a low-cost contradiction."

---

### Slide 9: Prototype Validation & Proof-of-Concept Success
* **Presenter:** Sri Harshith Goli / Vedang Vasant Avaghade
* **Time:** ~1:30 min
* **Key Talking Points:**
  * "We have implemented an initial prototype solver using the Frank-Wolfe algorithm and verified its core theorems on representative benchmark pairs:"
  * **Result 1:** "On context-shifted pairs, context gating $\Phi_{\text{gate}}$ successfully prevented 100% of false contradictions (0/50 misclassified), dropping the error rate to **0.00%** compared to 40%+ on baselines!"
  * **Result 2:** "Our unsigned graph view successfully uncovered structural analogies between epidemic SEIR models and network worm malware, achieving **100.0% analogy recall**."
  * **Deliverable:** "The prototype successfully outputs automated **Disagreement Certificates**, providing an auditable summary showing the exact character spans and physical scope differences."

---

### Slide 10: Scalability Architecture — Tri-Index Candidate Retrieval
* **Presenter:** Tarun Kumar Bantupalli
* **Time:** ~1:15 min
* **Key Talking Points:**
  * "A major practical question is scalability. Exact pairwise graph alignment across 2,054 papers requires nearly **2 million comparisons** ($O(N^2)$)."
  * "To make this feasible, we designed a hybrid candidate retrieval engine coupling three sparse indexes:"
  1. $\mathcal{I}_{\text{sem}}$: FAISS dense semantic index retrieving top-15 topical neighbors.
  2. $\mathcal{I}_{\text{graph}}$: 2-iteration Weisfeiler-Lehman topological graph hash retrieving top-15 structural candidates across disjoint vocabularies.
  3. $\mathcal{I}_{\text{ctx}}$: Context compatibility bitvector index retrieving top-10 scope-compatible papers.
  * "By taking the union of these indexes (~40 candidate papers per query), we reduce the search space from 2 million down to 80,000 pairs—achieving a **25.0x speedup** with an index build time of just 0.23 seconds."

---

### Slide 11: Team Governance & Division of Work
* **Presenter:** Vedang Vasant Avaghade
* **Time:** ~1:00 min
* **Key Talking Points:**
  * "Our progress has been driven by clear primary ownership across the team, exactly matching Table III from our proposal:"
  * **Shravan:** Led real corpus acquisition via PubMed and arXiv APIs, and implemented classical IR baselines.
  * **Sri Harshith:** Formulated the CCST optimization problem and built the Frank-Wolfe partial optimal transport solver.
  * **Subhash:** Built the multi-mechanism extraction pipeline and character-span provenance validator.
  * **Tarun:** Designed the tri-index candidate retrieval engine and performed scaling measurements.
  * **Vedang:** Coordinated the 4-stratum benchmark, calculated inter-annotator agreement ($\alpha = 0.767$), and designed the certificate evaluation.
  * "All members participated in independent benchmark labeling, code review, and report preparation."

---

### Slide 12: Roadmap & Timeline to Final December Submission
* **Presenter:** Vedang Vasant Avaghade / All Team Members
* **Time:** ~1:00 min
* **Key Talking Points:**
  * "To conclude, we have established strong data foundations, validated the core flaw of existing baselines, and proven our theoretical formulation on a working prototype."
  * "Our roadmap for the remainder of the semester leading into December:"
    * **Early November:** Systematic hyperparameter tuning of $\alpha, \lambda_u, \lambda_g$ and probability calibration on the development split.
    * **Mid November:** Full held-out test evaluation with 1,000 bootstrap resamples and comprehensive ablation studies.
    * **Late November:** Full-corpus scaling stress tests on the ASU SOL A100 GPU cluster and refinement of the certificate UI.
    * **December:** Final technical report submission, open-source code release, and conference paper manuscript.
  * "Thank you, and we welcome any questions!"

---

## Anticipated Q&A Cheat Sheet for the Team

| Likely Question from Professor / Reviewer | Recommended Answer |
| :--- | :--- |
| **"Why not just prompt GPT-4 or Claude to find contradictions?"** | *"LLM prompting operates as a black box: it lacks mathematical alignment guarantees, often fails on cross-domain analogies unless explicitly prompted with the analogy target, and cannot provide verifiable character-level provenance spans. CCST solves a convex optimal transport problem that produces an interpretable transportation matrix $T^*$ and an auditable certificate."* |
| **"How did you avoid data leakage in the benchmark?"** | *"We enforced a strict 60/20/20 train/dev/test split grouped by transformation family. The held-out test set remains completely frozen, and all threshold selection and calibration take place exclusively on the development split."* |
| **"Why Optimal Transport instead of Graph Neural Networks (GNNs)?"** | *"GNNs require thousands of labeled paired graphs to train node embeddings and struggle to generalize to unseen scientific domains. Optimal Transport provides exact zero-shot / few-shot graph alignment with explicit partial matching and formal mathematical distance bounds."* |
| **"How do you know the extraction didn't hallucinate context?"** | *"Our extractor enforces strict character span grounding: every single extracted mechanism must match its verbatim characters in the source abstract, verified with 100% character precision in our unit test suite."* |
