# CCST Project DEMO Presentation: 12-Minute Speaker Script & Notes

**Course:** Software & Web Mining (SWM) — Fall 2026, Arizona State University  
**Project:** Group 16, Project 16 — Context-Conditioned Causal Alignment for Scientific Literature (CCST)  
**Submission File:** `Group16-Project16-FA26-Group-DEMO-Presentation.pdf`  
**GitHub Repository:** [https://github.com/Vedang1308/Causal_Alignment_for_Scientific_Literature](https://github.com/Vedang1308/Causal_Alignment_for_Scientific_Literature)  
**Team Members & Presenters:**
1. **Shravan Krishna Vijay** (Slides 1–2: Introduction & Problem Definition)
2. **Sri Harshith Goli** (Slides 3–4: System Architecture & Mathematical Formulation)
3. **Subhash Shivaprasad** (Slides 5–6: Datasets & Baseline Evaluations)
4. **Tarun Kumar Bantupalli** (Slides 7 & 10: Component Ablations & Project Plan)
5. **Vedang Vasant Avaghade** (Slides 8, 9, 11: Live Demo Walkthrough, UI Visualization, GitHub Deliverables & Q&A)

---

## Timing Overview (Total: 12 Minutes)

| Slide # | Topic | Primary Speaker | Target Duration | Running Time |
| :---: | :--- | :--- | :---: | :---: |
| **Slide 1** | Cover Page: Title, Team, Scope | Shravan Krishna Vijay | 1.0 min | 00:00 – 01:00 |
| **Slide 2** | Problem Definition [1 Pt] | Shravan Krishna Vijay | 1.5 min | 01:00 – 02:30 |
| **Slide 3** | System Architecture: 6-Phase Pipeline [1 Pt] | Sri Harshith Goli | 1.0 min | 02:30 – 03:30 |
| **Slide 4** | Algorithm Formulation: Signed OT & Context Gating [1 Pt] | Sri Harshith Goli | 1.5 min | 03:30 – 05:00 |
| **Slide 5** | Datasets & Preprocessing: 2,054 Papers, Benchmark [1 Pt] | Subhash Shivaprasad | 1.0 min | 05:00 – 06:00 |
| **Slide 6** | Evaluations: Benchmark vs Baselines [2.5 Pts] | Subhash Shivaprasad | 1.5 min | 06:00 – 07:30 |
| **Slide 7** | Evaluations: Ablations & Certificates [2.5 Pts] | Tarun Kumar Bantupalli | 1.0 min | 07:30 – 08:30 |
| **Slide 8** | UI & Visualization: Live Demo & Graph Alignment [2.5 Pts] | Vedang Vasant Avaghade | 1.5 min | 08:30 – 10:00 |
| **Slide 9** | UI & Visualization: Context Diff & Certificate Viewer [2.5 Pts] | Vedang Vasant Avaghade | 1.0 min | 10:00 – 11:00 |
| **Slide 10** | Project Plan: Tasks, Milestones, Division of Work [1 Pt] | Tarun Kumar Bantupalli | 0.5 min | 11:00 – 11:30 |
| **Slide 11** | Software Deliverables, GitHub Repository & Conclusion | Vedang Vasant Avaghade | 0.5 min | 11:30 – 12:00 |

---

## Slide-by-Slide Speaking Script

### Slide 1: Cover Page (Shravan Krishna Vijay — 1.0 min)
> *"Good morning professor and fellow classmates. We are Group 16, presenting our final software demonstration for Project 16: **Context-Conditioned Causal Alignment for Scientific Literature (CCST)**.*
> 
> *Our team members are Sri Harshith Goli, Subhash Shivaprasad, Tarun Kumar Bantupalli, Vedang Vasant Avaghade, and myself, Shravan Krishna Vijay.*
> 
> *In this 12-minute presentation, we will walk you through the core scientific discovery problem, our end-to-end mathematical formulation, empirical evaluations across 2,054 published scientific papers, a live demonstration of our interactive software application, and our GitHub repository structure."*

---

### Slide 2: Problem Definition [1 Point] (Shravan Krishna Vijay — 1.5 min)
> *"Why does existing scientific retrieval fail? Today, research literature is expanding exponentially, but tools like Sentence-BERT, BM25, and LLM judges suffer from two catastrophic flaws:*
> 
> *First is **Cross-Domain Analogy Blindness**. When two distinct disciplines study identical causal mechanisms—such as Byzantine fault quorum consensus in distributed computer networks versus epidemic contact herd immunity in virology—they use entirely disjoint vocabularies. Because traditional search relies on lexical or semantic embedding overlap, they produce near-zero similarity and 0% analogy recall.*
> 
> *Second, and even more dangerously, is the **False Contradiction Trap**. Frequently, two biomedical studies appear to disagree on an outcome—for instance, one showing that metformin inhibits glucose production and another showing no effect. However, they were conducted in entirely different experimental contexts: isolated in-vitro mouse cells at supraphysiological doses versus in-vivo human clinical trials.*
> 
> *Our empirical experiments reveal that standard models misclassify context-shifted findings as causal contradictions between **38% and 44% of the time**.*
> 
> *To fix this, we created **CCST**: a framework that extracts reported mechanism graphs, conditions alignment on material scope conditions, and emits human-auditable Disagreement Certificates."*

---

### Slide 3: System Architecture & Algorithms (Part 1) (Sri Harshith Goli — 1.0 min)
> *"Thank you, Shravan. Moving to our system architecture, CCST is built as an end-to-end, six-phase modular software pipeline deployed on the ASU SOL Supercomputer with an NVIDIA A100 GPU.*
> 
> *In **Phases 1 and 2**, we ingest scientific literature directly from live public APIs—PubMed and arXiv—and maintain our 240-pair evaluation benchmark.*
> 
> *In **Phase 3**, our provenance-linked extractor parses abstracts into structured causal mechanism graphs, capturing entities, causal directions, and text offsets.*
> 
> *In **Phase 4**, we solve candidate retrieval using a Sparse Tri-Index combining BM25, dense neural embeddings, and causal role filters, which delivers a **25.0× search speedup** before feeding candidate pairs into our signed partial optimal transport solver.*
> 
> *Finally, in **Phases 5 and 6**, our calibrated 5-way classifier evaluates relation probabilities with selective abstention, generating human-auditable Disagreement Certificates in under 6 milliseconds per pair on the A100 GPU."*

---

### Slide 4: System Architecture & Algorithms (Part 2) (Sri Harshith Goli — 1.5 min)
> *"Let us examine the mathematical formulation of CCST.*
> 
> *We formulate causal alignment as a **Context-Conditioned Signed Partial Optimal Transport** problem. Unlike standard optimal transport which forces total mass conservation, our objective allows partial transport slack \(\lambda_u\), allowing mechanism graphs of unequal sizes to align without distortion.*
> 
> *The key innovation is our **Dual-View Graph Representation**:*
> *1. The **Signed View** penalizes causal sign inversions to detect true scientific disagreements.*
> *2. The **Unsigned View** relaxes polarities and domain vocabularies to identify shared causal topologies across disparate disciplines.*
> 
> *Crucially, we introduce **Material Context Gating (\(\Lambda_g\))**. We evaluate four material scope dimensions: Species or System, In-Vitro versus In-Vivo Setting, Dosage or Scale, and Time Horizon. When experimental contexts conflict or are unspecified, the compatibility score collapses to zero, which exponentially inflates the contradiction transport cost.*
> 
> *This provides a rigorous mathematical guarantee: false contradictions are blocked, and findings are correctly routed to Contextual Heterogeneity."*

---

### Slide 5: Datasets & Preprocessing [1 Point] (Subhash Shivaprasad — 1.0 min)
> *"Thank you, Sri Harshith. For our experimental evaluation, we insisted on **100% real published scientific data** with zero mocks or synthetic shortcuts.*
> 
> *We scraped **2,054 full papers**: 1,027 biomedical and pharmacological articles from NCBI PubMed E-Utilities, and 1,027 computer systems and control papers from the arXiv API across cs.CR, cs.NI, and cs.SY.*
> 
> *Our automated preprocessing pipeline enforces strict quality gates: abstract length at least 50 words, 100% English language detection, zero retracted papers, and MD5-hash deduplication.*
> 
> *From these papers, our pipeline extracted **16,638 mechanisms**, **11,945 graph nodes**, and **35,057 causal edges** with 100% character-level provenance spans verified against the raw text.*
> 
> *To evaluate accuracy, we constructed a balanced **240-pair 4-stratum benchmark** covering Replication, Cross-Domain Analogies, Context Shifts, and True Contradictions. Triple independent expert annotation achieved a Krippendorff's alpha of **0.767**, demonstrating high inter-annotator reliability."*

---

### Slide 6: Evaluations & Baseline Comparisons [2.5 Points] (Subhash Shivaprasad — 1.5 min)
> *"Now let's examine our empirical findings. We evaluated CCST against classical and modern retrieval baselines using a stratified 60/20/20 train/dev/test split with 1,000 bootstrap resamples for 95% confidence intervals.*
> 
> *As shown in Table I:*
> *1. **Lexical and Topic Models (BM25, TF-IDF + LDA)** achieve only 14.8% to 18.2% Macro F1 and fail completely on cross-domain analogies (0% to 5% recall).*
> *2. **Dense Neural Baselines (Sentence-BERT / SciBERT, Contriever, Direct LLM Judge)** fail severely on the false contradiction trap: they hallucinate causal contradictions between **40% and 44% of the time** on context shifts.*
> *3. **CCST (Proposed)** achieves a state-of-the-art Macro F1 of **43.9%** (95% CI [34.3, 51.8]), achieves **100.0% analogy recall**, and most importantly, achieves a **0.00% False Contradiction Rate**.*
> 
> *Paired McNemar tests confirm this improvement is statistically significant at \(p < 0.001\), running with an average latency of only **5.77 ms** on the NVIDIA A100 GPU."*

---

### Slide 7: Component Ablations & Certificates [2.5 Points] (Tarun Kumar Bantupalli — 1.0 min)
> *"Thank you, Subhash. To understand what makes CCST work, we conducted a rigorous component ablation study across all 240 benchmark pairs, shown in Table II.*
> 
> *When we ablate **Context Gating** by setting \(\lambda_g = 0\), the False Contradiction Rate explodes from **0.0% to 100.0%** on context-shifted pairs! This proves that context gating is strictly necessary to prevent false disagreements.*
> 
> *When we remove the **Unsigned Graph View**, Analogy Recall collapses from **100.0% down to 0.0%**, confirming that dual-view optimization is essential for discovering cross-disciplinary analogies.*
> 
> *Finally, regarding **Disagreement Certificates**: 100% of our predicted contradictions are accompanied by human-auditable certificates explaining the exact minimal set of sign-inverted causal edges, with 100% textual provenance grounding back to the original abstracts."*

---

### Slide 8: UI Interface & Live System Demo (Vedang Vasant Avaghade — 1.5 min)
> *(Presenter opens `code/ccst_interactive_demo.html` in Chrome or demonstrates the slide visuals)*
> 
> *"Now, let us show you our live software implementation in action.*
> 
> *Here on the screen is our **CCST Interactive Discovery Dashboard**. It is a zero-dependency HTML5 application located in our repository under `/code/ccst_interactive_demo.html`.*
> 
> *On the left, we can load test cases from our benchmark:*
> *1. Here is **Case 1: True Contradiction** in Cardiovascular Medicine. Both papers study low-dose aspirin in human patients. As you can see on the center SVG canvas, Document A's mechanism graph shows aspirin inhibiting platelet aggregation (red edge), while Document B shows paradoxical platelet activation in high-shear flow (green edge). The dashed lines show the partial optimal transport coupling \(T_{uu'}\), and the system flags a verified sign inversion.*
> 
> *2. Now watch what happens when we switch to **Case 3: Contextual Heterogeneity** on Metformin. Here, Document A is an in-vitro murine hepatocyte study at 10 millimolar concentration, while Document B is an in-vivo human clinical trial at 500 milligrams.*
> 
> *Notice our live slider for Context Gating Weight \(\lambda_g\). If we set \(\lambda_g = 1.0\), CCST correctly routes this to **Contextual Heterogeneity (88.2%)**. But if I drag the slider down to \(\lambda_g = 0\), simulating existing baselines, watch the prediction banner immediately turn red: it falsely flags a contradiction with 91.5% confidence! This interactively demonstrates our core contribution live."*

---

### Slide 9: UI Interface: Context Diff & Certificate Inspector (Vedang Vasant Avaghade — 1.0 min)
> *"Looking at the right side of our dashboard, we see the **Material Scope Diff Matrix** and the **Minimal Disagreement Certificate Inspector**.*
> 
> *The Material Scope Diff Matrix clearly itemizes each dimension:*
> *Species: Human vs Mouse &rarr; Red Material Conflict badge.*
> *Setting: In-Vivo Clinical vs In-Vitro Cell Culture &rarr; Red Material Conflict.*
> *Dosage: 500 mg vs 10 mM &rarr; Dose Shift.*
> 
> *Below that is our calibrated 5-way probability distribution and the **Minimal Disagreement Certificate**. The certificate provides the unique certificate ID, the exact conflicting causal links, and verbatim character-level provenance spans linking back to the source text.*
> 
> *Scientists can export this audit report directly to JSON, Markdown, or LaTeX with a single click, providing complete interpretability for regulatory and peer-review auditing."*

---

### Slide 10: Project Plan: Tasks, Milestones & Division of Work [1 Point] (Tarun Kumar Bantupalli — 0.5 min)
> *"Thank you, Vedang. Reviewing our project plan:*
> 
> *Over the semester, our team executed four major development cycles:*
> *In September, we completed schema definitions, API crawlers, and corpus collection.*
> *In October, we completed causal graph extraction, benchmark curation, and baseline models.*
> *In early November, we completed the CCST solver, calibrated classifier, and ASU SOL A100 GPU benchmarking.*
> *Over the past two weeks, we packaged our interactive demo application, completed full statistical verification, and prepared our software submission.*
> 
> *All five team members contributed equally across theoretical formulation, API ingestion, causal extraction, solver optimization, and system deployment."*

---

### Slide 11: Software Deliverables & GitHub Repository (Vedang Vasant Avaghade — 0.5 min)
> *"To conclude, our software is fully packaged and publicly hosted on GitHub at:*
> `https://github.com/Vedang1308/Causal_Alignment_for_Scientific_Literature`
> 
> *Per the assignment guidelines, the repository is cleanly divided into three primary sub-directories:*
> *1. **`/code`** containing all 6 modular pipeline phases, our one-click CLI demo runner (`run_demo.py`), and our interactive web dashboard.*
> *2. **`/data`** containing the 2,054 real scientific papers, document graphs, embeddings, and the 240-pair benchmark.*
> *3. **`/evaluations`** containing reproduction scripts, LaTeX result tables, and certificate audit artifacts.*
> 
> *All 32 PyTest unit and integration tests pass cleanly in 0.29s. Thank you very much, and we are now happy to take any questions!"*

---

## Anticipated Faculty Q&A Preparation

### Q1: "Why did you choose Optimal Transport instead of a Graph Neural Network (GNN)?"
**Answer (Shravan / Tarun):**
> *"GNNs require massive labeled pairwise graph datasets for supervised training, which do not exist for scientific contradiction detection with scope conditions. Moreover, GNN embeddings compress entire graphs into pooled vectors, losing character-level provenance. Optimal transport solves an exact convex alignment problem with zero training parameters, provides transparent bipartite coupling weights \(T_{uu'}\), and directly emits the minimal disagreement certificate with provable mass conservation bounds."*

### Q2: "How does CCST handle missing or unspecified context in papers?"
**Answer (Sri Harshith / Vedang):**
> *"Standard systems naively assume missing context implies compatibility. CCST enforces strict epistemic caution: missing context is classified as `UNKNOWN`. In our context gating formula, if a critical dimension is unspecified, the compatibility score defaults to 0, which activates the exponential barrier and suppresses contradiction classification, routing the pair to `Insufficient Comparability` or `Contextual Heterogeneity` rather than hallucinating a false contradiction."*

### Q3: "What makes your extraction pipeline provenance-grounded?"
**Answer (Subhash):**
> *"Every entity node and causal relationship extracted by our pipeline stores exact character offset start and end indices `[start_char, end_char]` referencing the source abstract text. Our verification suite includes automated integrity checks ensuring that slicing the raw text with these offsets matches the entity string with 100% precision."*

### Q4: "How does CCST scale to thousands of papers?"
**Answer (Tarun / Vedang):**
> *"Running pairwise optimal transport naively over \(N\) papers requires \(O(N^2)\) comparisons. To scale, we implemented a Sparse Tri-Index retriever in Phase 4 that combines BM25 lexical tokens, dense Sentence-BERT embeddings, and causal role filters. This prunes 96% of irrelevant comparisons—yielding a **25.0× search speedup**—and reduces pairwise OT alignment to only **5.77 ms** per candidate pair on the A100 GPU."*
