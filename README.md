# Context-Conditioned Causal Alignment for Scientific Literature (CCST)

> **Separating Cross-Domain Analogies, True Contradictions, and Contextual Heterogeneity**  
> *SWM Group Project — Arizona State University*  
> **Authors:** Shravan Krishna Vijay, Sri Harshith Goli, Subhash Shivaprasad, Tarun Kumar Bantupalli, Vedang Vasant Avaghade

---

## 1. Project Overview

Scientific papers often describe equivalent causal mechanisms using disjoint domain vocabularies, or appear to contradict each other because of differing experimental contexts (such as species, dosage, or time horizons). Standard vector-embedding models prioritize topical similarity and often misclassify context-shifted results as contradictions while missing cross-domain analogies.

**CCST (Context-Conditioned Signed Partial Transport)** addresses this by:
1. Extracting structured, provenance-linked causal mechanism graphs:
   $$q_{im} = (C, P, A, M, Y, \tau, \delta, \eta, u, p)_{im}$$
2. Computing context-gated partial optimal transport alignments that penalize false contradictions when contexts are incompatible or unknown.
3. Categorizing claims into a calibrated **5-way relation taxonomy**:
   * **Agreement**
   * **Cross-Domain Analogy**
   * **Contradiction Candidate**
   * **Contextual Heterogeneity**
   * **Insufficient Comparability (Abstention)**
4. Emitting a human-auditable, evidence-grounded **Minimal Disagreement Certificate**.

---

## 2. Six-Phase Architecture

```
├── phase1_foundation/                  # Core Pydantic schemas, primitives, graphs, and verification
├── phase2_data_and_benchmark/          # 2,000-paper corpus pipeline & 240-pair 4-stratum benchmark
├── phase3_extraction_and_baselines/    # Multi-mechanism LLM extractor & classical/modern baselines
├── phase4_ccst_solver_and_indexing/    # Three sparse ANN indexes & partial optimal transport solver
├── phase5_classification_and_certificates/ # Calibrated 5-way classifier, abstention & certificate generator
└── phase6_evaluation_and_experiments/  # Held-out evaluation, false-contradiction stress tests & ablations
```

---

## 3. Quickstart on ASU SOL Supercomputer (NVIDIA A100 GPU)

### Step 1: Clone or Pull the Latest Code
```bash
git clone https://github.com/Vedang1308/Causal_Alignment_for_Scientific_Literature.git
cd Causal_Alignment_for_Scientific_Literature
```
*(If already cloned, simply run `git pull origin main`)*

### Step 2: Request an Interactive GPU Session (or Direct Access)
If you are on a login node, start an interactive session with an A100 GPU:
```bash
interactive -p general -q public -t 4:00:00 --gres=gpu:a100:1
```
*(If you already have direct shell access to an A100 node, proceed to Step 3)*

### Step 3: Run the Automated SOL Setup Script
```bash
bash setup_sol.sh
```
This script will:
* Check NVIDIA A100 GPU status via `nvidia-smi`
* Load the latest CUDA (12.1) and Python/Mamba modules on SOL
* Create and activate the `ccst_env` environment
* Install all required dependencies (PyTorch with CUDA, POT, Faiss, Pydantic, NetworkX)
* Execute the Phase 1 Foundation verification suite

### Step 4: Verify Phase 1
You can run the Phase 1 verification manually anytime:
```bash
python phase1_foundation/run_phase1_verification.py
```
Or run the unit tests:
```bash
pytest phase1_foundation/tests/ -v
```

