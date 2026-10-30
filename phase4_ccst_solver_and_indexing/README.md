# Phase 4: CCST Solver & Sparse ANN Indexing

## Objectives
1. **Three Sparse Approximate Nearest Neighbor (ANN) Indexes:**
   - **Semantic Index ($k_s$):** Fast SBERT / Contriever cosine similarity search using Faiss.
   - **Unsigned Mechanism Index ($k_m$):** Structural graph similarity ignoring polarity (e.g., Weisfeiler-Lehman / graph hash kernels).
   - **Context Index ($k_c$):** Compatibility filtering across material context dimensions $c_h(i, j) \in [0, 1]$.
   - Pipeline scaling target: empirical $\mathcal{O}(N \log N + N k_{\text{ccst}})$ instead of quadratic $\mathcal{O}(N^2)$.
2. **Context-Conditioned Signed Partial Transport (CCST) Solver:**
   - Optimize partial transport coupling $T \ge 0$ with unmatched mass penalty $U(T) = \|p_i\|_1 + \|p_j\|_1 - 2\mathbf{1}^\top T \mathbf{1}$.
   - Unsigned view to localize shared causal topology (detect cross-domain analogies).
   - Signed view to identify polarity flips ($\delta_e \ne \delta_{e'}$).
   - Context gate $\Phi_{\text{gate}}$ preventing low structural distance from masquerading as a contradiction when conditions are incompatible or unknown.

## Planned Directory Structure
```
phase4_ccst_solver_and_indexing/
├── indexes/
│   ├── semantic_index.py    # Faiss dense embedding index
│   ├── mechanism_index.py   # Unsigned graph structural index
│   └── context_index.py     # Material scope conditions index
├── candidate_retriever.py   # Merges ks, km, kc candidate queries
├── ot_solver.py             # Python Optimal Transport (POT) partial FGW solver
├── cost_matrices.py         # Node cost M and Edge cost L builder
└── context_gate.py          # Penalty function Φ_gate implementation
```
