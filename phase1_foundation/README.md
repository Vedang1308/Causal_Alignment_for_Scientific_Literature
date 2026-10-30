# Phase 1: Foundation & Data Schemas

This directory establishes the formal mathematical foundations, Pydantic data schemas, and validation test suite for **CCST (Context-Conditioned Signed Partial Transport)**.

---

## 1. Mathematical Mapping to Code

| Proposal Notation | Concept | Python Schema / Class | Description |
| :--- | :--- | :--- | :--- |
| $q_{im}$ (Eq. 1) | Reported Mechanism | `MechanismClaim` | Preserves $(C, P, A, M, Y, \tau, \delta, \eta, u, p)_{im}$ |
| $C$ (Eq. 1) | Scope / Context | `MaterialContext` | Explicit tracking of conditions ($h$); missing fields evaluate to `UNKNOWN (-1.0)` |
| $\delta \in \{+1, -1, 0\}$ | Polarity / Sign | `Polarity` | $+1$ (positive/promotes), $-1$ (negative/inhibits), $0$ (null/neutral) |
| $\eta$ (Eq. 1) | Epistemic Status | `EpistemicStatus` | `OBSERVED`, `HYPOTHESIZED`, `SIMULATED`, `QUOTED_NOT_ENDORSED` |
| $u \in [0, 1]$ | Extraction Uncertainty | `float` field | Tracks extractor confidence for threshold-based abstention |
| $p$ (Eq. 1) | Provenance Span | `ProvenanceSpan` | `(start_char, end_char, text)` linking claims directly to source text |
| $G_i$ (Sec. IV-A) | Document Graph | `DocumentGraph` | $G_i = (V_i, E_i, X_i, R_i, Z_i, p_i, \rho_i)$ with role, mass, edges |
| $p_i$ ($\sum p_i = 1$) | Normalized Mass | `get_normalized_node_masses()` | Confidence-weighted mass distribution for CCST transport coupling |
| 5-Way Taxonomy | Relation Labels | `RelationType` | `AGREEMENT`, `CROSS_DOMAIN_ANALOGY`, `CONTRADICTION_CANDIDATE`, `CONTEXTUAL_HETEROGENEITY`, `INSUFFICIENT_COMPARABILITY` |
| Certificate | Explainability | `DisagreementCertificate` | Aligned subgraphs, context diffs, sign-flip edges, text spans, gap fraction |
| 240 Pairs | Benchmark Testbed | `BenchmarkPair` | 4 Strata: Controlled, Analogies, Natural Science, Adversarial Hard Negatives |

---

## 2. Directory Structure

```
phase1_foundation/
├── schemas/
│   ├── __init__.py           # Unified exports
│   ├── primitives.py         # RelationType, Polarity, EpistemicStatus, NodeRole, EdgeType, ContextDimension
│   ├── mechanism.py          # MaterialContext, MechanismClaim (Equation 1)
│   ├── graph.py              # GraphNode, GraphEdge, DocumentGraph (Section IV-A)
│   ├── certificate.py        # DisagreementCertificate, AlignedNodePair, ContextDiff (Section IV-D)
│   └── benchmark.py          # BenchmarkPair, StratumType, AnnotatorVote (Section II)
├── tests/
│   ├── __init__.py
│   └── test_schemas.py       # Unit tests verifying serialization, constraints, and context gating
├── run_phase1_verification.py# Executable end-to-end demonstration and validation script
└── README.md
```

---

## 3. How to Run Phase 1

### Option A: Run the End-to-End Demonstration Script
```bash
python phase1_foundation/run_phase1_verification.py
```

### Option B: Run the Pytest Unit Test Suite
```bash
python -m pytest phase1_foundation/tests/ -v
```

