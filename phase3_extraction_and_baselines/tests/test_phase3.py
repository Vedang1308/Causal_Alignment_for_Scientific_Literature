"""
Unit Tests for Phase 3: Multi-Mechanism Extraction & Retrieval Baselines.
Verifies:
1. Multi-mechanism extraction accuracy and strict character provenance spans.
2. Extracted document graph corpus integrity (data/processed/document_graphs.jsonl).
3. Classical BM25 lexical retrieval and ranking.
4. Classical TF-IDF and LDA topic modeling with K-Means clustering.
5. Modern Sentence-BERT / Contriever dense embedding retrieval.
6. Direct LLM judge and false-contradiction failure mode replication.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1_foundation.schemas import (
    DocumentRecord,
    DocumentGraph,
    Polarity,
    EpistemicStatus,
    RelationType,
    BenchmarkPair,
)
from phase3_extraction_and_baselines.extractor import MultiMechanismExtractor
from phase3_extraction_and_baselines.classical_baselines import BM25Retriever, TfidfLdaBaseline
from phase3_extraction_and_baselines.modern_baselines import SBERTRetriever, DirectLLMJudge


def test_extractor_provenance_and_polarity():
    """Verify that extracted claims retain exact character-level provenance spans."""
    text = (
        "In a cohort of hypertensive adults, Amlodipine 5mg oral daily reduced mean systolic blood pressure by 12 mmHg. "
        "Prior literature claimed that calcium channel blockers aggravated peripheral edema."
    )
    extractor = MultiMechanismExtractor()
    claims = extractor.extract_mechanisms_from_text(text, doc_id="TEST_DOC", domain="biomedical")

    assert len(claims) >= 2
    # First claim should be negative polarity (reduced blood pressure)
    assert claims[0].polarity == Polarity.NEGATIVE
    assert claims[0].epistemic_status == EpistemicStatus.OBSERVED
    # Strict provenance check: substring must match exact character offsets
    p0 = claims[0].provenance
    assert p0 is not None
    assert text[p0.start_char : p0.end_char] == p0.text

    # Second claim should be quoted-not-endorsed
    assert claims[1].epistemic_status == EpistemicStatus.QUOTED_NOT_ENDORSED
    p1 = claims[1].provenance
    assert text[p1.start_char : p1.end_char] == p1.text


def test_extracted_graphs_corpus_file():
    """Verify compiled DocumentGraph instances in data/processed/document_graphs.jsonl."""
    graphs_file = PROJECT_ROOT / "data" / "processed" / "document_graphs.jsonl"
    assert graphs_file.exists(), f"Graphs file not found at {graphs_file}"

    count = 0
    with open(graphs_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 50:  # Validate first 50 graphs
                g = DocumentGraph.model_validate_json(line.strip())
                assert len(g.nodes) > 0
                assert len(g.edges) > 0
                assert pytest.approx(sum(g.get_normalized_node_masses()), rel=1e-4) == 1.0
            count += 1
    assert count == 2000, f"Expected 2000 document graphs, got {count}"


def test_bm25_retrieval_and_metrics():
    """Verify BM25 lexical indexing and retrieval metrics."""
    doc_a = DocumentRecord(
        doc_id="D1",
        title="Atorvastatin efficacy in coronary artery disease",
        abstract="Atorvastatin significantly reduced LDL-C levels in coronary patients.",
        domain="biomedical",
    )
    doc_b = DocumentRecord(
        doc_id="D2",
        title="Rosuvastatin in hypercholesterolemia",
        abstract="Rosuvastatin decreased serum cholesterol in hypercholesterolemic cohorts.",
        domain="biomedical",
    )
    doc_c = DocumentRecord(
        doc_id="D3",
        title="Topological SDN worm propagation",
        abstract="Scanning malware spreads through software-defined network clusters.",
        domain="computer_security",
    )

    retriever = BM25Retriever().fit([doc_a, doc_b, doc_c])
    res = retriever.query("Atorvastatin LDL-C coronary", top_k=2)

    assert len(res) > 0
    assert res[0][0] == "D1"  # Highest match should be D1
    assert res[0][1] > 0.0


def test_tfidf_lda_baseline():
    """Verify TF-IDF, LDA topic model, and K-Means clustering."""
    doc_a = DocumentRecord(doc_id="D1", title="Atorvastatin therapy", abstract="Reduces cholesterol.", domain="biomedical")
    doc_b = DocumentRecord(doc_id="D2", title="Simvastatin therapy", abstract="Lowers LDL.", domain="biomedical")
    doc_c = DocumentRecord(doc_id="D3", title="Firewall rules", abstract="Blocks DDoS packets.", domain="computer_security")

    baseline = TfidfLdaBaseline(n_topics=2, n_clusters=2).fit([doc_a, doc_b, doc_c])
    assert baseline.tfidf_matrix.shape[0] == 3
    assert baseline.topic_distributions.shape == (3, 2)
    assert len(baseline.cluster_labels) == 3

    res = baseline.query_by_tfidf("cholesterol", top_k=2)
    assert len(res) > 0


def test_sbert_retriever_ranking():
    """Verify dense SBERT retrieval and cosine similarity ranking."""
    doc_a = DocumentRecord(doc_id="D1", title="Cardiology paper", abstract="Heart disease therapy.", domain="biomedical")
    doc_b = DocumentRecord(doc_id="D2", title="Malware paper", abstract="Network virus infection.", domain="computer_security")

    retriever = SBERTRetriever(use_mock_fallback=True).fit([doc_a, doc_b])
    res = retriever.query("heart disease and cardiology", top_k=2)
    assert len(res) == 2
    assert res[0][0] == "D1"


def test_direct_llm_judge_baseline():
    """Verify direct LLM judge reproduces baseline false-contradiction failure on context shifts."""
    doc_human = DocumentRecord(
        doc_id="D_H",
        title="Drug X in clinical trial",
        abstract="Drug X reduced blood pressure in adult patients.",
        domain="biomedical",
    )
    doc_mouse = DocumentRecord(
        doc_id="D_M",
        title="Drug X in toxic rat study",
        abstract="Drug X increased blood pressure in mice under toxic overdose.",
        domain="biomedical",
    )
    pair = BenchmarkPair(
        pair_id="TEST_PAIR",
        stratum=pytest.importorskip("phase1_foundation.schemas").StratumType.STRATUM_3_NATURAL,
        subtype=pytest.importorskip("phase1_foundation.schemas").StratumSubtype.NATURAL_CONTEXTUAL_HETEROGENEITY,
        doc_a=doc_human,
        doc_b=doc_mouse,
        ground_truth_relation=RelationType.CONTEXTUAL_HETEROGENEITY,
    )

    judge = DirectLLMJudge()
    pred = judge.predict_pair(pair)
    # The baseline flaw: naive textual comparison predicts CONTRADICTION_CANDIDATE because reduced vs increased
    assert pred == RelationType.CONTRADICTION_CANDIDATE

