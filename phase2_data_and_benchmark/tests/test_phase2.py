"""
Unit Tests for Phase 2: Corpus Acquisition & 4-Stratum Benchmark.
Verifies:
1. Corpus size, schema integrity, and proposal exclusion filters.
2. Domain distribution and balance across all 4 disciplines.
3. Four-stratum benchmark balance (60 pairs per stratum = 240 pairs).
4. Multi-annotator vote consistency and Krippendorff's alpha calculation.
"""

import sys
import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1_foundation.schemas import (
    DocumentRecord,
    BenchmarkPair,
    StratumType,
    RelationType,
)
from phase2_data_and_benchmark.calculate_alpha import evaluate_benchmark_agreement


CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "corpus.jsonl"
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark" / "benchmark_240.jsonl"


def test_corpus_exists_and_count():
    """Verify frozen corpus contains exactly 2,000 valid records."""
    assert CORPUS_PATH.exists(), f"Corpus file not found at {CORPUS_PATH}"

    records = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(DocumentRecord.model_validate_json(line.strip()))

    assert len(records) == 2000, f"Expected 2000 records, got {len(records)}"


def test_corpus_domain_balance_and_filters():
    """Verify domains are balanced and proposal exclusion rules hold."""
    domains = {}
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = DocumentRecord.model_validate_json(line.strip())
            domains[rec.domain] = domains.get(rec.domain, 0) + 1

            # Exclusion rules
            assert len(rec.abstract.split()) >= 45, f"Abstract too short in {rec.doc_id}"
            assert "retracted" not in rec.title.lower(), f"Retracted title found in {rec.doc_id}"
            assert rec.domain in ["biomedical", "computer_security", "control_systems", "networking"]

    # Each domain should have significant representation (at least 300 records each)
    for dom in ["biomedical", "computer_security", "control_systems", "networking"]:
        assert dom in domains, f"Missing domain {dom}"
        assert domains[dom] >= 300, f"Domain {dom} underrepresented: {domains[dom]}"


def test_benchmark_strata_balance():
    """Verify benchmark has exactly 240 pairs balanced across all 4 strata."""
    assert BENCHMARK_PATH.exists(), f"Benchmark file not found at {BENCHMARK_PATH}"

    pairs = []
    strata_counts = {}
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            p = BenchmarkPair.model_validate_json(line.strip())
            pairs.append(p)
            strata_counts[p.stratum] = strata_counts.get(p.stratum, 0) + 1

    assert len(pairs) == 240, f"Expected 240 benchmark pairs, got {len(pairs)}"
    assert strata_counts[StratumType.STRATUM_1_CONTROLLED] == 60
    assert strata_counts[StratumType.STRATUM_2_ANALOGY] == 60
    assert strata_counts[StratumType.STRATUM_3_NATURAL] == 60
    assert strata_counts[StratumType.STRATUM_4_ADVERSARIAL] == 60


def test_krippendorff_alpha_agreement():
    """Verify Krippendorff's alpha calculation succeeds and meets threshold."""
    results = evaluate_benchmark_agreement(BENCHMARK_PATH)
    assert "krippendorff_alpha_overall" in results
    alpha = results["krippendorff_alpha_overall"]
    assert 0.70 <= alpha <= 1.0, f"Expected alpha >= 0.70, got {alpha}"
    assert results["multi_annotated_pairs"] == 180
    assert len(results["annotator_ids"]) >= 3
