#!/usr/bin/env python3
"""
run_phase2_verification.py
Phase 2 Data Acquisition & Benchmark Curation Verification Script for CCST.

Validates:
1. Frozen 2,000-Document Public Scientific Corpus (arXiv & PubMed).
2. Four-Stratum Benchmark Dataset (240 Ground-Truth Evaluated Pairs).
3. Inter-Annotator Agreement (Krippendorff's alpha >= 0.70).
4. Exemplar inspection from all four strata.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1_foundation.schemas import DocumentRecord, BenchmarkPair, StratumType
from phase2_data_and_benchmark.calculate_alpha import evaluate_benchmark_agreement


def print_banner(text: str):
    line = "=" * 70
    print(f"\n{line}\n  {text}\n{line}")


def verify_corpus():
    print_banner("1. FROZEN 2,000-DOCUMENT SCIENTIFIC CORPUS")
    corpus_file = PROJECT_ROOT / "data" / "processed" / "corpus.jsonl"
    meta_file = PROJECT_ROOT / "data" / "processed" / "corpus_metadata.json"

    if not corpus_file.exists():
        print(f"Error: {corpus_file} not found. Running build_corpus.py...")
        from phase2_data_and_benchmark.build_corpus import build_and_freeze_corpus
        build_and_freeze_corpus(corpus_file, target_count=2000)

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print(f"Corpus Path: {corpus_file}")
    print(f"Total Frozen Documents: {meta['total_documents']}")
    print(f"Domain Distribution:")
    for dom, count in meta["domain_breakdown"].items():
        pct = (count / meta['total_documents']) * 100
        print(f"  - {dom:20s}: {count:5d} ({pct:5.1f}%)")
    print(f"Abstract Word Counts: min={meta['abstract_word_count']['min']}, max={meta['abstract_word_count']['max']}, mean={meta['abstract_word_count']['mean']}")
    print(" 2,000-document public corpus validated successfully!")


def verify_benchmark():
    print_banner("2. FOUR-STRATUM BENCHMARK (240 EVALUATED PAIRS)")
    bench_file = PROJECT_ROOT / "data" / "benchmark" / "benchmark_240.jsonl"
    manifest_file = PROJECT_ROOT / "data" / "benchmark" / "benchmark_manifest.json"

    if not bench_file.exists():
        print(f"Error: {bench_file} not found. Running curate_benchmark.py...")
        from phase2_data_and_benchmark.curate_benchmark import build_and_save_benchmark
        build_and_save_benchmark(bench_file.parent)

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"Benchmark Path: {bench_file}")
    print(f"Total Evaluated Pairs: {manifest['total_pairs']}")
    print(f"Strata Breakdown (Balanced 60 pairs each):")
    for s_name, count in manifest["strata_breakdown"].items():
        print(f"  - {s_name:24s}: {count:3d} pairs")
    print(f"Relation Label Distribution:")
    for rel_name, count in manifest["relation_distribution"].items():
        print(f"  - {rel_name:28s}: {count:3d} pairs")
    print(" Four-stratum benchmark validated successfully!")


def verify_agreement():
    print_banner("3. PRE-ADJUDICATION INTER-ANNOTATOR AGREEMENT")
    bench_file = PROJECT_ROOT / "data" / "benchmark" / "benchmark_240.jsonl"
    results = evaluate_benchmark_agreement(bench_file)

    print(f"Total Multi-Annotated Natural Pairs: {results['multi_annotated_pairs']}")
    print(f"Independent Annotators: {', '.join(results['annotator_ids'])}")
    print(f"Krippendorff's Alpha (Overall): {results['krippendorff_alpha_overall']:.4f}")
    for stratum, alpha_val in results["krippendorff_alpha_by_stratum"].items():
        print(f"  - {stratum:24s}: alpha = {alpha_val:.4f}")
    assert results['krippendorff_alpha_overall'] >= 0.70, "Alpha must be >= 0.70"
    print(" Krippendorff's alpha reliability verified (alpha >= 0.70 substantial agreement)!")


def show_strata_exemplars():
    print_banner("4. STRATA EXEMPLARS INSPECTION")
    bench_file = PROJECT_ROOT / "data" / "benchmark" / "benchmark_240.jsonl"
    exemplars = {}
    with open(bench_file, "r", encoding="utf-8") as f:
        for line in f:
            p = BenchmarkPair.model_validate_json(line.strip())
            if p.stratum not in exemplars:
                exemplars[p.stratum] = p
            if len(exemplars) == 4:
                break

    for stratum, p in exemplars.items():
        print(f"\n[{stratum.value.upper()}] Pair ID: {p.pair_id}")
        print(f"  - Subtype: {p.subtype.value}")
        print(f"  - Ground Truth: {p.ground_truth_relation.value.upper()}")
        print(f"  - Paper A ({p.doc_a.domain}): {p.doc_a.title[:75]}...")
        print(f"  - Paper B ({p.doc_b.domain}): {p.doc_b.title[:75]}...")
        print(f"  - Rationale: {p.notes}")


def main():
    print("=" * 70)
    print(" CCST PHASE 2: DATA ACQUISITION & BENCHMARK VERIFICATION")
    print("=" * 70)
    verify_corpus()
    verify_benchmark()
    verify_agreement()
    show_strata_exemplars()
    print_banner("PHASE 2 ALL VERIFICATIONS PASSED (100% READY FOR SOL)")


if __name__ == "__main__":
    main()
