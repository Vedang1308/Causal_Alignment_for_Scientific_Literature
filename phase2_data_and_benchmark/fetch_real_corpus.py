"""
fetch_real_corpus.py - Production Live Fetcher for CCST Corpus.
Downloads 2,000 real, published scientific papers directly from:
- PubMed E-Utilities API (500 Biomedical papers: pharmacology, cardiology, oncology)
- arXiv API (500 Computer Security papers: cs.CR)
- arXiv API (500 Networking papers: cs.NI)
- arXiv API (500 Control Systems papers: cs.SY, eess.SY)

Applies all proposal Section II filters:
- Exclusion of retractions
- Strict English language validation
- Minimum abstract length (>= 50 words)
- Deduplication by normalized title hash
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1_foundation.schemas import DocumentRecord
from phase2_data_and_benchmark.fetch_arxiv import fetch_arxiv_records
from phase2_data_and_benchmark.fetch_pubmed import fetch_pubmed_records
from phase2_data_and_benchmark.build_corpus import is_english, is_retracted, title_fingerprint


def fetch_2000_real_documents(
    output_path: Path,
    target_per_domain: int = 500,
) -> List[DocumentRecord]:
    """
    Fetches real papers across 4 scientific domains and freezes them into JSONL format.
    Supports resuming from existing partially collected records.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_records: List[DocumentRecord] = []
    seen_fingerprints: Set[str] = set()

    # If output_path already has records (e.g. arXiv records from prior run), load them
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = DocumentRecord.model_validate_json(line)
                        fp = title_fingerprint(rec.title)
                        if fp not in seen_fingerprints:
                            seen_fingerprints.add(fp)
                            all_records.append(rec)
            print(f"Loaded {len(all_records)} existing records from {output_path.name}")
        except Exception:
            all_records = []
            seen_fingerprints = set()

    def count_by_domain(domain_name: str) -> int:
        return sum(1 for r in all_records if r.domain == domain_name)

    def filter_and_add(candidates: List[DocumentRecord], domain_label: str) -> int:
        added = 0
        for rec in candidates:
            if len(rec.abstract.split()) < 50:
                continue
            if not is_english(rec.abstract):
                continue
            if is_retracted(rec.title, rec.abstract):
                continue
            fp = title_fingerprint(rec.title)
            if fp in seen_fingerprints:
                continue
            seen_fingerprints.add(fp)
            all_records.append(rec)
            added += 1
        return added

    print("=" * 70)
    print(" FETCHING REAL SCIENTIFIC DOCUMENTS FROM LIVE APIS")
    print("=" * 70)

    # 1. Biomedical from PubMed
    current_bio = count_by_domain("biomedical")
    if current_bio < target_per_domain:
        needed = target_per_domain - current_bio
        print(f"\n[1/4] Fetching {needed} real biomedical papers from PubMed E-Utilities...")
        try:
            pubmed_recs = fetch_pubmed_records(
                query="(atherosclerosis OR hypertension OR metformin OR 'calcium channel' OR kinase OR diabetes OR oncology) AND hasabstract[text]",
                max_records=needed + 150,
                batch_size=100,
                delay_seconds=0.5,
            )
            added_bio = filter_and_add(pubmed_recs, "biomedical")
            print(f" -> Successfully acquired {added_bio} real PubMed biomedical records (Total bio: {count_by_domain('biomedical')}).")
        except Exception as e:
            print(f"PubMed live fetch error: {e}")
    else:
        print(f"\n[1/4] Biomedical papers already satisfied ({current_bio} records).")

    # 2. Computer Security from arXiv (cs.CR)
    current_sec = count_by_domain("computer_security")
    if current_sec < target_per_domain:
        needed = target_per_domain - current_sec
        print(f"\n[2/4] Fetching {needed} real computer security papers from arXiv (cs.CR)...")
        try:
            sec_recs = fetch_arxiv_records(
                query="cat:cs.CR",
                max_records=needed + 150,
                batch_size=100,
                delay_seconds=3.0,
            )
            for r in sec_recs:
                r.domain = "computer_security"
            added_sec = filter_and_add(sec_recs, "computer_security")
            print(f" -> Successfully acquired {added_sec} real arXiv security records (Total sec: {count_by_domain('computer_security')}).")
        except Exception as e:
            print(f"arXiv security live fetch error: {e}")
    else:
        print(f"\n[2/4] Computer security papers already satisfied ({current_sec} records).")

    # 3. Networking from arXiv (cs.NI)
    current_net = count_by_domain("networking")
    if current_net < target_per_domain:
        needed = target_per_domain - current_net
        print(f"\n[3/4] Fetching {needed} real networking papers from arXiv (cs.NI)...")
        try:
            net_recs = fetch_arxiv_records(
                query="cat:cs.NI",
                max_records=needed + 150,
                batch_size=100,
                delay_seconds=3.0,
            )
            for r in net_recs:
                r.domain = "networking"
            added_net = filter_and_add(net_recs, "networking")
            print(f" -> Successfully acquired {added_net} real arXiv networking records (Total net: {count_by_domain('networking')}).")
        except Exception as e:
            print(f"arXiv networking live fetch error: {e}")
    else:
        print(f"\n[3/4] Networking papers already satisfied ({current_net} records).")

    # 4. Control Systems from arXiv (cs.SY / eess.SY)
    current_ctrl = count_by_domain("control_systems")
    if current_ctrl < target_per_domain:
        needed = target_per_domain - current_ctrl
        print(f"\n[4/4] Fetching {needed} real control systems papers from arXiv (cs.SY / eess.SY)...")
        try:
            ctrl_recs = fetch_arxiv_records(
                query="cat:cs.SY OR cat:eess.SY",
                max_records=needed + 150,
                batch_size=100,
                delay_seconds=3.0,
            )
            for r in ctrl_recs:
                r.domain = "control_systems"
            added_ctrl = filter_and_add(ctrl_recs, "control_systems")
            print(f" -> Successfully acquired {added_ctrl} real arXiv control systems records (Total ctrl: {count_by_domain('control_systems')}).")
        except Exception as e:
            print(f"arXiv control live fetch error: {e}")
    else:
        print(f"\n[4/4] Control systems papers already satisfied ({current_ctrl} records).")

    print("\n" + "=" * 70)
    print(f" Total Real Public Records Collected: {len(all_records)}")
    print("=" * 70)

    # Save to both output_path and official corpus.jsonl
    print(f"Freezing real corpus records to: {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(rec.model_dump_json() + "\n")

    corpus_official = PROJECT_ROOT / "data" / "processed" / "corpus.jsonl"
    print(f"Updating official corpus file: {corpus_official}...")
    with open(corpus_official, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(rec.model_dump_json() + "\n")

    # Generate metadata report
    domain_counts = {}
    for r in all_records:
        domain_counts[r.domain] = domain_counts.get(r.domain, 0) + 1

    print("Domain distribution of real corpus:")
    for d, c in domain_counts.items():
        print(f"  - {d}: {c} papers")

    return all_records


if __name__ == "__main__":
    out_file = PROJECT_ROOT / "data" / "processed" / "corpus_real.jsonl"
    fetch_2000_real_documents(output_path=out_file, target_per_domain=500)

