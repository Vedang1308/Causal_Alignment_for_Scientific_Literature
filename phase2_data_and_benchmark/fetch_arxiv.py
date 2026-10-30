"""
fetch_arxiv.py - Automated arXiv API Fetcher for CCST Corpus.
Fetches scientific title-abstract records across computer security (cs.CR),
networking (cs.NI), and control systems (cs.SY / eess.SY).
"""

import sys
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1_foundation.schemas import DocumentRecord

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def clean_text(text: Optional[str]) -> str:
    """Normalizes whitespace and removes extraneous linebreaks."""
    if not text:
        return ""
    return " ".join(text.strip().split())


def parse_arxiv_entry(entry: ET.Element) -> Optional[DocumentRecord]:
    """Parses a single Atom entry element into a DocumentRecord."""
    id_elem = entry.find("atom:id", ATOM_NS)
    title_elem = entry.find("atom:title", ATOM_NS)
    summary_elem = entry.find("atom:summary", ATOM_NS)
    published_elem = entry.find("atom:published", ATOM_NS)

    if id_elem is None or title_elem is None or summary_elem is None:
        return None

    raw_id = id_elem.text.strip() if id_elem.text else ""
    # Extract clean arXiv ID from URL (e.g. http://arxiv.org/abs/2301.12345v1 -> arxiv:2301.12345)
    arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id
    if "v" in arxiv_id:
        arxiv_id = arxiv_id.split("v")[0]
    doc_id = f"arxiv:{arxiv_id}"

    title = clean_text(title_elem.text)
    abstract = clean_text(summary_elem.text)

    # Authors
    authors = []
    for author_elem in entry.findall("atom:author", ATOM_NS):
        name_elem = author_elem.find("atom:name", ATOM_NS)
        if name_elem is not None and name_elem.text:
            authors.append(name_elem.text.strip())

    # Categories
    categories = []
    for cat_elem in entry.findall("atom:category", ATOM_NS):
        term = cat_elem.get("term")
        if term:
            categories.append(term)

    # Determine domain
    domain = "computer_systems"
    if any(c.startswith("cs.CR") for c in categories):
        domain = "computer_security"
    elif any(c.startswith("cs.SY") or c.startswith("eess.SY") for c in categories):
        domain = "control_systems"
    elif any(c.startswith("cs.NI") for c in categories):
        domain = "networking"

    year = 2023
    if published_elem is not None and published_elem.text:
        try:
            year = int(published_elem.text[:4])
        except ValueError:
            year = 2023

    return DocumentRecord(
        doc_id=doc_id,
        title=title,
        abstract=abstract,
        domain=domain,
        metadata={
            "authors": authors,
            "year": year,
            "categories": categories,
            "source": "arxiv",
        },
    )


def fetch_arxiv_records(
    query: str = "cat:cs.CR OR cat:cs.NI OR cat:cs.SY",
    max_records: int = 1000,
    batch_size: int = 100,
    delay_seconds: float = 3.0,
) -> List[DocumentRecord]:
    """
    Fetches title-abstract records from the arXiv API using pagination.
    Adheres to arXiv API rate-limiting recommendations.
    """
    records: List[DocumentRecord] = []
    start = 0

    print(f"Fetching up to {max_records} records from arXiv API for query: '{query}'...")

    while start < max_records:
        current_batch = min(batch_size, max_records - start)
        params = {
            "search_query": query,
            "start": start,
            "max_results": current_batch,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "CCST-Research-Agent/1.0"})

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)

            entries = root.findall("atom:entry", ATOM_NS)
            if not entries:
                print(f"No more entries returned at offset {start}. Stopping.")
                break

            for entry in entries:
                rec = parse_arxiv_entry(entry)
                if rec and len(rec.abstract.split()) >= 60:
                    records.append(rec)

            print(f"Fetched {len(records)} total valid arXiv records so far (offset: {start + current_batch})...")
            start += current_batch

            if start < max_records:
                time.sleep(delay_seconds)

        except Exception as e:
            print(f"Network warning while querying arXiv API: {e}. Returning collected records.")
            break

    return records


if __name__ == "__main__":
    records = fetch_arxiv_records(max_records=10)
    print(f"Sample fetched: {len(records)} records")
    if records:
        print(f"First record: {records[0].doc_id} - {records[0].title}")

