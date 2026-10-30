"""
fetch_pubmed.py - Automated PubMed E-Utilities Fetcher for CCST Corpus.
Fetches scientific biomedical title-abstract records via NCBI Entrez E-Utilities.
"""

import sys
import time
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1_foundation.schemas import DocumentRecord

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def clean_text(text: Optional[str]) -> str:
    """Normalizes whitespace and removes extraneous linebreaks."""
    if not text:
        return ""
    return " ".join(text.strip().split())


def search_pubmed_ids(query: str, max_records: int = 1000) -> List[str]:
    """Queries ESearch to retrieve PubMed IDs matching the scientific query."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_records,
        "sort": "pub_date",
    }
    url = f"{ESEARCH_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "CCST-Biomed-Agent/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        id_list = data.get("esearchresult", {}).get("idlist", [])
        return id_list
    except Exception as e:
        print(f"Error querying PubMed ESearch: {e}")
        return []


def fetch_pubmed_abstracts(pmids: List[str], batch_size: int = 100, delay_seconds: float = 0.5) -> List[DocumentRecord]:
    """Fetches full metadata and abstracts for a list of PMIDs via EFetch XML."""
    records: List[DocumentRecord] = []

    for i in range(0, len(pmids), batch_size):
        batch_ids = pmids[i : i + batch_size]
        params = {
            "db": "pubmed",
            "id": ",".join(batch_ids),
            "retmode": "xml",
        }
        url = f"{EFETCH_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "CCST-Biomed-Agent/1.0"})

        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                xml_data = resp.read()

            root = ET.fromstring(xml_data)
            for article in root.findall(".//PubmedArticle"):
                pmid_elem = article.find(".//MedlineCitation/PMID")
                title_elem = article.find(".//ArticleTitle")
                abstract_elem = article.find(".//Abstract/AbstractText")
                year_elem = article.find(".//Journal/JournalIssue/PubDate/Year")

                if pmid_elem is None or title_elem is None or abstract_elem is None:
                    continue

                pmid = pmid_elem.text.strip() if pmid_elem.text else ""
                title = clean_text(title_elem.text)
                
                # Abstract could have multiple AbstractText elements (structured abstract)
                abs_parts = []
                for abs_node in article.findall(".//Abstract/AbstractText"):
                    label = abs_node.get("Label")
                    text_val = abs_node.text or ""
                    if label:
                        abs_parts.append(f"{label}: {clean_text(text_val)}")
                    else:
                        abs_parts.append(clean_text(text_val))
                abstract = " ".join(abs_parts).strip()

                if not abstract or len(abstract.split()) < 50:
                    continue

                # Authors
                authors = []
                for auth in article.findall(".//AuthorList/Author"):
                    last = auth.find("LastName")
                    fore = auth.find("ForeName")
                    if last is not None and last.text:
                        authors.append(f"{fore.text if fore is not None else ''} {last.text}".strip())

                year = 2023
                if year_elem is not None and year_elem.text:
                    try:
                        year = int(year_elem.text)
                    except ValueError:
                        year = 2023

                records.append(
                    DocumentRecord(
                        doc_id=f"pmid:{pmid}",
                        title=title,
                        abstract=abstract,
                        domain="biomedical",
                        metadata={
                            "authors": authors,
                            "year": year,
                            "source": "pubmed",
                        },
                    )
                )

            print(f"Fetched {len(records)} biomedical records so far...")
            time.sleep(delay_seconds)

        except Exception as e:
            print(f"Warning fetching batch {i} from PubMed: {e}")
            continue

    return records


def fetch_pubmed_records(
    query: str = "(hypertension OR atherosclerosis OR oncology OR infection OR drug therapy) AND hasabstract[text]",
    max_records: int = 1000,
) -> List[DocumentRecord]:
    """Top-level function to search and fetch biomedical records."""
    print(f"Searching PubMed for up to {max_records} records with query: '{query}'...")
    pmids = search_pubmed_ids(query, max_records=max_records)
    print(f"Retrieved {len(pmids)} PMIDs. Fetching full abstract records...")
    return fetch_pubmed_abstracts(pmids)


if __name__ == "__main__":
    records = fetch_pubmed_records(max_records=10)
    print(f"Sample fetched: {len(records)} records")
    if records:
        print(f"First record: {records[0].doc_id} - {records[0].title}")

