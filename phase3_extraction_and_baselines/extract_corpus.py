"""
extract_corpus.py - Batch Extraction Script for 2,000 Documents.

Applies MultiMechanismExtractor across data/processed/corpus.jsonl to produce:
1. data/processed/extracted_mechanisms.jsonl (Individual q_im claims)
2. data/processed/document_graphs.jsonl (Compiled DocumentGraph Gi instances)
3. data/processed/extraction_metadata.json (Statistical audit manifest)
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1_foundation.schemas import DocumentRecord, DocumentGraph
from phase3_extraction_and_baselines.extractor import MultiMechanismExtractor


def run_batch_extraction(
    corpus_file: Path,
    graphs_out_file: Path,
    mechanisms_out_file: Path,
    max_mechanisms_per_doc: int = 10,
) -> Dict[str, Any]:
    """Processes all documents in corpus.jsonl and serializes extracted graphs."""
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

    graphs_out_file.parent.mkdir(parents=True, exist_ok=True)
    mechanisms_out_file.parent.mkdir(parents=True, exist_ok=True)

    extractor = MultiMechanismExtractor()
    total_docs = 0
    total_mechanisms = 0
    total_nodes = 0
    total_edges = 0
    polarity_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NULL": 0}

    start_time = time.time()
    print(f"Starting batch extraction on {corpus_file}...")

    with open(corpus_file, "r", encoding="utf-8") as f_in, \
         open(graphs_out_file, "w", encoding="utf-8") as f_graphs, \
         open(mechanisms_out_file, "w", encoding="utf-8") as f_mechs:

        for line in f_in:
            if not line.strip():
                continue
            doc = DocumentRecord.model_validate_json(line.strip())
            total_docs += 1

            # Extract mechanisms and compile graph
            mechanisms = extractor.extract_mechanisms_from_text(
                abstract=doc.abstract,
                doc_id=doc.doc_id,
                domain=doc.domain,
                max_mechanisms=max_mechanisms_per_doc,
            )

            for m in mechanisms:
                f_mechs.write(m.model_dump_json() + "\n")
                total_mechanisms += 1
                polarity_counts[m.polarity.name] += 1

            doc_graph = DocumentGraph.from_mechanisms(
                doc_id=doc.doc_id,
                title=doc.title,
                abstract=doc.abstract,
                domain=doc.domain,
                mechanisms=mechanisms,
                max_mechanisms=max_mechanisms_per_doc,
            )

            total_nodes += len(doc_graph.nodes)
            total_edges += len(doc_graph.edges)
            f_graphs.write(doc_graph.model_dump_json() + "\n")

            if total_docs % 500 == 0:
                print(f"Processed {total_docs}/2000 documents ({total_mechanisms} mechanisms extracted)...")

    elapsed = time.time() - start_time
    stats = {
        "total_documents_processed": total_docs,
        "total_mechanisms_extracted": total_mechanisms,
        "avg_mechanisms_per_doc": round(total_mechanisms / max(1, total_docs), 2),
        "total_graph_nodes": total_nodes,
        "total_graph_edges": total_edges,
        "avg_nodes_per_graph": round(total_nodes / max(1, total_docs), 2),
        "avg_edges_per_graph": round(total_edges / max(1, total_docs), 2),
        "polarity_distribution": polarity_counts,
        "elapsed_seconds": round(elapsed, 2),
        "throughput_docs_per_sec": round(total_docs / max(0.01, elapsed), 1),
    }

    meta_file = graphs_out_file.parent / "extraction_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nBatch extraction complete:")
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    c_path = PROJECT_ROOT / "data" / "processed" / "corpus.jsonl"
    g_path = PROJECT_ROOT / "data" / "processed" / "document_graphs.jsonl"
    m_path = PROJECT_ROOT / "data" / "processed" / "extracted_mechanisms.jsonl"
    run_batch_extraction(c_path, g_path, m_path)
