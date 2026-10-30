"""
generate_real_embeddings.py - GPU Batch Embeddings Generator for CCST Corpus.
Encodes 2,000 real scientific document abstracts using real pre-trained Sentence-BERT / SciBERT
on NVIDIA A100 GPU (PyTorch + CUDA).

Outputs:
- data/processed/corpus_embeddings.npy: shape (2000, 384) float32
- data/processed/corpus_doc_ids.json: mapping of array row index to document ID
"""

import sys
import json
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1_foundation.schemas import DocumentRecord


def compute_and_save_real_embeddings(
    corpus_jsonl_path: Path,
    output_embeddings_path: Path,
    output_doc_ids_path: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
):
    print("=" * 70)
    print(" GENERATING REAL NEURAL EMBEDDINGS ON GPU (A100)")
    print("=" * 70)

    # 1. Load Documents
    print(f"Loading corpus from: {corpus_jsonl_path}...")
    documents = []
    with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(DocumentRecord.model_validate_json(line))

    print(f"Loaded {len(documents):,} documents.")

    # 2. Check GPU & PyTorch
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        print(f"Using NVIDIA GPU: {gpu_name} ({vram_gb} GB VRAM)")
    else:
        print("Using CPU (Warning: GPU not detected, computation may take longer)")

    # 3. Load Real Model
    print(f"Loading pre-trained model '{model_name}' on {device}...")
    t0 = time.time()
    model = SentenceTransformer(model_name, device=device)
    print(f"Model loaded in {time.time() - t0:.2f}s.")

    # 4. Encode in Batches on GPU
    texts = [f"{d.title}. {d.abstract}" for d in documents]
    doc_ids = [d.doc_id for d in documents]

    print(f"Encoding {len(texts):,} documents in batches of {batch_size}...")
    t_enc_start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    enc_duration = time.time() - t_enc_start
    print(f"Encoding completed in {enc_duration:.2f}s ({len(texts)/enc_duration:.1f} docs/sec).")
    print(f"Generated embedding matrix shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    # 5. Save Artifacts
    output_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_embeddings_path, embeddings.astype(np.float32))
    print(f"Saved real embeddings to: {output_embeddings_path}")

    with open(output_doc_ids_path, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, indent=2)
    print(f"Saved doc ID mappings to: {output_doc_ids_path}")

    return embeddings, doc_ids


if __name__ == "__main__":
    c_path = PROJECT_ROOT / "data" / "processed" / "corpus.jsonl"
    emb_path = PROJECT_ROOT / "data" / "processed" / "corpus_embeddings.npy"
    ids_path = PROJECT_ROOT / "data" / "processed" / "corpus_doc_ids.json"
    compute_and_save_real_embeddings(c_path, emb_path, ids_path)
