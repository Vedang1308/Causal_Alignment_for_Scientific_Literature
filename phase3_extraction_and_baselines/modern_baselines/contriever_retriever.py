"""
contriever_retriever.py - Modern Dense Contriever Retrieval Baseline.

Proposal References:
Table I: Modern AI/DL Retrieval - Contriever [10]
Section V: Unsupervised dense information retrieval with contrastive learning.
"""

import sys
import math
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1_foundation.schemas import DocumentRecord, BenchmarkPair
from phase3_extraction_and_baselines.modern_baselines.sbert_retriever import SBERTRetriever


class ContrieverRetriever(SBERTRetriever):
    """
    Contriever dense retrieval baseline (Izacard et al. [10]).
    Uses facebook/contriever model weights.
    """

    def __init__(
        self,
        model_name: str = "facebook/contriever",
        device: Optional[str] = None,
        use_mock_fallback: bool = True,
    ):
        super().__init__(model_name=model_name, device=device, use_mock_fallback=use_mock_fallback)


if __name__ == "__main__":
    from phase2_data_and_benchmark.curate_benchmark import generate_stratum3_natural
    pairs = generate_stratum3_natural(count=5)
    docs = [p.doc_a for p in pairs] + [p.doc_b for p in pairs]
    retriever = ContrieverRetriever().fit(docs)
    res = retriever.evaluate_retrieval_metrics(pairs)
    print("Contriever Retriever Evaluation:", res)
