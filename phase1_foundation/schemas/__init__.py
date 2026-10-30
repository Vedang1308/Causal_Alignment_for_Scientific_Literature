"""
Core Schema package for Context-Conditioned Causal Alignment (CCST).
"""

from .primitives import (
    RelationType,
    Polarity,
    EpistemicStatus,
    NodeRole,
    EdgeType,
    ContextDimension,
    ContextCompatibility,
    ProvenanceSpan,
)
from .mechanism import (
    MaterialContext,
    MechanismClaim,
)
from .graph import (
    GraphNode,
    GraphEdge,
    DocumentGraph,
)
from .certificate import (
    AlignedNodePair,
    AlignedEdgePair,
    ContextDiff,
    DisagreementCertificate,
)
from .benchmark import (
    StratumType,
    StratumSubtype,
    AnnotatorVote,
    DocumentRecord,
    BenchmarkPair,
)

__all__ = [
    # Primitives
    "RelationType",
    "Polarity",
    "EpistemicStatus",
    "NodeRole",
    "EdgeType",
    "ContextDimension",
    "ContextCompatibility",
    "ProvenanceSpan",
    # Mechanism
    "MaterialContext",
    "MechanismClaim",
    # Graph
    "GraphNode",
    "GraphEdge",
    "DocumentGraph",
    # Certificate
    "AlignedNodePair",
    "AlignedEdgePair",
    "ContextDiff",
    "DisagreementCertificate",
    # Benchmark
    "StratumType",
    "StratumSubtype",
    "AnnotatorVote",
    "DocumentRecord",
    "BenchmarkPair",
]
