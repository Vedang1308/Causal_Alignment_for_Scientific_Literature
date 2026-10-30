"""
Document Graph representation G_i = (V_i, E_i, X_i, R_i, Z_i, p_i, ρ_i)
defined in Section IV-A of the CCST proposal.

Preserves:
- Nodes (V_i): population, intervention, mediator, outcome, and context entities.
- Node mass (p_i): confidence-weighted mass for partial optimal transport (sum(p_i) = 1).
- Edges (E_i): typed causal links with direction, sign (δ), time lag (τ), and confidence.
- Document level context, domain, and textual provenance.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from .primitives import (
    NodeRole,
    EdgeType,
    Polarity,
    ProvenanceSpan,
    ContextDimension,
)
from .mechanism import MechanismClaim, MaterialContext


class GraphNode(BaseModel):
    """
    Individual node in the reported mechanism graph.
    """
    id: str = Field(description="Unique identifier for the node within the document graph")
    role: NodeRole = Field(description="Functional role (population, intervention, mediator, outcome, context)")
    label: str = Field(description="Canonical textual label (e.g. 'atorvastatin', 'blood pressure')")
    mass: float = Field(default=1.0, gt=0.0, description="Confidence-weighted mass p_i for optimal transport")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Supplementary entity attributes")
    embedding: Optional[List[float]] = Field(default=None, description="Optional vector embedding")
    provenance: Optional[ProvenanceSpan] = Field(default=None, description="Character span in source text")

    @field_validator("id", "label")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class GraphEdge(BaseModel):
    """
    Typed, signed directed edge in the reported mechanism graph.
    """
    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    edge_type: EdgeType = Field(default=EdgeType.CAUSES, description="Semantic edge type")
    direction: str = Field(default="forward", description="Causal direction: 'forward', 'bidirectional'")
    sign: Polarity = Field(default=Polarity.POSITIVE, description="Polarity δ (+1: promotes/increases, -1: inhibits/decreases, 0: null)")
    time_lag: Optional[str] = Field(default=None, description="Temporal duration or lag τ")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction / edge confidence")
    provenance: Optional[ProvenanceSpan] = Field(default=None, description="Evidence span supporting this edge")


class DocumentGraph(BaseModel):
    """
    Variable-size reported-mechanism graph G_i representing a document's causal claims.
    """
    doc_id: str = Field(description="Document identifier (e.g. arXiv:2301.12345 or PMID:34567890)")
    title: str = Field(description="Paper title")
    abstract: str = Field(description="Original paper abstract text")
    domain: str = Field(description="Scientific discipline (e.g. 'biomedical', 'computer_security', 'control_systems')")
    nodes: List[GraphNode] = Field(default_factory=list, description="Set of graph nodes V_i")
    edges: List[GraphEdge] = Field(default_factory=list, description="Set of graph edges E_i")
    mechanisms: List[MechanismClaim] = Field(default_factory=list, description="Underlying mechanism claims q_im")
    context_summary: MaterialContext = Field(default_factory=MaterialContext, description="Aggregated material scope conditions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional document metadata (year, authors, etc.)")

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Lookup node by ID."""
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def get_normalized_node_masses(self) -> List[float]:
        """
        Returns normalized node mass distribution p_i where sum(p_i) == 1.0,
        required for the CCST partial coupling T1 <= p_i.
        """
        if not self.nodes:
            return []
        total_mass = sum(n.mass for n in self.nodes)
        if total_mass <= 0:
            return [1.0 / len(self.nodes)] * len(self.nodes)
        return [n.mass / total_mass for n in self.nodes]

    def to_networkx(self):
        """
        Converts the DocumentGraph into a networkx.DiGraph with full node/edge attributes.
        Requires networkx installed; returns the DiGraph instance.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required for to_networkx(). Install via 'pip install networkx'.")

        G = nx.DiGraph()
        G.graph["doc_id"] = self.doc_id
        G.graph["title"] = self.title
        G.graph["domain"] = self.domain

        for n in self.nodes:
            G.add_node(
                n.id,
                role=n.role.value,
                label=n.label,
                mass=n.mass,
                attributes=n.attributes,
                embedding=n.embedding,
                provenance=n.provenance.model_dump() if n.provenance else None,
            )

        for e in self.edges:
            G.add_edge(
                e.source,
                e.target,
                edge_type=e.edge_type.value,
                sign=e.sign.value,
                direction=e.direction,
                time_lag=e.time_lag,
                confidence=e.confidence,
                provenance=e.provenance.model_dump() if e.provenance else None,
            )

        return G

    @classmethod
    def from_networkx(
        cls,
        G,
        doc_id: str,
        title: str = "",
        abstract: str = "",
        domain: str = "general",
    ) -> "DocumentGraph":
        """
        Constructs a DocumentGraph instance from a networkx.DiGraph.
        """
        nodes = []
        for node_id, data in G.nodes(data=True):
            role_val = data.get("role", NodeRole.INTERVENTION.value)
            prov_data = data.get("provenance")
            nodes.append(
                GraphNode(
                    id=str(node_id),
                    role=NodeRole(role_val) if role_val in [r.value for r in NodeRole] else NodeRole.INTERVENTION,
                    label=data.get("label", str(node_id)),
                    mass=float(data.get("mass", 1.0)),
                    attributes=data.get("attributes", {}),
                    embedding=data.get("embedding"),
                    provenance=ProvenanceSpan(**prov_data) if prov_data else None,
                )
            )

        edges = []
        for u, v, data in G.edges(data=True):
            prov_data = data.get("provenance")
            edges.append(
                GraphEdge(
                    source=str(u),
                    target=str(v),
                    edge_type=EdgeType(data.get("edge_type", EdgeType.CAUSES.value)),
                    direction=data.get("direction", "forward"),
                    sign=Polarity(data.get("sign", Polarity.POSITIVE.value)),
                    time_lag=data.get("time_lag"),
                    confidence=float(data.get("confidence", 1.0)),
                    provenance=ProvenanceSpan(**prov_data) if prov_data else None,
                )
            )

        return cls(
            doc_id=doc_id,
            title=title or G.graph.get("title", ""),
            abstract=abstract,
            domain=domain or G.graph.get("domain", "general"),
            nodes=nodes,
            edges=edges,
        )

    @classmethod
    def from_mechanisms(
        cls,
        doc_id: str,
        title: str,
        abstract: str,
        domain: str,
        mechanisms: List[MechanismClaim],
        max_mechanisms: int = 10,
    ) -> "DocumentGraph":
        """
        Compiles a list of MechanismClaims q_im into a unified DocumentGraph G_i.
        Retains up to max_mechanisms (M_max) ordered by certainty/importance.
        """
        retained = sorted(mechanisms, key=lambda m: (1.0 - m.uncertainty), reverse=True)[:max_mechanisms]

        node_map: Dict[str, GraphNode] = {}
        edge_list: List[GraphEdge] = []
        aggregated_context = MaterialContext()

        def get_or_create_node(name: str, role: NodeRole, prov: Optional[ProvenanceSpan] = None) -> str:
            clean_name = name.strip()
            node_key = f"{role.value}_{clean_name.lower().replace(' ', '_')}"
            if node_key not in node_map:
                node_map[node_key] = GraphNode(
                    id=node_key,
                    role=role,
                    label=clean_name,
                    mass=1.0,
                    provenance=prov,
                )
            else:
                node_map[node_key].mass += 0.5  # Increase mass for frequently referenced entities
            return node_key

        for mech in retained:
            # Aggregate material context conditions
            for dim, val in mech.context.conditions.items():
                if dim not in aggregated_context.conditions:
                    aggregated_context.conditions[dim] = val

            pop_id = get_or_create_node(mech.population, NodeRole.POPULATION, mech.provenance)
            act_id = get_or_create_node(mech.intervention, NodeRole.INTERVENTION, mech.provenance)
            out_id = get_or_create_node(mech.outcome, NodeRole.OUTCOME, mech.provenance)

            # Population context edge
            edge_list.append(
                GraphEdge(
                    source=pop_id,
                    target=act_id,
                    edge_type=EdgeType.REGULATES,
                    sign=Polarity.NULL,
                    confidence=1.0 - mech.uncertainty,
                    provenance=mech.provenance,
                )
            )

            # Causal pathway: Intervention -> Mediators -> Outcome
            curr_src = act_id
            if mech.mediating_steps:
                for med_text in mech.mediating_steps:
                    med_id = get_or_create_node(med_text, NodeRole.MEDIATOR, mech.provenance)
                    edge_list.append(
                        GraphEdge(
                            source=curr_src,
                            target=med_id,
                            edge_type=EdgeType.MEDIATES,
                            sign=Polarity.POSITIVE,
                            confidence=1.0 - mech.uncertainty,
                            provenance=mech.provenance,
                        )
                    )
                    curr_src = med_id

            # Final edge to outcome carrying polarity delta
            edge_list.append(
                GraphEdge(
                    source=curr_src,
                    target=out_id,
                    edge_type=EdgeType.CAUSES if mech.polarity != Polarity.NEGATIVE else EdgeType.INHIBITS,
                    sign=mech.polarity,
                    time_lag=mech.time_horizon,
                    confidence=1.0 - mech.uncertainty,
                    provenance=mech.provenance,
                )
            )

        return cls(
            doc_id=doc_id,
            title=title,
            abstract=abstract,
            domain=domain,
            nodes=list(node_map.values()),
            edges=edge_list,
            mechanisms=retained,
            context_summary=aggregated_context,
        )

