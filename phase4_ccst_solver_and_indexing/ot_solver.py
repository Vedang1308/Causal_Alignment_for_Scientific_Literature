"""
Context-Conditioned Signed Partial Optimal Transport (CCST) Solver.
Implements Equation (2) of Section IV-B:
    D_CCST(G_i, G_j) = min_{T >= 0} [ (1 - alpha) <M, T> + alpha E_GW(T) + lambda_u U(T) + lambda_g Phi_gate 1^T T 1 ]
subject to T 1 <= p_i, T^T 1 <= p_j.

Computes:
- Coupling matrix T* (|V_i| x |V_j|)
- Signed distance D_signed(G_i, G_j)
- Unsigned distance D_unsigned(G_i, G_j)
- Sign-flip score gap Delta_sign = D_signed - D_unsigned
- Unaligned mass fraction U(T*)
- Detailed node and edge alignments with provenance
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

from phase1_foundation.schemas.graph import DocumentGraph
from phase1_foundation.schemas.certificate import AlignedNodePair, AlignedEdgePair
from phase1_foundation.schemas.primitives import Polarity, NodeRole
from .cost_matrices import compute_all_cost_matrices
from .context_gate import ContextGate, ContextGateEvaluation


@dataclass
class AlignmentResult:
    """
    Result of a single (signed or unsigned) CCST optimal transport solve.
    """
    coupling_matrix: np.ndarray
    transport_cost: float
    semantic_cost: float
    gromov_cost: float
    unaligned_mass_penalty: float
    gate_penalty_cost: float
    total_aligned_mass: float
    unaligned_mass: float
    iterations: int
    converged: bool


@dataclass
class CCSTPairAlignment:
    """
    Full dual-view (signed + unsigned) alignment for a pair of document graphs.
    """
    doc_a_id: str
    doc_b_id: str
    signed_distance: float
    unsigned_distance: float
    signed_vs_unsigned_gap: float  # Delta_sign = D_signed - D_unsigned
    
    signed_result: AlignmentResult
    unsigned_result: AlignmentResult
    context_gate: ContextGateEvaluation
    
    aligned_nodes: List[AlignedNodePair] = field(default_factory=list)
    aligned_edges: List[AlignedEdgePair] = field(default_factory=list)
    sign_flip_edges: List[AlignedEdgePair] = field(default_factory=list)
    
    unaligned_nodes_a: List[str] = field(default_factory=list)
    unaligned_nodes_b: List[str] = field(default_factory=list)


def _solve_linear_partial_ot_sinkhorn(
    cost_matrix: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    reg: float = 0.05,
    max_iter: int = 60,
) -> np.ndarray:
    """
    Solves linear partial optimal transport subproblem via augmented Sinkhorn.
    Augments cost with a dummy row and column to absorb mass where cost >= 0.
    """
    n1, n2 = cost_matrix.shape
    # Augmented matrix: (n1 + 1) x (n2 + 1)
    C_aug = np.zeros((n1 + 1, n2 + 1), dtype=np.float32)
    C_aug[:n1, :n2] = cost_matrix
    # Thresholding: unmatched mass pays 0 marginal cost
    C_aug[:n1, n2] = 0.0
    C_aug[n1, :n2] = 0.0
    C_aug[n1, n2] = 0.0

    p_aug = np.append(p, 1.0)
    q_aug = np.append(q, 1.0)

    # Sinkhorn iterations
    K = np.exp(-C_aug / max(1e-4, reg))
    u = np.ones_like(p_aug)
    v = np.ones_like(q_aug)

    for _ in range(max_iter):
        u = p_aug / (np.dot(K, v) + 1e-12)
        v = q_aug / (np.dot(K.T, u) + 1e-12)

    T_aug = np.outer(u, v) * K
    # Extract real-to-real coupling
    T = T_aug[:n1, :n2]
    # Bound coupling by marginals
    r_sums = T.sum(axis=1, keepdims=True)
    scale_r = np.minimum(1.0, p[:, None] / (r_sums + 1e-12))
    T = T * scale_r
    c_sums = T.sum(axis=0, keepdims=True)
    scale_c = np.minimum(1.0, q[None, :] / (c_sums + 1e-12))
    T = T * scale_c

    return T


class CCSTOptimalTransportSolver:
    """
    Context-Conditioned Signed Partial Optimal Transport Solver.
    """
    def __init__(
        self,
        alpha: float = 0.4,          # Tradeoff between semantic (1-alpha) and structural (alpha)
        lambda_u: float = 0.35,      # Penalty for unaligned mass U(T)
        lambda_g: float = 0.5,       # Multiplier for context gating penalty Phi_gate
        max_iter: int = 35,          # Frank-Wolfe iterations
        tol: float = 1e-4,           # Convergence tolerance
        reg_sinkhorn: float = 0.04,  # Entropic regularization for subproblem
    ):
        self.alpha = alpha
        self.lambda_u = lambda_u
        self.lambda_g = lambda_g
        self.max_iter = max_iter
        self.tol = tol
        self.reg_sinkhorn = reg_sinkhorn
        self.context_gate = ContextGate()

    def _gw_loss_and_grad(
        self, C1: np.ndarray, C2: np.ndarray, T: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Computes Gromov-Wasserstein loss and gradient with squared Euclidean distortion.
        L(a, b) = (a - b)^2 = a^2 - 2ab + b^2
        """
        n1, n2 = T.shape
        m1 = T.sum(axis=1)  # (n1,)
        m2 = T.sum(axis=0)  # (n2,)

        C1_sq = C1 ** 2
        C2_sq = C2 ** 2

        # Quadratic terms
        term1 = float(np.dot(m1, np.dot(C1_sq, m1)))
        term3 = float(np.dot(m2, np.dot(C2_sq, m2)))
        # Cross term: Tr(C1 T C2.T T.T)
        C1_T_C2T = np.dot(C1, np.dot(T, C2.T))
        term2 = float(np.sum(C1_T_C2T * T))

        gw_loss = term1 + term3 - 2.0 * term2

        # Gradient wrt T
        grad_term1 = np.outer(np.dot(C1_sq, m1), np.ones(n2))
        grad_term3 = np.outer(np.ones(n1), np.dot(C2_sq.T, m2))
        grad_cross = 2.0 * (C1_T_C2T + np.dot(C1.T, np.dot(T, C2)))

        grad = 2.0 * (grad_term1 + grad_term3) - grad_cross
        return max(0.0, gw_loss), grad

    def _solve_pfgw(
        self,
        M: np.ndarray,
        C1: np.ndarray,
        C2: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        gate_penalty: float,
    ) -> AlignmentResult:
        """
        Executes Frank-Wolfe optimization for the CCST partial FGW objective.
        """
        n1, n2 = M.shape

        # Initial coupling: uniform partial product
        init_scale = 0.5
        T = init_scale * np.outer(p1, p2)

        prev_obj = float("inf")
        converged = False
        final_iter = 0

        # Effective linear cost component:
        # lambda_u * (2 - 2 * sum(T)) + lambda_g * Phi_gate * sum(T) + (1 - alpha) * <M, T>
        linear_bias = self.lambda_g * gate_penalty - 2.0 * self.lambda_u
        M_eff = (1.0 - self.alpha) * M + linear_bias

        for it in range(self.max_iter):
            final_iter = it + 1
            # 1. Compute GW loss and gradient
            gw_loss, gw_grad = self._gw_loss_and_grad(C1, C2, T)

            # 2. Total objective
            sem_cost = float(np.sum(M * T))
            aligned_mass = float(np.sum(T))
            unaligned_penalty = self.lambda_u * (2.0 - 2.0 * aligned_mass)
            gate_cost = self.lambda_g * gate_penalty * aligned_mass

            total_obj = (1.0 - self.alpha) * sem_cost + self.alpha * gw_loss + unaligned_penalty + gate_cost

            if abs(prev_obj - total_obj) < self.tol:
                converged = True
                break
            prev_obj = total_obj

            # 3. Total gradient wrt T
            grad = M_eff + self.alpha * gw_grad

            # 4. Solve linear partial transport subproblem
            S = _solve_linear_partial_ot_sinkhorn(
                grad, p1, p2, reg=self.reg_sinkhorn
            )

            # 5. Exact quadratic line search for step size gamma in [0, 1]
            D = S - T
            # Line search evaluates f(T + gamma D) = a gamma^2 + b gamma + c
            # Directional derivative b = <grad, D>
            b = float(np.sum(grad * D))
            
            # Quadratic coefficient from GW:
            # E_GW(T + gamma D) = E_GW(T) + gamma <grad_GW, D> + gamma^2 E_GW(D)
            gw_D, _ = self._gw_loss_and_grad(C1, C2, D)
            a = self.alpha * gw_D

            if a > 1e-9:
                gamma = max(0.0, min(1.0, -b / (2.0 * a)))
            else:
                gamma = 1.0 if b < 0 else 0.0

            # Fallback damping
            gamma = max(0.02, min(0.9, gamma))
            T = (1.0 - gamma) * T + gamma * S

        # Final evaluation
        final_gw, _ = self._gw_loss_and_grad(C1, C2, T)
        final_sem = float(np.sum(M * T))
        final_mass = float(np.sum(T))
        final_unaligned_pen = self.lambda_u * (2.0 - 2.0 * final_mass)
        final_gate_cost = self.lambda_g * gate_penalty * final_mass

        final_total = (1.0 - self.alpha) * final_sem + self.alpha * final_gw + final_unaligned_pen + final_gate_cost

        return AlignmentResult(
            coupling_matrix=T,
            transport_cost=float(final_total),
            semantic_cost=final_sem,
            gromov_cost=final_gw,
            unaligned_mass_penalty=final_unaligned_pen,
            gate_penalty_cost=final_gate_cost,
            total_aligned_mass=final_mass,
            unaligned_mass=max(0.0, 2.0 - 2.0 * final_mass),
            iterations=final_iter,
            converged=converged,
        )

    def align(
        self,
        g1: DocumentGraph,
        g2: DocumentGraph,
        signed: bool = True,
    ) -> AlignmentResult:
        """
        Solves CCST partial alignment under either signed or unsigned view.
        """
        matrices = compute_all_cost_matrices(g1, g2)
        M = matrices["M"]
        p1 = matrices["p1"]
        p2 = matrices["p2"]
        C1 = matrices["C1_signed"] if signed else matrices["C1_unsigned"]
        C2 = matrices["C2_signed"] if signed else matrices["C2_unsigned"]

        gate_eval = self.context_gate.evaluate(g1, g2)
        # In unsigned mode (detecting topological analogies), context gate is neutralized
        gate_penalty = gate_eval.gate_penalty if signed else 0.0

        return self._solve_pfgw(M, C1, C2, p1, p2, gate_penalty)

    def solve_pair(
        self,
        g1: DocumentGraph,
        g2: DocumentGraph,
        coupling_threshold: float = 0.04,
    ) -> CCSTPairAlignment:
        """
        Executes full dual-view CCST analysis:
        1. Solves signed alignment -> D_signed
        2. Solves unsigned alignment -> D_unsigned
        3. Computes signed-vs-unsigned gap: Delta_sign = D_signed - D_unsigned
        4. Extracts aligned node pairs and conflicting signed edges
        """
        # Context compatibility evaluation
        gate_eval = self.context_gate.evaluate(g1, g2)

        # 1. Signed solve
        res_signed = self.align(g1, g2, signed=True)

        # 2. Unsigned solve
        res_unsigned = self.align(g1, g2, signed=False)

        # 3. Gap computation
        gap = max(0.0, res_signed.transport_cost - res_unsigned.transport_cost)

        # 4. Extract aligned node pairs from coupling matrix
        T = res_signed.coupling_matrix
        aligned_nodes: List[AlignedNodePair] = []
        node_a_aligned = set()
        node_b_aligned = set()

        for i, node_a in enumerate(g1.nodes):
            for j, node_b in enumerate(g2.nodes):
                w = float(T[i, j])
                if w >= coupling_threshold:
                    aligned_nodes.append(
                        AlignedNodePair(
                            node_a_id=node_a.id,
                            node_b_id=node_b.id,
                            label_a=node_a.label,
                            label_b=node_b.label,
                            role_a=node_a.role,
                            role_b=node_b.role,
                            coupling_weight=w,
                        )
                    )
                    node_a_aligned.add(node_a.id)
                    node_b_aligned.add(node_b.id)

        # Unaligned nodes
        unaligned_a = [n.id for n in g1.nodes if n.id not in node_a_aligned]
        unaligned_b = [n.id for n in g2.nodes if n.id not in node_b_aligned]

        # 5. Extract aligned edge pairs and detect sign flips
        aligned_edges: List[AlignedEdgePair] = []
        sign_flip_edges: List[AlignedEdgePair] = []

        # Map node pairings
        node_map = {p.node_a_id: p.node_b_id for p in aligned_nodes}

        for edge_a in g1.edges:
            target_b_src = node_map.get(edge_a.source)
            target_b_tgt = node_map.get(edge_a.target)

            if target_b_src and target_b_tgt:
                # Look for corresponding edge in g2
                for edge_b in g2.edges:
                    if edge_b.source == target_b_src and edge_b.target == target_b_tgt:
                        is_flip = (
                            edge_a.sign != edge_b.sign
                            and edge_a.sign != Polarity.NULL
                            and edge_b.sign != Polarity.NULL
                        )
                        pair = AlignedEdgePair(
                            edge_a=(edge_a.source, edge_a.target),
                            edge_b=(edge_b.source, edge_b.target),
                            sign_a=edge_a.sign,
                            sign_b=edge_b.sign,
                            is_sign_flip=is_flip,
                            edge_type_a=edge_a.edge_type.value,
                            edge_type_b=edge_b.edge_type.value,
                        )
                        aligned_edges.append(pair)
                        if is_flip:
                            sign_flip_edges.append(pair)

        return CCSTPairAlignment(
            doc_a_id=g1.doc_id,
            doc_b_id=g2.doc_id,
            signed_distance=res_signed.transport_cost,
            unsigned_distance=res_unsigned.transport_cost,
            signed_vs_unsigned_gap=float(gap),
            signed_result=res_signed,
            unsigned_result=res_unsigned,
            context_gate=gate_eval,
            aligned_nodes=aligned_nodes,
            aligned_edges=aligned_edges,
            sign_flip_edges=sign_flip_edges,
            unaligned_nodes_a=unaligned_a,
            unaligned_nodes_b=unaligned_b,
        )

