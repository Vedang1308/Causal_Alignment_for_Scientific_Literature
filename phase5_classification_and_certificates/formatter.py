"""
Certificate Formatter and Exporter.
Exports DisagreementCertificate instances to:
1. Rich GitHub-flavored Markdown report with tables and evidence callouts
2. Canonical JSON artifact
3. LaTeX table row for paper inclusion
"""

import json
from typing import Dict, Any
from phase1_foundation.schemas.certificate import DisagreementCertificate


class CertificateFormatter:
    """
    Exporters for DisagreementCertificate instances.
    """
    @staticmethod
    def to_markdown(cert: DisagreementCertificate) -> str:
        """Generates an extensive, publication-grade markdown audit report."""
        lines = [
            f"# CCST Disagreement Certificate: `{cert.certificate_id}`",
            "",
            f"- **Document A:** `{cert.doc_a_id}`",
            f"- **Document B:** `{cert.doc_b_id}`",
            f"- **Assigned Relation:** **`{cert.predicted_relation.value.upper()}`**",
            f"- **Abstained:** `{'YES - ' + str(cert.abstention_reason) if cert.abstained else 'NO'}`",
            f"- **Faithfulness Gap Explained:** `{cert.gap_explained_fraction * 100:.1f}%`",
            "",
            "## 1. Calibrated Relation Probabilities",
            "| Relation Type | Calibrated Probability |",
            "| :--- | :--- |",
        ]
        for rel, p in cert.relation_probabilities.items():
            lines.append(f"| `{rel.value}` | `{p * 100:.2f}%` |")

        lines.extend([
            "",
            "## 2. Material Scope / Context Diff",
            "| Dimension | Document A | Document B | Compatibility Status |",
            "| :--- | :--- | :--- | :--- |",
        ])
        for cd in cert.context_diffs:
            val_a = cd.value_a or "*[Unspecified]*"
            val_b = cd.value_b or "*[Unspecified]*"
            lines.append(f"| **{cd.dimension.value}** | {val_a} | {val_b} | `{cd.compatibility.name}` |")

        lines.extend([
            "",
            "## 3. Minimal Disagreement Set (Conflicting Causal Edges)",
        ])
        if cert.minimal_disagreement_set:
            for item in cert.minimal_disagreement_set:
                lines.append(f"-  {item}")
        else:
            lines.append("*(No sign-inverted causal edges detected between aligned components)*")

        lines.extend([
            "",
            "## 4. Aligned Subgraph Components",
            f"**Total Aligned Nodes:** {len(cert.aligned_nodes)} | **Total Aligned Edges:** {len(cert.aligned_edges)}",
            "",
            "| Node A (Role) | Node B (Role) | Coupling Weight (T_uu') |",
            "| :--- | :--- | :--- |",
        ])
        for node in cert.aligned_nodes[:8]:  # display top alignments
            lines.append(f"| `{node.label_a}` ({node.role_a.value}) | `{node.label_b}` ({node.role_b.value}) | `{node.coupling_weight:.3f}` |")

        lines.extend([
            "",
            "## 5. Source Text Provenance Grounding",
        ])
        if cert.evidence_spans_a:
            lines.append("**Document A Evidence Grounding:**")
            for span in cert.evidence_spans_a[:3]:
                lines.append(f"> \"{span.text}\" (chars {span.start_char}-{span.end_char})")
        if cert.evidence_spans_b:
            lines.append("\n**Document B Evidence Grounding:**")
            for span in cert.evidence_spans_b[:3]:
                lines.append(f"> \"{span.text}\" (chars {span.start_char}-{span.end_char})")

        return "\n".join(lines)

    @staticmethod
    def to_json(cert: DisagreementCertificate, indent: int = 2) -> str:
        """Exports certificate to clean JSON string."""
        return cert.model_dump_json(indent=indent)

    @staticmethod
    def to_latex_row(cert: DisagreementCertificate) -> str:
        """Formats certificate as a row for LaTeX Table I."""
        rel = cert.predicted_relation.value.replace("_", r"\_")
        top_prob = max(cert.relation_probabilities.values()) if cert.relation_probabilities else 0.0
        n_flips = len(cert.minimal_disagreement_set)
        abstained_str = "Yes" if cert.abstained else "No"
        return f"{cert.doc_a_id} & {cert.doc_b_id} & {rel} & {top_prob:.2f} & {n_flips} & {abstained_str} \\\\"
