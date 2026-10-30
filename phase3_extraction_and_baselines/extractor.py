"""
extractor.py - Provenance-Linked Multi-Mechanism Extractor for CCST.

Extracts reported causal mechanisms according to Equation (1):
    q_im = (C, P, A, M, Y, τ, δ, η, u, p)_im

Features:
- Preserves exact source text character spans (p = (start_char, end_char))
- Resolves material context dimensions (C) and marks missing dimensions as UNKNOWN
- Classifies polarity (δ ∈ {+1, -1, 0}) and epistemic status (η)
- Detects multi-step mediating pathways (M)
- Fast, deterministic local parser + optional LLM schema extractor
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1_foundation.schemas import (
    MechanismClaim,
    MaterialContext,
    ContextDimension,
    Polarity,
    EpistemicStatus,
    ProvenanceSpan,
    DocumentRecord,
    DocumentGraph,
)


class MultiMechanismExtractor:
    """
    Provenance-linked causal mechanism extractor.
    Segments claim-bearing spans and produces structured MechanismClaim instances.
    """

    # Polarity Lexicons
    NEGATIVE_VERBS = {
        "decreases", "decreased", "decreasing",
        "reduces", "reduced", "reducing", "reduction",
        "inhibits", "inhibited", "inhibiting", "inhibition",
        "suppresses", "suppressed", "suppressing", "suppression",
        "blocks", "blocked", "blocking",
        "attenuates", "attenuated", "attenuating",
        "downregulates", "downregulated", "downregulation",
        "prevents", "prevented", "preventing", "prevention",
        "eliminates", "eliminated", "eliminating",
        "mitigates", "mitigated", "mitigating", "mitigation",
        "lowers", "lowered", "lowering",
        "restricts", "restricted", "restricting",
    }

    POSITIVE_VERBS = {
        "increases", "increased", "increasing",
        "stimulates", "stimulated", "stimulating", "stimulation",
        "upregulates", "upregulated", "upregulation",
        "triggers", "triggered", "triggering",
        "enhances", "enhanced", "enhancing",
        "promotes", "promoted", "promoting",
        "induces", "induced", "inducing", "induction",
        "elevates", "elevated", "elevating",
        "improves", "improved", "improving",
        "maximizes", "maximized", "maximizing",
        "stabilizes", "stabilized", "stabilizing",
        "accelerates", "accelerated", "accelerating",
    }

    NULL_PATTERNS = [
        r"\bno significant (effect|alteration|difference|change)\b",
        r"\bfailed to (reduce|increase|lower|alter|modulate)\b",
        r"\bexerted no (inhibitory|stimulatory|therapeutic) effect\b",
        r"\bnegligible association\b",
        r"\bnull correlation\b",
    ]

    # Epistemic Status Markers
    QUOTED_MARKERS = [
        r"\bprior (studies|literature|work|reports) (claimed|suggested|asserted)\b",
        r"\bearlier assertions were flawed\b",
        r"\bquoted but not endorsed\b",
        r"\bcontrary to previous claims\b",
        r"\balleged that\b",
    ]

    SIMULATION_MARKERS = [
        r"\bin-silico\b",
        r"\bnumerical simulations?\b",
        r"\bcomputational fluid dynamic\b",
        r"\bidealized simulation\b",
        r"\bvirtualized testbed\b",
    ]

    HYPOTHESIS_MARKERS = [
        r"\bwe hypothesize\b",
        r"\btheoretical conjecture\b",
        r"\bfuture work will examine\b",
        r"\bmay potentially\b",
    ]

    def __init__(self, default_uncertainty: float = 0.05):
        self.default_uncertainty = default_uncertainty

    def segment_sentences_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """Segments text into sentences while recording precise (start_char, end_char) offsets."""
        sentence_spans = []
        # Match sentences ending with period, question mark, or exclamation mark followed by space or EOF
        pattern = re.compile(r"([^.!?]+[.!?]*)", re.DOTALL)
        for match in pattern.finditer(text):
            sent = match.group(1).strip()
            if len(sent) > 5:
                # Find exact start and end in text
                start = match.start()
                # Adjust start if leading whitespace
                raw_match = match.group(1)
                lead_space = len(raw_match) - len(raw_match.lstrip())
                start += lead_space
                end = start + len(sent)
                sentence_spans.append((sent, start, end))
        return sentence_spans

    def detect_polarity(self, sentence: str) -> Polarity:
        """Determines the causal polarity delta in {+1, -1, 0}."""
        lower = sentence.lower()

        # Check null / neutral patterns first
        for p in self.NULL_PATTERNS:
            if re.search(p, lower):
                return Polarity.NULL

        words = set(re.findall(r"\b[a-z]+\b", lower))

        has_neg = bool(words.intersection(self.NEGATIVE_VERBS))
        has_pos = bool(words.intersection(self.POSITIVE_VERBS))

        # Check for negation words like "not", "failed to"
        has_negation = bool(re.search(r"\b(not|no|failed|neither|nor|without)\b", lower))

        if has_neg and not has_pos:
            return Polarity.POSITIVE if has_negation else Polarity.NEGATIVE
        elif has_pos and not has_neg:
            return Polarity.NEGATIVE if has_negation else Polarity.POSITIVE
        elif has_neg and has_pos:
            # If both, prioritize the main verb closest to the outcome
            if "reduction" in words or "decreased" in words or "reduced" in words:
                return Polarity.NEGATIVE
            return Polarity.POSITIVE

        return Polarity.NULL

    def detect_epistemic_status(self, sentence: str) -> EpistemicStatus:
        """Identifies epistemic status eta."""
        lower = sentence.lower()
        for p in self.QUOTED_MARKERS:
            if re.search(p, lower):
                return EpistemicStatus.QUOTED_NOT_ENDORSED
        for p in self.SIMULATION_MARKERS:
            if re.search(p, lower):
                return EpistemicStatus.SIMULATED
        for p in self.HYPOTHESIS_MARKERS:
            if re.search(p, lower):
                return EpistemicStatus.HYPOTHESIZED
        return EpistemicStatus.OBSERVED

    def extract_context_dimensions(self, text: str) -> MaterialContext:
        """
        Extracts material scope conditions C across defined dimensions.
        Missing context is explicitly tracked.
        """
        ctx = MaterialContext()
        lower = text.lower()

        # 1. Species / System
        if any(w in lower for w in ["patient", "patients", "human", "adults", "subjects", "participants", "clinical cohort"]):
            ctx.set_dimension(ContextDimension.SPECIES_OR_SYSTEM, "Homo sapiens (humans)")
        elif any(w in lower for w in ["murine", "mice", "mouse", "c57bl/6", "rat", "rats", "animal model"]):
            ctx.set_dimension(ContextDimension.SPECIES_OR_SYSTEM, "Mus musculus / Rattus (rodents)")
        elif any(w in lower for w in ["router", "switch", "sdn", "network node", "packet", "cluster", "linux", "cloud"]):
            ctx.set_dimension(ContextDimension.SPECIES_OR_SYSTEM, "Computer network / server cluster")
        elif any(w in lower for w in ["reactor", "quadrotor", "robot", "manipulator", "aerial vehicle", "inverter"]):
            ctx.set_dimension(ContextDimension.SPECIES_OR_SYSTEM, "Mechanical / physical control plant")
        else:
            ctx.set_dimension(ContextDimension.SPECIES_OR_SYSTEM, None)

        # 2. Environment
        if "in vitro" in lower or "cell culture" in lower or "cell-free" in lower:
            ctx.set_dimension(ContextDimension.ENVIRONMENT, "in vitro")
        elif "in vivo" in lower or "clinical trial" in lower or "double-blind" in lower:
            ctx.set_dimension(ContextDimension.ENVIRONMENT, "in vivo")
        elif "datacenter" in lower or "cloud" in lower:
            ctx.set_dimension(ContextDimension.ENVIRONMENT, "datacenter infrastructure")
        elif "wind tunnel" in lower:
            ctx.set_dimension(ContextDimension.ENVIRONMENT, "wind tunnel aerodynamic testbed")
        elif "simulation" in lower:
            ctx.set_dimension(ContextDimension.ENVIRONMENT, "computational simulation environment")
        else:
            ctx.set_dimension(ContextDimension.ENVIRONMENT, None)

        # 3. Dose or Scale
        dose_match = re.search(r"(\d+(?:\.\d+)?\s*(?:mg|mg/kg|g|mcg|ml|gbps|mbps|rpm|v|a|kw))\b", lower)
        if dose_match:
            ctx.set_dimension(ContextDimension.DOSE_OR_SCALE, dose_match.group(1))
        elif "high-dose" in lower or "toxic dose" in lower:
            ctx.set_dimension(ContextDimension.DOSE_OR_SCALE, "high supra-therapeutic dose")
        elif "low-dose" in lower:
            ctx.set_dimension(ContextDimension.DOSE_OR_SCALE, "low therapeutic dose")
        else:
            ctx.set_dimension(ContextDimension.DOSE_OR_SCALE, None)

        # 4. Time Horizon
        time_match = re.search(r"(\d+\s*(?:weeks?|months?|years?|days?|hours?|milliseconds?|ms|seconds?))\b", lower)
        if time_match:
            ctx.set_dimension(ContextDimension.TIME_HORIZON, time_match.group(1))
        else:
            ctx.set_dimension(ContextDimension.TIME_HORIZON, None)

        return ctx

    def extract_mechanisms_from_text(
        self,
        abstract: str,
        doc_id: str = "doc_0",
        domain: str = "general",
        max_mechanisms: int = 10,
    ) -> List[MechanismClaim]:
        """
        Parses an abstract into a list of MechanismClaim instances q_im.
        Retains up to max_mechanisms (M_max).
        """
        sentence_spans = self.segment_sentences_with_spans(abstract)
        context = self.extract_context_dimensions(abstract)

        claims: List[MechanismClaim] = []
        mech_idx = 1

        for sent, start_char, end_char in sentence_spans:
            # Check if sentence expresses a causal claim
            pol = self.detect_polarity(sent)
            epistemic = self.detect_epistemic_status(sent)

            # Look for mediating indicators ("via", "through", "by means of", "by")
            mediators = []
            med_match = re.search(r"\b(?:via|through|by)\s+([a-zA-Z0-9\s\-_]+?)(?:\s*(?:and|,|\.|$))", sent, re.IGNORECASE)
            if med_match:
                med_phrase = med_match.group(1).strip()
                if 3 < len(med_phrase) < 60:
                    mediators.append(med_phrase)

            # Heuristic entity extraction for A, P, Y
            # Intervention: often subject or noun phrase before the causal verb
            # Outcome: object noun phrase following the causal verb
            words = sent.split()
            if len(words) < 5:
                continue

            # Identify target population/system
            pop_val = context.get_dimension(ContextDimension.SPECIES_OR_SYSTEM) or f"{domain.capitalize()} system"

            # Derive concise intervention and outcome candidates
            intervention = "Target intervention"
            outcome = "Measured target outcome"

            lower_sent = sent.lower()
            all_causal_verbs = list(self.NEGATIVE_VERBS.union(self.POSITIVE_VERBS))
            verb_positions = [(v, lower_sent.find(v)) for v in all_causal_verbs if lower_sent.find(v) != -1]

            if verb_positions:
                # Sort by earliest verb
                verb_positions.sort(key=lambda x: x[1])
                first_verb, pos = verb_positions[0]

                left_part = sent[:pos].strip(" ,;:")
                right_part = sent[pos + len(first_verb):].strip(" ,;:")

                if left_part:
                    # Clean left part to extract core intervention
                    intervention = " ".join(left_part.split()[-5:])
                if right_part:
                    # Clean right part to extract core outcome
                    outcome_words = right_part.split()[:6]
                    outcome = " ".join(outcome_words).strip(" ,;.")

            provenance = ProvenanceSpan(
                start_char=start_char,
                end_char=end_char,
                text=sent,
                section="Abstract",
            )

            # Uncertainty estimate based on epistemic markers and sentence length
            uncertainty = self.default_uncertainty
            if epistemic == EpistemicStatus.QUOTED_NOT_ENDORSED:
                uncertainty = 0.35
            elif epistemic == EpistemicStatus.HYPOTHESIZED:
                uncertainty = 0.25

            claim = MechanismClaim(
                mechanism_id=f"{doc_id}_m{mech_idx}",
                context=context,
                population=pop_val,
                intervention=intervention,
                mediating_steps=mediators,
                outcome=outcome,
                time_horizon=context.get_dimension(ContextDimension.TIME_HORIZON),
                polarity=pol,
                epistemic_status=epistemic,
                uncertainty=uncertainty,
                provenance=provenance,
            )
            claims.append(claim)
            mech_idx += 1

            if len(claims) >= max_mechanisms:
                break

        return claims

    def extract_document_graph(self, doc: DocumentRecord, max_mechanisms: int = 10) -> DocumentGraph:
        """
        Extracts mechanisms and compiles them directly into a DocumentGraph Gi.
        """
        mechanisms = self.extract_mechanisms_from_text(
            abstract=doc.abstract,
            doc_id=doc.doc_id,
            domain=doc.domain,
            max_mechanisms=max_mechanisms,
        )

        return DocumentGraph.from_mechanisms(
            doc_id=doc.doc_id,
            title=doc.title,
            abstract=doc.abstract,
            domain=doc.domain,
            mechanisms=mechanisms,
            max_mechanisms=max_mechanisms,
        )


if __name__ == "__main__":
    sample_text = (
        "In a cohort of hypertensive adults, Amlodipine 5mg oral once daily reduced mean systolic blood pressure "
        "by 12 mmHg via vascular peripheral resistance decrease after 4 weeks. "
        "Prior literature claimed that calcium channel blockers worsened pedal edema; however, our study found no significant adverse effect."
    )
    extractor = MultiMechanismExtractor()
    claims = extractor.extract_mechanisms_from_text(sample_text, doc_id="TEST_001", domain="biomedical")
    print(f"Extracted {len(claims)} mechanisms:")
    for c in claims:
        print(f"  - [{c.polarity.name}] {c.summary()} (chars: {c.provenance.start_char}-{c.provenance.end_char})")
