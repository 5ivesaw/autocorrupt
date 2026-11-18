"""
Slice 25: Global Deterministic Utility Scorer & Cooldown Ring Buffer.
Evaluates all candidate MutationHypos produced by Slices 19–24 using a single deterministic
objective equation and enforces a 16-entry circular cooldown filter to prevent repetitive gags.
"""
from __future__ import annotations
from collections import deque
from dataclasses import replace
from autocorrupt.core.atoms.slice_00_types import DomainMask, MutationHypo, POS
from autocorrupt.core.perception.slice_09_mmap_trie import MmapLexiconStore
from autocorrupt.core.solvers.slice_19_solve_antipodal_sense import solve_antipodal_sense
from autocorrupt.core.solvers.slice_20_solve_pos_cross import solve_pos_cross
from autocorrupt.core.solvers.slice_21_solve_acoustic_beam import solve_acoustic_beam
from autocorrupt.core.solvers.slice_22_solve_minimal_pair import solve_minimal_pair
from autocorrupt.core.solvers.slice_23_solve_register_projection import solve_register_projection
from autocorrupt.core.solvers.slice_24_solve_spoonerism_swap import solve_spoonerism_swap
from autocorrupt.core.syntax.slice_11_sentence_tape import SentenceTape

CLOSED_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "to", "in", "on", "at", "for", "with", "by", "from", "of",
    "and", "or", "but", "so", "i", "you", "he", "she", "it", "we", "they", "me",
    "my", "your", "his", "her", "our", "their", "is", "am", "are", "was", "were",
    "be", "been", "will", "would", "can", "could", "this", "that", "these", "those",
    "im", "i'm", "just", "some", "if", "what", "when", "where", "who", "how",
})


class UtilityOptimizer:
    """
    Deterministic GOFAI utility scorer.
    Objective: Utility = 0.34 * FormTightness + 0.22 * GrammarFit + 0.44 * CollisionScore - CooldownPenalty
    """

    def __init__(self, min_utility_threshold: float = 68.0) -> None:
        self.min_utility_threshold = min_utility_threshold
        self.recent_replacements: deque[str] = deque(maxlen=16)

    def gather_sentence_domains(self, tape: SentenceTape) -> DomainMask:
        mask = DomainMask.NONE
        for tok in tape.tokens:
            mask |= tok.meaning_vec.domains
        return mask

    def evaluate_slot(
        self,
        tape: SentenceTape,
        slot_idx: int,
        store: MmapLexiconStore,
        sentence_domains: DomainMask,
    ) -> list[MutationHypo]:
        if slot_idx < 0 or slot_idx >= len(tape.tokens):
            return []
        tok = tape.tokens[slot_idx]
        w_low = tok.raw_typed.lower()

        raw_hypos: list[MutationHypo] = []

        raw_hypos.extend(solve_minimal_pair(tape, slot_idx, store, sentence_domains))
        raw_hypos.extend(solve_spoonerism_swap(tape, slot_idx, store, sentence_domains))
        raw_hypos.extend(solve_acoustic_beam(tape, slot_idx, store, sentence_domains))

        if w_low not in CLOSED_STOPWORDS and tok.pos not in (POS.DET, POS.PREP, POS.CONJ, POS.PRON):
            raw_hypos.extend(solve_antipodal_sense(tape, slot_idx, store, sentence_domains))
            raw_hypos.extend(solve_pos_cross(tape, slot_idx, store, sentence_domains))
            raw_hypos.extend(solve_register_projection(tape, slot_idx, store, sentence_domains))
        else:
            raw_hypos = [h for h in raw_hypos if len(h.target_slots) > 1]

        scored: list[MutationHypo] = []
        for h in raw_hypos:
            rep_key = " ".join(h.replacement_words).lower()
            cooldown_pen = 18.0 if rep_key in self.recent_replacements else 0.0
            # Multi-word structural corruptions (mondegreens, spoonerisms, junctures) get a structural synergy bonus
            span_bonus = (
                7.5
                if (len(h.target_slots) > 1 or h.solver_name.startswith("slice_21_acoustic_mondegreen"))
                else 0.0
            )
            util = (
                0.34 * h.form_tightness
                + 0.22 * h.grammar_fit
                + 0.44 * h.collision_score
                + span_bonus
                - cooldown_pen
            )
            scored.append(replace(h, total_utility=round(util, 3)))

        scored.sort(key=lambda x: x.total_utility, reverse=True)
        return scored

    def record_used(self, hypo: MutationHypo) -> None:
        self.recent_replacements.append(" ".join(hypo.replacement_words).lower())
