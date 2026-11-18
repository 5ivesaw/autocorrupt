"""
Slice 26: Retrospective "Time-Travel" Back-Propagation Planner.
Whenever a new token `k` is committed on the SentenceTape:
1. Updates Viterbi POS tags across the sentence (Slice 12).
2. Parses live syntactic DependencyArcs (Slice 13).
3. Evaluates BOTH the new token `k` AND earlier linked tokens `i < k` using the updated
   sentence domain mask and POS tags.
4. Enforces a Clause Density Governor so a sentence gets surgical, high-impact corruptions
   (and upgrades an earlier corruption or triggers a time-travel edit when a later word unlocks
   a significantly higher utility score).
"""
from __future__ import annotations
from autocorrupt.core.atoms.slice_00_types import MutationHypo
from autocorrupt.core.perception.slice_09_mmap_trie import MmapLexiconStore
from autocorrupt.core.planner.slice_25_utility_scorer import (
    CLOSED_STOPWORDS,
    UtilityOptimizer,
)
from autocorrupt.core.syntax.slice_11_sentence_tape import SentenceTape
from autocorrupt.core.syntax.slice_12_viterbi_pos import update_pos_tags
from autocorrupt.core.syntax.slice_13_shift_reduce_parser import (
    get_retro_linked_slots,
    parse_dependencies,
)


class RetroBackpropPlanner:
    """
    Orchestrates immediate vs. retroactive ("Time-Travel") corruption selection.
    """

    def __init__(
        self,
        optimizer: UtilityOptimizer | None = None,
        retro_improvement_delta: float = 5.0,
        defer_ambiguous_until_context: bool = True,
        clause_cooldown_slots: int = 4,
    ) -> None:
        self.optimizer = optimizer or UtilityOptimizer(min_utility_threshold=74.0)
        self.retro_improvement_delta = retro_improvement_delta
        self.defer_ambiguous_until_context = defer_ambiguous_until_context
        self.clause_cooldown_slots = clause_cooldown_slots

    def _recent_mutation_max_utility(
        self,
        tape: SentenceTape,
        curr_idx: int,
        exclude_slots: tuple[int, ...] = (),
    ) -> float:
        """Returns the highest utility of any already-mutated token within `clause_cooldown_slots`."""
        start = max(0, curr_idx - self.clause_cooldown_slots)
        best = 0.0
        for i in range(start, curr_idx):
            if i in exclude_slots:
                continue
            t = tape.tokens[i]
            if t.mutated_count > 0 and t.last_utility_score > best:
                best = t.last_utility_score
        return best

    def step_on_word_commit(
        self,
        tape: SentenceTape,
        store: MmapLexiconStore,
    ) -> tuple[MutationHypo | None, bool]:
        """
        Called immediately when the user finishes typing a word (Space / punctuation).
        Returns `(winning_hypothesis, is_retroactive)`.
        """
        n = len(tape.tokens)
        if n == 0:
            return (None, False)
        curr_idx = n - 1
        curr_tok = tape.tokens[curr_idx]
        curr_low = curr_tok.raw_typed.lower()

        # Step 1: Update grammatical POS tags across the active window
        update_pos_tags(tape, store, window=8)

        # Step 2: Build dependency arcs and aggregate active sentence domains
        arcs = parse_dependencies(tape, window=12)
        sentence_domains = self.optimizer.gather_sentence_domains(tape)

        # Step 3: Evaluate the newly committed slot `curr_idx` (including multi-token junctures/spoonerisms)
        curr_hypos = self.optimizer.evaluate_slot(tape, curr_idx, store, sentence_domains)
        best_curr_hypo: MutationHypo | None = None
        if curr_hypos and curr_hypos[0].total_utility >= self.optimizer.min_utility_threshold:
            best_curr_hypo = curr_hypos[0]
            if best_curr_hypo.target_slots[0] < curr_idx:
                return (best_curr_hypo, True)

        # Step 4: Check retroactive ("Time-Travel") back-propagation on earlier linked tokens
        best_retro_hypo: MutationHypo | None = None
        if curr_low not in CLOSED_STOPWORDS or int(curr_tok.meaning_vec.domains) != 0:
            retro_slots = get_retro_linked_slots(tape, curr_idx, arcs)
            for r_idx in retro_slots:
                r_tok = tape.tokens[r_idx]
                if r_tok.locked or r_tok.mutated_count >= 1:
                    continue
                r_hypos = self.optimizer.evaluate_slot(tape, r_idx, store, sentence_domains)
                if not r_hypos:
                    continue
                top_r = r_hypos[0]
                if top_r.total_utility < self.optimizer.min_utility_threshold:
                    continue
                if len(top_r.target_slots) == 1 and top_r.replacement_words[0] == r_tok.current_text:
                    continue

                if top_r.total_utility >= r_tok.last_utility_score + self.retro_improvement_delta:
                    if best_retro_hypo is None or top_r.total_utility > best_retro_hypo.total_utility:
                        best_retro_hypo = top_r

        recent_max_util = self._recent_mutation_max_utility(tape, curr_idx)

        if best_retro_hypo is not None:
            # Only fire a single-slot retroactive edit if no nearby token in the same 4-word clause was already mutated
            if recent_max_util == 0.0 or len(best_retro_hypo.target_slots) > 1:
                if best_curr_hypo is None or best_retro_hypo.total_utility + 3.0 >= best_curr_hypo.total_utility:
                    return (best_retro_hypo, True)

        if best_curr_hypo is not None:
            if recent_max_util > 0.0 and len(best_curr_hypo.target_slots) == 1:
                return (None, False)

            other_domains = 0
            for i in range(curr_idx):
                other_domains |= int(tape.tokens[i].meaning_vec.domains)

            entry = store.get_entry(curr_tok.lemma)
            is_polysemous = entry is not None and len(entry.synset_ids) >= 2
            if (
                self.defer_ambiguous_until_context
                and is_polysemous
                and other_domains == 0
                and curr_tok.trailing_sep == " "
                and curr_idx <= 2
                and best_curr_hypo.total_utility < 91.0
            ):
                curr_tok.last_utility_score = best_curr_hypo.total_utility - 8.0
                return (None, False)

            return (best_curr_hypo, False)

        return (None, False)
