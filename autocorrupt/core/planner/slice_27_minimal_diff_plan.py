"""
Slice 27: Minimal-Span Screen Diff Planner.
Given a winning MutationHypo targeting any slot(s) on the SentenceTape (whether immediate
at slot `k` or retroactive 5 words back at slot `i`), computes the exact minimal number of
backspaces and the exact replacement string to inject—trimming any shared character prefix
so the engine never deletes unchanged characters.
"""
from __future__ import annotations
import time
from autocorrupt.core.atoms.slice_00_types import MutationHypo, RewriteAction
from autocorrupt.core.syntax.slice_11_sentence_tape import SentenceTape


def build_minimal_rewrite_action(
    tape: SentenceTape,
    hypo: MutationHypo,
    is_retroactive: bool,
    start_time_ns: int = 0,
) -> RewriteAction:
    """
    Applies `hypo` to `tape` in place and returns the minimal `RewriteAction(backspaces, inject_text)`
    needed to transform the user's current screen text into the mutated sentence.
    """
    first_slot = hypo.target_slots[0]
    last_slot = hypo.target_slots[-1]

    # 1. Capture the exact screen text currently sitting from `first_slot` to the cursor
    old_screen_tail = tape.span_screen_text(first_slot)

    # 2. Mutate the target slots on the SentenceTape
    if len(hypo.target_slots) == len(hypo.replacement_words):
        for slot_i, new_w in zip(hypo.target_slots, hypo.replacement_words):
            tok = tape.tokens[slot_i]
            tok.current_text = new_w
            tok.mutated_count += 1
            tok.applied_solver = hypo.solver_name
            tok.last_utility_score = hypo.total_utility
    else:
        # Multi-token span collapsed or expanded into a single phrase (e.g., "lamp" + "on" -> "tampon")
        head_tok = tape.tokens[first_slot]
        tail_sep = tape.tokens[last_slot].trailing_sep
        head_tok.current_text = " ".join(hypo.replacement_words)
        head_tok.trailing_sep = tail_sep
        head_tok.mutated_count += 1
        head_tok.applied_solver = hypo.solver_name
        head_tok.last_utility_score = hypo.total_utility
        # Clear intermediate collapsed slots' visible text so indices stay stable or collapse cleanly
        for mid_i in hypo.target_slots[1:]:
            tape.tokens[mid_i].current_text = ""
            tape.tokens[mid_i].trailing_sep = ""
            tape.tokens[mid_i].locked = True

    # 3. Capture the new screen tail from `first_slot` to the cursor
    new_screen_tail = tape.span_screen_text(first_slot)

    # 4. Strip longest common prefix so we only backspace from the first changed character!
    common_len = 0
    max_common = min(len(old_screen_tail), len(new_screen_tail))
    while common_len < max_common and old_screen_tail[common_len] == new_screen_tail[common_len]:
        common_len += 1

    backspaces_needed = len(old_screen_tail) - common_len
    inject_str = new_screen_tail[common_len:]

    elapsed_us = (
        (time.perf_counter_ns() - start_time_ns) / 1000.0
        if start_time_ns > 0
        else 0.0
    )

    return RewriteAction(
        backspaces=backspaces_needed,
        inject_text=inject_str,
        target_slots=hypo.target_slots,
        solver_name=hypo.solver_name,
        utility_score=hypo.total_utility,
        is_retroactive=is_retroactive,
        trace=hypo.explanation_trace,
        latency_us=round(elapsed_us, 2),
    )
