"""
Slice 13: Deterministic Left-Corner / Shift-Reduce Dependency Parser.
Constructs live syntactic dependency arcs (SUBJ_VERB, VERB_OBJ, MOD_NOUN, VERB_MOD)
across the SentenceTape so that when a later word is typed (e.g., "kids" after "auditioning",
"chicken" after "polenta", or "car" after "park"), the engine knows which earlier words
on the tape are syntactically linked to the new word.
"""
from __future__ import annotations
from autocorrupt.core.atoms.slice_00_types import DependencyArc, POS
from autocorrupt.core.syntax.slice_11_sentence_tape import SentenceTape


def parse_dependencies(tape: SentenceTape, window: int = 12) -> list[DependencyArc]:
    """
    Extracts directed DependencyArcs across the active sentence window.
    Runs in O(W) time (< 15 microseconds).
    """
    tokens = tape.tokens
    n = len(tokens)
    if n < 2:
        return []

    start = max(0, n - window)
    arcs: list[DependencyArc] = []

    last_verb_idx: int = -1
    last_subj_idx: int = -1
    pending_modifiers: list[int] = []

    for i in range(start, n):
        tok = tokens[i]
        p = tok.pos
        w_low = tok.raw_typed.lower()

        if p == POS.DET:
            pending_modifiers.append(i)
            continue

        if p == POS.ADJ:
            pending_modifiers.append(i)
            if last_verb_idx >= start and i == last_verb_idx + 1:
                arcs.append(DependencyArc(head_slot=last_verb_idx, dep_slot=i, rel_type="VERB_MOD"))
            continue

        if p in (POS.PRON, POS.NOUN):
            # Attach any preceding DET/ADJ or noun-adjuncts to this noun
            for mod_idx in pending_modifiers:
                if mod_idx != i:
                    arcs.append(DependencyArc(head_slot=i, dep_slot=mod_idx, rel_type="MOD_NOUN"))
            # Check if previous token was also a NOUN (compound noun adjunct, e.g. "Darth Vader pillow")
            if i > start and tokens[i - 1].pos == POS.NOUN:
                arcs.append(DependencyArc(head_slot=i, dep_slot=i - 1, rel_type="MOD_NOUN"))
            pending_modifiers.clear()

            if last_verb_idx < start:
                last_subj_idx = i
            else:
                # Object of the active verb (direct object or prepositional object)
                arcs.append(DependencyArc(head_slot=last_verb_idx, dep_slot=i, rel_type="VERB_OBJ"))
            continue

        if p == POS.VERB and w_low not in ("am", "is", "are", "was", "were", "will", "can"):
            pending_modifiers.clear()
            if last_subj_idx >= start:
                arcs.append(DependencyArc(head_slot=i, dep_slot=last_subj_idx, rel_type="SUBJ_VERB"))
            last_verb_idx = i
            continue

        if p in (POS.ADV, POS.PREP):
            if last_verb_idx >= start:
                arcs.append(DependencyArc(head_slot=last_verb_idx, dep_slot=i, rel_type="VERB_MOD"))

    return arcs


def get_retro_linked_slots(tape: SentenceTape, current_slot: int, arcs: list[DependencyArc]) -> list[int]:
    """
    Given the newly committed `current_slot`, returns all earlier slot indices on the tape
    that share a syntactic dependency arc or clause neighborhood with `current_slot`.
    """
    linked: set[int] = set()
    for arc in arcs:
        if arc.dep_slot == current_slot and arc.head_slot < current_slot:
            linked.add(arc.head_slot)
        elif arc.head_slot == current_slot and arc.dep_slot < current_slot:
            linked.add(arc.dep_slot)

    # Also include content words (NOUN, VERB, ADJ, ADV) within a 5-token clause window
    # so cross-clause domain synergy (e.g., "polenta ... gluten ... chicken") propagates backward.
    start = max(0, current_slot - 6)
    for idx in range(start, current_slot):
        if tape.tokens[idx].pos in (POS.NOUN, POS.VERB, POS.ADJ, POS.ADV):
            linked.add(idx)

    return sorted(linked, reverse=True)
