"""
Slice 20: Cross-POS (Noun <-> Verb) & Capitonym Flipper Solver.
Identifies words that exist across multiple Parts of Speech in the lexicon (e.g., "book",
"park", "season", "lead", "wind", "polish", "water"), locks onto the synset belonging to
the WRONG Part of Speech (opposite to the Viterbi POS tag on the SentenceTape), and pulls
its primary concrete synonym into the current grammatical slot.
"""
from __future__ import annotations
from autocorrupt.core.atoms.slice_00_types import DomainMask, MutationHypo
from autocorrupt.core.atoms.slice_03_case_prosody import apply_case_profile
from autocorrupt.core.morphology.slice_05_conjugator import conjugate
from autocorrupt.core.perception.slice_09_mmap_trie import MmapLexiconStore
from autocorrupt.core.semantics.slice_18_domain_collision import compute_domain_collision
from autocorrupt.core.syntax.slice_11_sentence_tape import SentenceTape


def solve_pos_cross(
    tape: SentenceTape,
    slot_idx: int,
    store: MmapLexiconStore,
    sentence_domains: DomainMask,
) -> list[MutationHypo]:
    tok = tape.tokens[slot_idx]
    entry = store.get_entry(tok.lemma) or store.get_entry(tok.raw_typed.lower())
    if entry is None or len(entry.synset_ids) < 2:
        return []

    active_syn = None
    cross_syns = []
    for sid in entry.synset_ids:
        syn = store.synsets[sid]
        if syn.pos == tok.pos and active_syn is None:
            active_syn = syn
        elif syn.pos != tok.pos:
            cross_syns.append(syn)

    if active_syn is None or not cross_syns:
        return []

    hypos: list[MutationHypo] = []
    for wrong_pos_syn in cross_syns:
        collision = compute_domain_collision(
            active_syn.meaning_vec,
            wrong_pos_syn.meaning_vec,
            sentence_domains,
        )
        # Cross-POS polysemy is an exact surface-word match (100% orthographic fidelity to original root)
        collision = min(100.0, collision + 32.0)
        for cand_lemma in wrong_pos_syn.lemmas:
            if cand_lemma.lower() == tok.lemma.lower():
                continue
            inflected = conjugate(cand_lemma, tok.inflection, tok.prefix)
            final_word = apply_case_profile(inflected, tok.case_profile, tok.raw_typed)
            hypos.append(
                MutationHypo(
                    target_slots=(slot_idx,),
                    replacement_words=(final_word,),
                    solver_name="slice_20_pos_cross",
                    form_tightness=97.0,
                    grammar_fit=95.0,
                    collision_score=collision,
                    total_utility=0.0,
                    explanation_trace=(
                        f"Cross-POS flip on '{tok.raw_typed}' (active {tok.pos.name} -> "
                        f"wrong {wrong_pos_syn.pos.name} synset '{wrong_pos_syn.gloss}') -> '{final_word}'"
                    ),
                )
            )
            break

    return hypos
