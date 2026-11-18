"""
Slice 19: Antipodal Word-Sense Synonym Solver (Polysemy Traps).
Finds the user's intended WordNet sense via Lesk (Slice 16), then searches sibling senses
of the exact same surface word that share the same POS but have maximum semantic/domain
distance, and conjugates a concrete synonym from that wrong sense (Slice 05).
"""
from __future__ import annotations
from autocorrupt.core.atoms.slice_00_types import DomainMask, MutationHypo
from autocorrupt.core.atoms.slice_03_case_prosody import apply_case_profile
from autocorrupt.core.morphology.slice_05_conjugator import conjugate
from autocorrupt.core.perception.slice_09_mmap_trie import MmapLexiconStore
from autocorrupt.core.semantics.slice_16_lesk_sense_detector import detect_intended_sense
from autocorrupt.core.semantics.slice_18_domain_collision import compute_domain_collision
from autocorrupt.core.syntax.slice_11_sentence_tape import SentenceTape


def solve_antipodal_sense(
    tape: SentenceTape,
    slot_idx: int,
    store: MmapLexiconStore,
    sentence_domains: DomainMask,
) -> list[MutationHypo]:
    tok = tape.tokens[slot_idx]
    entry = store.get_entry(tok.lemma) or store.get_entry(tok.raw_typed.lower())
    if entry is None or len(entry.synset_ids) < 2:
        return []

    intended_sid = detect_intended_sense(tape, slot_idx, store)
    if intended_sid < 0:
        return []
    intended_syn = store.synsets[intended_sid]

    hypos: list[MutationHypo] = []
    for sid in entry.synset_ids:
        if sid == intended_sid:
            continue
        wrong_syn = store.synsets[sid]
        if wrong_syn.pos != tok.pos:
            continue

        collision = compute_domain_collision(
            intended_syn.meaning_vec,
            wrong_syn.meaning_vec,
            sentence_domains,
        )
        # Boost polysemy traps when the intended sense was strongly disambiguated by context
        collision = min(100.0, collision + 26.0)

        for cand_lemma in wrong_syn.lemmas:
            if cand_lemma.lower() == tok.lemma.lower() or cand_lemma.lower() == tok.raw_typed.lower():
                continue
            inflected = conjugate(cand_lemma, tok.inflection, tok.prefix)
            final_word = apply_case_profile(inflected, tok.case_profile, tok.raw_typed)
            hypos.append(
                MutationHypo(
                    target_slots=(slot_idx,),
                    replacement_words=(final_word,),
                    solver_name="slice_19_antipodal_sense",
                    form_tightness=92.0,
                    grammar_fit=98.0,
                    collision_score=collision,
                    total_utility=0.0,
                    explanation_trace=(
                        f"Polysemy sense flip on '{tok.raw_typed}': intended sense "
                        f"({intended_syn.gloss}) -> antipodal sense ({wrong_syn.gloss}) -> '{final_word}'"
                    ),
                )
            )
            break

    return hypos
