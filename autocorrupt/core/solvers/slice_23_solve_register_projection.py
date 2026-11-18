"""
Slice 23: 6-Axis Clinical, Visceral & Dysphemistic Register Projector.
Walks the intended SynsetNode of a word and projects it onto its highest-clinical or
highest-visceral synonym (e.g., touch -> fondle, eat -> ingest, friend -> bedfellow,
wet -> moist, finish -> climax, kiss -> exchange oral mucosa, butt -> gluteal cleft)
while also checking contranyms/antonyms (definitely -> reluctantly, BS -> Birdseed).
"""
from __future__ import annotations
from autocorrupt.core.atoms.slice_00_types import DomainMask, MutationHypo
from autocorrupt.core.atoms.slice_03_case_prosody import apply_case_profile
from autocorrupt.core.morphology.slice_05_conjugator import conjugate
from autocorrupt.core.perception.slice_09_mmap_trie import MmapLexiconStore
from autocorrupt.core.semantics.slice_16_lesk_sense_detector import detect_intended_sense
from autocorrupt.core.semantics.slice_17_connotation_axes import compute_meaning_vec_from_gloss
from autocorrupt.core.semantics.slice_18_domain_collision import compute_domain_collision
from autocorrupt.core.syntax.slice_11_sentence_tape import SentenceTape


def solve_register_projection(
    tape: SentenceTape,
    slot_idx: int,
    store: MmapLexiconStore,
    sentence_domains: DomainMask,
) -> list[MutationHypo]:
    tok = tape.tokens[slot_idx]
    entry = store.get_entry(tok.lemma) or store.get_entry(tok.raw_typed.lower())
    if entry is None or not entry.synset_ids:
        return []

    sid = detect_intended_sense(tape, slot_idx, store)
    if sid < 0:
        sid = entry.synset_ids[0]
    syn = store.synsets[sid]

    hypos: list[MutationHypo] = []
    for cand_lemma in syn.lemmas:
        if cand_lemma.lower() == tok.lemma.lower() or cand_lemma.lower() == tok.raw_typed.lower():
            continue
        cand_entry = store.get_entry(cand_lemma)
        cand_vec = (
            cand_entry.primary_meaning
            if cand_entry
            else compute_meaning_vec_from_gloss(cand_lemma, syn.gloss + " " + cand_lemma)
        )
        # Compute specific register spike of the candidate word itself
        word_self_vec = compute_meaning_vec_from_gloss(cand_lemma, syn.gloss + " " + cand_lemma)
        coll = compute_domain_collision(tok.meaning_vec, word_self_vec, sentence_domains)
        register_spike = (word_self_vec.clinical + word_self_vec.visceral + abs(word_self_vec.formality - 50)) * 0.45
        coll = min(100.0, coll + register_spike)

        inflected = conjugate(cand_lemma, tok.inflection, tok.prefix)
        final_word = apply_case_profile(inflected, tok.case_profile, tok.raw_typed)
        hypos.append(
            MutationHypo(
                target_slots=(slot_idx,),
                replacement_words=(final_word,),
                solver_name="slice_23_register_projection",
                form_tightness=88.0,
                grammar_fit=99.0,
                collision_score=coll,
                total_utility=0.0,
                explanation_trace=(
                    f"6-axis register projection within synset '{syn.gloss}': "
                    f"'{tok.raw_typed}' -> '{final_word}' (clinical={word_self_vec.clinical}, visceral={word_self_vec.visceral})"
                ),
            )
        )

    # Also evaluate contranym / gaslight antonym flips if defined on the synset
    for ant_lemma in syn.antonym_lemmas:
        if ant_lemma.lower() in ("reluctantly", "birdseed", "boycott", "negligence", "tolerate"):
            inflected = conjugate(ant_lemma, tok.inflection, tok.prefix)
            final_word = apply_case_profile(inflected, tok.case_profile, tok.raw_typed)
            hypos.append(
                MutationHypo(
                    target_slots=(slot_idx,),
                    replacement_words=(final_word,),
                    solver_name="slice_23_contranym_gaslight",
                    form_tightness=86.0,
                    grammar_fit=96.0,
                    collision_score=85.0,
                    total_utility=0.0,
                    explanation_trace=f"Contranym / bowdlerization flip: '{tok.raw_typed}' -> '{final_word}'",
                )
            )

    return hypos
