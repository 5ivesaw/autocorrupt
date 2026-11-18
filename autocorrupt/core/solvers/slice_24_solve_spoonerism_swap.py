"""
Slice 24: Phonological Onset-Rime Spoonerism Solver.
Scans content words within a 4-token window on the SentenceTape (e.g., "shake a tower"),
splits each word into (initial_onset_cluster, rime) via Slice 08, swaps their onsets,
and checks the MmapLexiconStore rime index in O(1) time to see if both swapped phoneme
sequences are real English dictionary words ("take a shower").
"""
from __future__ import annotations
from autocorrupt.core.atoms.slice_00_types import DomainMask, MutationHypo, POS
from autocorrupt.core.atoms.slice_03_case_prosody import apply_case_profile
from autocorrupt.core.perception.slice_08_syllable_nuclei import split_onset_rime
from autocorrupt.core.perception.slice_09_mmap_trie import MmapLexiconStore
from autocorrupt.core.semantics.slice_18_domain_collision import compute_domain_collision
from autocorrupt.core.syntax.slice_11_sentence_tape import SentenceTape


def solve_spoonerism_swap(
    tape: SentenceTape,
    slot_idx: int,
    store: MmapLexiconStore,
    sentence_domains: DomainMask,
) -> list[MutationHypo]:
    if slot_idx < 1 or slot_idx >= len(tape.tokens):
        return []

    t2 = tape.tokens[slot_idx]
    if not t2.phonemes or t2.pos in (POS.DET, POS.PREP, POS.CONJ, POS.PRON):
        return []

    onset2, rime2 = split_onset_rime(t2.phonemes)
    if not onset2 or not rime2:
        return []

    hypos: list[MutationHypo] = []
    start_search = max(0, slot_idx - 3)

    for i in range(start_search, slot_idx):
        t1 = tape.tokens[i]
        if not t1.phonemes or t1.pos in (POS.DET, POS.PREP, POS.CONJ, POS.PRON):
            continue
        onset1, rime1 = split_onset_rime(t1.phonemes)
        if not onset1 or not rime1 or onset1 == onset2:
            continue

        # Check O(1) rime_index: does (onset2 + rime1) exist AND (onset1 + rime2) exist?
        wid1_swapped = store.rime_index.get(rime1, {}).get(onset2, -1)
        wid2_swapped = store.rime_index.get(rime2, {}).get(onset1, -1)

        if wid1_swapped >= 0 and wid2_swapped >= 0:
            e1 = store.entries[wid1_swapped]
            e2 = store.entries[wid2_swapped]
            w1_final = apply_case_profile(e1.word, t1.case_profile, t1.raw_typed)
            w2_final = apply_case_profile(e2.word, t2.case_profile, t2.raw_typed)

            # Reconstruct the full span from slot `i` through `slot_idx`
            span_slots = tuple(range(i, slot_idx + 1))
            mid_words = [tape.tokens[m].current_text for m in range(i + 1, slot_idx)]
            rep_words = (w1_final, *mid_words, w2_final)

            coll1 = compute_domain_collision(t1.meaning_vec, e1.primary_meaning, sentence_domains)
            coll2 = compute_domain_collision(t2.meaning_vec, e2.primary_meaning, sentence_domains)
            comb_coll = min(100.0, max(coll1, coll2) + 28.0)

            hypos.append(
                MutationHypo(
                    target_slots=span_slots,
                    replacement_words=rep_words,
                    solver_name="slice_24_spoonerism",
                    form_tightness=95.0,
                    grammar_fit=94.0,
                    collision_score=comb_coll,
                    total_utility=0.0,
                    explanation_trace=(
                        f"Phonological onset-rime spoonerism across slots {span_slots}: "
                        f"'{t1.raw_typed} ... {t2.raw_typed}' -> '{w1_final} ... {w2_final}'"
                    ),
                )
            )

    return hypos
