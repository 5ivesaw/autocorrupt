"""
Slice 22: Minimal-Pair, QWERTY Fat-Finger & Cross-Word Oronym Juncture Solver.
Intersects the SymSpell Levenshtein-1 Automaton (Slice 10) and Keyboard Geometry (Slice 02)
across both single words (e.g., shirt -> shit, public -> pubic, regards -> retards, sec -> sex)
and fused multi-word junctures (e.g., "lamp" + "on" -> "tampon", "for" + "the" + "play" -> "foreplay").
"""
from __future__ import annotations
from autocorrupt.core.atoms.slice_00_types import DomainMask, MutationHypo
from autocorrupt.core.atoms.slice_03_case_prosody import apply_case_profile
from autocorrupt.core.perception.slice_09_mmap_trie import MmapLexiconStore
from autocorrupt.core.perception.slice_10_levenshtein_automaton import find_orthographic_neighbors
from autocorrupt.core.semantics.slice_18_domain_collision import compute_domain_collision
from autocorrupt.core.syntax.slice_11_sentence_tape import SentenceTape
from autocorrupt.core.syntax.slice_14_juncture_window import extract_juncture_spans

SAFE_FRAME_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "to", "in", "on", "at", "for", "with", "by", "from", "of",
    "and", "or", "but", "so", "i", "you", "he", "she", "it", "we", "they", "me",
    "my", "your", "his", "her", "our", "their", "is", "am", "are", "was", "were",
    "will", "would", "can", "could", "this", "that", "well", "stop", "hate", "need",
    "some", "if", "what", "when", "where", "who", "how", "just", "considered",
})


def solve_minimal_pair(
    tape: SentenceTape,
    slot_idx: int,
    store: MmapLexiconStore,
    sentence_domains: DomainMask,
) -> list[MutationHypo]:
    tok = tape.tokens[slot_idx]
    w_low = tok.raw_typed.lower()
    hypos: list[MutationHypo] = []

    spans = extract_juncture_spans(tape, end_slot=slot_idx)
    for span in spans:
        is_multi = len(span.slot_indices) > 1
        if not is_multi and (len(w_low) < 3 or w_low in SAFE_FRAME_WORDS):
            continue
        if is_multi and len(span.fused_ortho) < 5:
            continue

        # Check exact fused match (0 edits after space removal)
        exact_entry = store.get_entry(span.fused_ortho)
        if is_multi and exact_entry is not None:
            coll = compute_domain_collision(tok.meaning_vec, exact_entry.primary_meaning, sentence_domains)
            if coll >= 45.0:
                hypos.append(
                    MutationHypo(
                        target_slots=span.slot_indices,
                        replacement_words=(exact_entry.word,),
                        solver_name="slice_22_juncture_fusion",
                        form_tightness=96.0,
                        grammar_fit=92.0,
                        collision_score=min(100.0, coll + 22.0),
                        total_utility=0.0,
                        explanation_trace=f"Oronym juncture fusion '{' '.join(span.raw_words)}' -> '{exact_entry.word}'",
                    )
                )

        neighbors = find_orthographic_neighbors(span.fused_ortho, store)
        for nb in neighbors:
            # For multi-word juncture spans (e.g. "lamp on" -> "tampon", "for the play" -> "foreplay"),
            # only allow high-visceral/clinical target words so normal word pairs don't fuse randomly
            cand_entry = store.entries[nb.word_id]
            if is_multi and (cand_entry.primary_meaning.visceral + cand_entry.primary_meaning.clinical) < 25:
                continue
            if not is_multi and nb.edit_type == "sub_1" and (cand_entry.primary_meaning.visceral < 30):
                # Prefer true QWERTY key_slip, delete_1, insert_1, or transpose for single-word orthographic traps
                continue

            coll = compute_domain_collision(tok.meaning_vec, cand_entry.primary_meaning, sentence_domains)
            if coll < 45.0:
                continue
            if is_multi:
                coll = min(100.0, coll + 25.0)
                final_w = apply_case_profile(nb.word, tape.tokens[span.slot_indices[0]].case_profile)
            else:
                final_w = apply_case_profile(nb.word, tok.case_profile, tok.raw_typed)

            grammar_fit = 94.0 if (tok.pos in cand_entry.pos_tags or is_multi) else 80.0
            hypos.append(
                MutationHypo(
                    target_slots=span.slot_indices,
                    replacement_words=(final_w,),
                    solver_name=f"slice_22_{nb.edit_type}",
                    form_tightness=nb.form_similarity,
                    grammar_fit=grammar_fit,
                    collision_score=coll,
                    total_utility=0.0,
                    explanation_trace=(
                        f"Orthographic {nb.edit_type} on '{' '.join(span.raw_words)}' -> '{final_w}' "
                        f"(sim={nb.form_similarity:.1f}, collision={coll:.1f})"
                    ),
                )
            )

    return hypos
