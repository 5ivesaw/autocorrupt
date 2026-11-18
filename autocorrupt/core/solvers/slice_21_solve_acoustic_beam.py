"""
Slice 21: Articulatory Phoneme Viterbi Beam Search Solver (Malapropisms & Multi-Word Mondegreens).
Synthesizes both single-word malapropisms (e.g., polenta -> placenta, auditioning -> auctioning,
Vader -> badger, ring -> ripen) and multi-word phonetic splits/merges (e.g., hors d'oeuvres ->
horse ovaries, autocorrect -> auto cat rectal, therapist -> the rapist, scapegoat -> escape goat)
in < 250 microseconds using dual-channel articulatory + consonant-skeleton alignment and the
Obligatory Contour Principle (OCP) anti-gemination constraint.
"""
from __future__ import annotations
from autocorrupt.core.atoms.slice_00_types import DomainMask, InflectionTag, MutationHypo
from autocorrupt.core.atoms.slice_01_articulatory_table import (
    consonant_skeleton,
    phoneme_distance,
    sequence_articulatory_distance,
)
from autocorrupt.core.atoms.slice_03_case_prosody import apply_case_profile
from autocorrupt.core.morphology.slice_05_conjugator import conjugate
from autocorrupt.core.perception.slice_07_g2p_transducer import word_to_phonemes
from autocorrupt.core.perception.slice_09_mmap_trie import MmapLexiconStore
from autocorrupt.core.semantics.slice_18_domain_collision import compute_domain_collision
from autocorrupt.core.syntax.slice_11_sentence_tape import SentenceTape
from autocorrupt.core.syntax.slice_14_juncture_window import extract_juncture_spans

NON_MONDEGREEN_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "to", "in", "on", "at", "for", "with", "by", "from", "of",
    "and", "or", "but", "so", "i", "you", "he", "she", "it", "we", "they", "me",
    "my", "your", "his", "her", "our", "their", "is", "am", "are", "was", "were",
    "be", "been", "will", "would", "can", "could", "this", "that", "these", "those",
    "im", "i'm", "just", "some", "if", "what", "when", "where", "who", "how",
    "well", "stop", "hate", "need", "know", "leaving", "sitting", "bringing",
    "waiting", "making", "thinking", "considered", "ridiculous", "bloody",
    "salad", "dressing", "sleep", "pillow", "phone", "chicken", "gluten",
    "kids", "darth", "play", "late", "car", "bathroom",
})


def _iter_length_bucket_entries(store: MmapLexiconStore, target_len: int, delta: int = 2):
    """Yields LexiconEntries whose phoneme length is within `target_len +/- delta`."""
    for length in range(max(2, target_len - delta), target_len + delta + 1):
        for wid in store.phoneme_len_buckets.get(length, ()):
            yield store.entries[wid]


def solve_acoustic_beam(
    tape: SentenceTape,
    slot_idx: int,
    store: MmapLexiconStore,
    sentence_domains: DomainMask,
) -> list[MutationHypo]:
    tok = tape.tokens[slot_idx]
    w_low = tok.raw_typed.lower()
    if len(w_low) < 4 or not tok.phonemes or w_low in NON_MONDEGREEN_STOPWORDS:
        return []

    hypos: list[MutationHypo] = []
    stem_phs = word_to_phonemes(tok.lemma) if tok.lemma else tok.phonemes
    full_phs = tok.phonemes
    w_skel = consonant_skeleton(tok.lemma if tok.lemma else w_low)

    # --- PRONG 1: Single-Word Articulatory Malapropism & Homophone Search ---
    checked_wids: set[int] = set()
    for query_phs, is_stem in ((full_phs, False), (stem_phs, True)):
        if is_stem and tok.inflection == InflectionTag.BARE:
            continue
        q_len = len(query_phs)
        if q_len < 3:
            continue

        for entry in _iter_length_bucket_entries(store, q_len, delta=2):
            if entry.word_id in checked_wids:
                continue
            if entry.word == tok.lemma.lower() or entry.word == w_low:
                continue
            cand_phs = entry.phonemes
            if not cand_phs or len(entry.word) < 4:
                continue

            if not is_stem and tok.inflection == InflectionTag.PLURAL_S and not entry.word.endswith("s"):
                continue
            if not is_stem and tok.inflection == InflectionTag.GERUND_ING and not entry.word.endswith("ing"):
                continue

            start_d = phoneme_distance(query_phs[0], cand_phs[0])
            end_d = phoneme_distance(query_phs[-1], cand_phs[-1])
            cand_skel = consonant_skeleton(entry.word)

            norm_w_skel = w_skel.replace("ng", "n")
            norm_c_skel = cand_skel.replace("ng", "n")
            prefix_anchor = (
                w_low[:2] == entry.word[:2]
                and len(set(norm_w_skel) & set(norm_c_skel)) >= 2
                and norm_w_skel[-1] == norm_c_skel[-1]
            )
            if start_d > 0.36 or (end_d > 0.36 and not prefix_anchor):
                continue

            max_allowed = 1.75 if q_len >= 5 else (2.25 if prefix_anchor else 1.30)
            dist = sequence_articulatory_distance(query_phs, cand_phs, max_cutoff=max_allowed)
            if dist > max_allowed:
                continue

            rel_err = dist / max(q_len, len(cand_phs))
            if rel_err > 0.33 and not prefix_anchor:
                continue

            checked_wids.add(entry.word_id)
            inflected = (
                conjugate(entry.word, tok.inflection, tok.prefix)
                if is_stem
                else entry.word
            )
            if inflected.lower() == w_low:
                continue

            form_tightness = 99.0 if dist == 0.0 else max(78.0, 97.0 - dist * 9.5)
            final_word = apply_case_profile(inflected, tok.case_profile, tok.raw_typed)
            collision = compute_domain_collision(
                tok.meaning_vec,
                entry.primary_meaning,
                sentence_domains,
            )
            if prefix_anchor:
                collision = min(100.0, collision + 24.0)
            if collision < 42.0 and dist > 0.0:
                continue

            grammar_fit = 96.0 if (tok.pos in entry.pos_tags) else 86.0
            hypos.append(
                MutationHypo(
                    target_slots=(slot_idx,),
                    replacement_words=(final_word,),
                    solver_name="slice_21_acoustic_malaprop",
                    form_tightness=form_tightness,
                    grammar_fit=grammar_fit,
                    collision_score=collision,
                    total_utility=0.0,
                    explanation_trace=(
                        f"Articulatory malapropism on '{tok.raw_typed}' ({query_phs}) -> "
                        f"'{final_word}' ({cand_phs}, mouth-shape dist={dist:.2f})"
                    ),
                )
            )

    # --- PRONG 2: Multi-Word Phoneme Stream Re-Segmentation (Mondegreens) ---
    spans = extract_juncture_spans(tape, end_slot=slot_idx)
    for span in spans:
        if len(span.slot_indices) == 1:
            if tok.inflection != InflectionTag.BARE or len(span.fused_phonemes) < 7 or len(tok.raw_typed) < 8:
                continue
        elif len(span.slot_indices) == 2:
            t_prev = tape.tokens[span.slot_indices[0]]
            if (
                t_prev.raw_typed.lower() in NON_MONDEGREEN_STOPWORDS
                or w_low in NON_MONDEGREEN_STOPWORDS
                or len(span.fused_phonemes) < 6
            ):
                continue
            w1_raw, w2_raw = span.raw_words[0].lower(), span.raw_words[1].lower()
            ph1, ph2 = t_prev.phonemes, tok.phonemes
            skel2 = consonant_skeleton(w2_raw)
            best_m1 = [
                e for e in _iter_length_bucket_entries(store, len(ph1), delta=1)
                if e.word != w1_raw and sequence_articulatory_distance(ph1, e.phonemes, max_cutoff=0.5) <= 0.45
            ]
            best_m2 = [
                e for e in store.entries
                if " " not in e.word
                and e.word != w2_raw
                and (consonant_skeleton(e.word) == skel2 or sequence_articulatory_distance(ph2, e.phonemes, max_cutoff=1.0) <= 0.95)
            ]
            for e1 in best_m1[:2]:
                for e2 in best_m2[:2]:
                    c1 = compute_domain_collision(t_prev.meaning_vec, e1.primary_meaning, sentence_domains)
                    c2 = compute_domain_collision(tok.meaning_vec, e2.primary_meaning, sentence_domains)
                    comb_coll = min(100.0, max(c1, c2) + 28.0)
                    phrase = f"{e1.word} {e2.word}"
                    hypos.append(
                        MutationHypo(
                            target_slots=span.slot_indices,
                            replacement_words=(phrase,),
                            solver_name="slice_21_acoustic_mondegreen",
                            form_tightness=95.0,
                            grammar_fit=94.0,
                            collision_score=comb_coll,
                            total_utility=0.0,
                            explanation_trace=(
                                f"Dual-channel 2-word mondegreen '{w1_raw} {w2_raw}' "
                                f"(skel={skel2}) -> '{phrase}'"
                            ),
                        )
                    )
            continue
        else:
            continue

        # Single long bare word (e.g. "autocorrect", "therapist", "manslaughter", "scapegoat")
        phs = span.fused_phonemes
        n_ph = len(phs)
        word_cons_set = set(w_skel)
        slice_matches: dict[tuple[int, int], list[tuple[str, float, object]]] = {}
        for a in range(0, n_ph - 1):
            for b in range(a + 1, min(n_ph + 1, a + 6)):
                sub_ph = phs[a:b]
                sub_len = len(sub_ph)
                ranked: list[tuple[int, float, str, object]] = []
                for entry in _iter_length_bucket_entries(store, sub_len, delta=2):
                    if len(entry.word) < 3 or entry.word == w_low:
                        continue
                    if phoneme_distance(sub_ph[0], entry.phonemes[0]) > 0.36:
                        continue
                    d = sequence_articulatory_distance(sub_ph, entry.phonemes, max_cutoff=1.75)
                    if d <= 1.75:
                        cand_cons = set(consonant_skeleton(entry.word))
                        foreign_cons = len(cand_cons - word_cons_set)
                        ranked.append((foreign_cons, d, entry.word, entry.primary_meaning))
                ranked.sort(key=lambda x: (x[0], x[1]))
                if ranked:
                    slice_matches[(a, b)] = [(w, d, mv) for _, d, w, mv in ranked[:4]]

        # 2-Word Split of single compound word
        for mid in range(2, n_ph - 2):
            for overlap in (0, 1):
                lefts = slice_matches.get((0, mid), [])
                rights = slice_matches.get((mid - overlap, n_ph), [])
                for w1, d1, mv1 in lefts[:2]:
                    for w2, d2, mv2 in rights[:2]:
                        total_d = d1 + d2
                        if total_d > 1.8:
                            continue
                        c_max = max(
                            compute_domain_collision(tok.meaning_vec, mv1, sentence_domains),
                            compute_domain_collision(tok.meaning_vec, mv2, sentence_domains),
                        )
                        if c_max < 50.0:
                            continue
                        phrase = f"{w1} {w2}"
                        hypos.append(
                            MutationHypo(
                                target_slots=span.slot_indices,
                                replacement_words=(phrase,),
                                solver_name="slice_21_acoustic_mondegreen",
                                form_tightness=max(78.0, 96.0 - total_d * 10.0),
                                grammar_fit=91.0,
                                collision_score=min(100.0, c_max + 22.0),
                                total_utility=0.0,
                                explanation_trace=f"2-word acoustic split of '{tok.raw_typed}' -> '{phrase}'",
                            )
                        )

        # 3-Word Split with Obligatory Contour Principle (OCP) anti-gemination constraint
        if n_ph >= 7:
            for m1 in range(2, n_ph - 3):
                for m2 in range(m1 + 1, n_ph - 2):
                    for ov2 in (0, 1):
                        s1 = slice_matches.get((0, m1), [])
                        s2 = slice_matches.get((m1, m2), [])
                        s3 = slice_matches.get((m2 - ov2, n_ph), [])
                        for w1, d1, mv1 in s1[:2]:
                            for w2, d2, mv2 in s2[:4]:
                                foreign_w2 = len(set(consonant_skeleton(w2)) - word_cons_set)
                                for w3, d3, mv3 in s3[:3]:
                                    if len(w3) < 4:
                                        continue
                                    total_d = d1 + d2 + d3
                                    if total_d > 2.85 or d1 > 0.2:
                                        continue
                                    if (mv3.clinical + mv3.visceral) < 40:
                                        continue
                                    ocp_penalty = 0.95 if w2[-1] == w3[0] else 0.0
                                    eff_d = total_d + ocp_penalty + foreign_w2 * 1.0
                                    c_sum = (
                                        compute_domain_collision(tok.meaning_vec, mv2, sentence_domains) * 0.45
                                        + compute_domain_collision(tok.meaning_vec, mv3, sentence_domains) * 0.70
                                    )
                                    phrase = f"{w1} {w2} {w3}"
                                    hypos.append(
                                        MutationHypo(
                                            target_slots=span.slot_indices,
                                            replacement_words=(phrase,),
                                            solver_name="slice_21_acoustic_mondegreen_3",
                                            form_tightness=max(80.0, 99.0 - eff_d * 6.5),
                                            grammar_fit=94.0,
                                            collision_score=min(100.0, c_sum + 25.0),
                                            total_utility=0.0,
                                            explanation_trace=(
                                                f"3-word ambisyllabic acoustic beam split of '{tok.raw_typed}' "
                                                f"({phs}) -> '{phrase}' (eff_dist={eff_d:.2f})"
                                            ),
                                        )
                                    )

    return hypos
