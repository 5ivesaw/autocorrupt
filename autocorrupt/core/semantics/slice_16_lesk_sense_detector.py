"""
Slice 16: Deterministic Lesk Word-Sense Disambiguation (WSD) Detector.
Given a token on the SentenceTape and its surrounding sentence context, intersects
the sentence's words and domains with each candidate SynsetNode's definition gloss
to lock onto the exact dictionary sense the user intended.
"""
from __future__ import annotations
import re
from autocorrupt.core.perception.slice_09_mmap_trie import MmapLexiconStore
from autocorrupt.core.syntax.slice_11_sentence_tape import SentenceTape


def detect_intended_sense(
    tape: SentenceTape,
    slot_idx: int,
    store: MmapLexiconStore,
) -> int:
    """
    Returns the `synset_id` of the sense that best matches the surrounding sentence
    and the active POS tag of `tape.tokens[slot_idx]`.
    """
    if slot_idx < 0 or slot_idx >= len(tape.tokens):
        return -1
    target_tok = tape.tokens[slot_idx]
    entry = store.get_entry(target_tok.lemma) or store.get_entry(target_tok.raw_typed.lower())
    if entry is None or not entry.synset_ids:
        return -1
    if len(entry.synset_ids) == 1:
        target_tok.intended_synset_id = entry.synset_ids[0]
        target_tok.meaning_vec = store.synsets[entry.synset_ids[0]].meaning_vec
        return entry.synset_ids[0]

    # Collect context stems and domain masks from all other tokens on the tape
    ctx_words: set[str] = set()
    ctx_domains = 0
    for i, t in enumerate(tape.tokens):
        if i == slot_idx:
            continue
        ctx_words.add(t.raw_typed.lower())
        if t.lemma:
            ctx_words.add(t.lemma.lower())
        ctx_domains |= int(t.meaning_vec.domains)

    best_syn_id = entry.synset_ids[0]
    best_score = -999

    for sid in entry.synset_ids:
        syn = store.synsets[sid]
        score = 0
        # 1. Grammatical POS agreement
        if syn.pos == target_tok.pos:
            score += 25

        # 2. Gloss lexical overlap (classic Lesk intersection)
        gloss_words = set(re.findall(r"[a-z]+", syn.gloss.lower()))
        overlap = len(ctx_words & gloss_words)
        score += overlap * 35

        # 3. Semantic DomainMask overlap with context tokens
        shared_dom_bits = (int(syn.meaning_vec.domains) & ctx_domains).bit_count()
        score += shared_dom_bits * 22

        if score > best_score:
            best_score = score
            best_syn_id = sid

    target_tok.intended_synset_id = best_syn_id
    target_tok.meaning_vec = store.synsets[best_syn_id].meaning_vec
    return best_syn_id
