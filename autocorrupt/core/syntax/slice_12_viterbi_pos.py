"""
Slice 12: Deterministic Bigram Viterbi Part-of-Speech (POS) Tagger.
Updates the grammatical roles (NOUN, VERB, ADJ, ADV, PREP, DET, PRON) of tokens
on the SentenceTape in < 20 microseconds using transition log-weights and morphology.
Crucially resolves Noun/Verb ambiguity (e.g. "park", "book", "run", "lead", "season")
both immediately and retroactively when following words arrive.
"""
from __future__ import annotations
from autocorrupt.core.atoms.slice_00_types import InflectionTag, POS
from autocorrupt.core.perception.slice_09_mmap_trie import MmapLexiconStore
from autocorrupt.core.syntax.slice_11_sentence_tape import SentenceTape

#Closed-class function word POS lookup (strictly grammatical syntax)
FUNCTION_WORDS: dict[str, POS] = {
    "the": POS.DET, "a": POS.DET, "an": POS.DET, "this": POS.DET, "that": POS.DET,
    "these": POS.DET, "those": POS.DET, "my": POS.DET, "your": POS.DET, "his": POS.DET,
    "her": POS.DET, "our": POS.DET, "their": POS.DET, "some": POS.DET, "any": POS.DET,
    "every": POS.DET, "no": POS.DET,
    "i": POS.PRON, "you": POS.PRON, "he": POS.PRON, "she": POS.PRON, "it": POS.PRON,
    "we": POS.PRON, "they": POS.PRON, "me": POS.PRON, "him": POS.PRON, "us": POS.PRON,
    "them": POS.PRON, "im": POS.PRON, "i'm": POS.PRON, "you're": POS.PRON, "we're": POS.PRON,
    "to": POS.PREP, "in": POS.PREP, "on": POS.PREP, "at": POS.PREP, "for": POS.PREP,
    "with": POS.PREP, "by": POS.PREP, "from": POS.PREP, "of": POS.PREP, "into": POS.PREP,
    "out": POS.PREP, "up": POS.PREP, "down": POS.PREP, "over": POS.PREP, "under": POS.PREP,
    "after": POS.PREP, "before": POS.PREP, "besides": POS.PREP, "without": POS.PREP,
    "and": POS.CONJ, "or": POS.CONJ, "but": POS.CONJ, "so": POS.CONJ, "because": POS.CONJ,
    "very": POS.ADV, "really": POS.ADV, "just": POS.ADV, "always": POS.ADV, "never": POS.ADV,
    "not": POS.ADV, "definitely": POS.ADV, "reluctantly": POS.ADV, "totally": POS.ADV,
    "tomorrow": POS.ADV, "today": POS.ADV, "here": POS.ADV, "there": POS.ADV, "now": POS.ADV,
}

# Auxiliary / Modal verbs that force a following ambiguous Noun/Verb word into VERB
AUX_MODAL_WORDS: frozenset[str] = frozenset({
    "will", "would", "can", "could", "shall", "should", "may", "might", "must",
    "do", "does", "did", "don't", "dont", "didn't", "ill", "i'll", "we'll", "lets", "let's",
    "please", "gonna", "wanna",
})

COPULA_AUX: frozenset[str] = frozenset({
    "am", "is", "are", "was", "were", "be", "been", "being", "i'm", "im", "you're", "he's", "she's",
})

# Bigram transition weights: (prev_POS, curr_POS) -> integer score
TRANSITION_SCORES: dict[tuple[POS, POS], int] = {
    (POS.DET, POS.NOUN): 45,
    (POS.DET, POS.ADJ): 38,
    (POS.ADJ, POS.NOUN): 45,
    (POS.ADJ, POS.ADJ): 20,
    (POS.PRON, POS.VERB): 48,
    (POS.PRON, POS.ADV): 30,
    (POS.NOUN, POS.VERB): 32,
    (POS.NOUN, POS.NOUN): 22,  # noun adjuncts (e.g. "Darth Vader pillow", "banana phone")
    (POS.NOUN, POS.PREP): 35,
    (POS.VERB, POS.DET): 40,
    (POS.VERB, POS.NOUN): 36,
    (POS.VERB, POS.ADV): 38,
    (POS.VERB, POS.PREP): 38,
    (POS.VERB, POS.ADJ): 28,
    (POS.ADV, POS.VERB): 40,
    (POS.ADV, POS.ADJ): 38,
    (POS.PREP, POS.DET): 42,
    (POS.PREP, POS.NOUN): 38,
    (POS.PREP, POS.VERB): 25,  # infinitive "to" + VERB handled explicitly below
}


def update_pos_tags(tape: SentenceTape, store: MmapLexiconStore, window: int = 8) -> None:
    """
    Runs a deterministic Viterbi / contextual pass over the last `window` tokens
    on `tape`, updating `token.pos` in place.
    """
    n = len(tape.tokens)
    if n == 0:
        return
    start = max(0, n - window)

    for i in range(start, n):
        tok = tape.tokens[i]
        w_low = tok.raw_typed.lower()

        if w_low in FUNCTION_WORDS:
            tok.pos = FUNCTION_WORDS[w_low]
            continue

        if w_low in AUX_MODAL_WORDS or w_low in COPULA_AUX:
            tok.pos = POS.VERB
            continue

        entry = store.get_entry(tok.lemma) or store.get_entry(w_low)
        allowed_pos = list(entry.pos_tags) if (entry and entry.pos_tags) else [POS.NOUN, POS.VERB, POS.ADJ]

        # Morphological suffix overrides
        if tok.inflection == InflectionTag.ADVERB_LY:
            tok.pos = POS.ADV
            continue
        if tok.inflection in (InflectionTag.COMPARATIVE_ER, InflectionTag.SUPERLATIVE_EST):
            tok.pos = POS.ADJ
            continue
        if tok.inflection == InflectionTag.GERUND_ING:
            # After copula ("I'm running", "am auditioning") or at clause start -> VERB
            if POS.VERB in allowed_pos:
                tok.pos = POS.VERB
                continue

        if len(allowed_pos) == 1:
            tok.pos = allowed_pos[0]
            continue

        prev_tok = tape.tokens[i - 1] if i > 0 else None
        next_tok = tape.tokens[i + 1] if i + 1 < n else None
        prev_low = prev_tok.raw_typed.lower() if prev_tok else ""

        # High-certainty syntactic frames:
        # 1. Infinitive "to <verb>" or modal "will/can/let's <verb>"
        if (prev_low == "to" or prev_low in AUX_MODAL_WORDS) and POS.VERB in allowed_pos:
            # Verify it's not "to the" or "to <proper_noun>"
            if next_tok is None or next_tok.pos in (POS.DET, POS.PRON, POS.NOUN, POS.ADV, POS.PREP):
                tok.pos = POS.VERB
                continue

        # 2. Imperative clause start followed by Determiner ("Park the car", "Season the chicken")
        if i == 0 and next_tok is not None and next_tok.pos in (POS.DET, POS.PRON) and POS.VERB in allowed_pos:
            tok.pos = POS.VERB
            continue

        # 3. Preceded by Determiner ("the book", "a park", "that shirt", "the bass")
        if prev_tok is not None and prev_tok.pos == POS.DET:
            if next_tok is not None and next_tok.pos == POS.NOUN and POS.ADJ in allowed_pos:
                tok.pos = POS.ADJ
            elif POS.NOUN in allowed_pos:
                tok.pos = POS.NOUN
            continue

        # Score remaining candidates via Bigram Transition + Lookahead
        best_p = allowed_pos[0]
        best_score = -999
        for cand_p in allowed_pos:
            score = 0
            if prev_tok is not None:
                score += TRANSITION_SCORES.get((prev_tok.pos, cand_p), 5)
            if next_tok is not None:
                score += TRANSITION_SCORES.get((cand_p, next_tok.pos), 5)
            # Prefer primary lexical order on tie
            if cand_p == allowed_pos[0]:
                score += 4
            if score > best_score:
                best_score = score
                best_p = cand_p
        tok.pos = best_p
