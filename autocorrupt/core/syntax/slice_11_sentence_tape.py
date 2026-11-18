"""
Slice 11: The 64-Token SentenceTape Ring Buffer & Live Screen Offset Tracker.
Maintains the active sentence/paragraph in memory, tracks exact character widths on screen,
and enables retroactive ("Time-Travel") rewrites of tokens up to 8 words back.
"""
from __future__ import annotations
from autocorrupt.core.atoms.slice_00_types import InflectionTag, POS, TokenNode
from autocorrupt.core.atoms.slice_03_case_prosody import extract_case_profile
from autocorrupt.core.morphology.slice_04_affix_stripper import strip_affixes
from autocorrupt.core.perception.slice_07_g2p_transducer import word_to_phonemes
from autocorrupt.core.perception.slice_09_mmap_trie import MmapLexiconStore


class SentenceTape:
    """
    Deterministic 64-slot token tape representing the user's current typing line/paragraph.
    Never forgets earlier words in the sentence until a hard cursor navigation or newline reset.
    """

    MAX_SLOTS = 64

    def __init__(self) -> None:
        self.tokens: list[TokenNode] = []
        self.active_chars: list[str] = []
        self.active_trie_state: int = 0

    def hard_reset(self) -> None:
        """Clears the tape on cursor jump, mouse click, or window focus change."""
        self.tokens.clear()
        self.active_chars.clear()
        self.active_trie_state = 0

    def feed_char(self, ch: str, store: MmapLexiconStore) -> int:
        """
        Appends one character to the active word buffer and steps the character trie
        in ~15 nanoseconds.
        """
        self.active_chars.append(ch)
        if self.active_trie_state >= 0:
            self.active_trie_state = store.step_char(self.active_trie_state, ch)
        return self.active_trie_state

    def handle_backspace(self, store: MmapLexiconStore) -> None:
        """Handles a physical user Backspace keystroke accurately across active or committed tokens."""
        if self.active_chars:
            self.active_chars.pop()
            # Re-step trie state for remaining active chars
            state = 0
            for c in self.active_chars:
                state = store.step_char(state, c)
                if state < 0:
                    break
            self.active_trie_state = state
        elif self.tokens:
            last_tok = self.tokens.pop()
            # Restore the last token (minus its trailing separator if any) into active_chars
            if len(last_tok.trailing_sep) > 1:
                last_tok.trailing_sep = last_tok.trailing_sep[:-1]
                self.tokens.append(last_tok)
            else:
                self.active_chars = list(last_tok.current_text)
                state = 0
                for c in self.active_chars:
                    state = store.step_char(state, c)
                    if state < 0:
                        break
                self.active_trie_state = state

    def commit_active_word(self, trailing_sep: str, store: MmapLexiconStore) -> TokenNode | None:
        """
        Finalizes `active_chars` when Space or punctuation is pressed and pushes a new
        TokenNode onto the SentenceTape.
        """
        raw = "".join(self.active_chars)
        self.active_chars.clear()
        self.active_trie_state = 0

        if not raw:
            if self.tokens:
                self.tokens[-1].trailing_sep += trailing_sep
            return None

        case_prof = extract_case_profile(raw)
        prefix, lemma, infl = strip_affixes(raw, store.known_words)
        entry = store.get_entry(lemma) or store.get_entry(raw.lower())

        if entry is not None:
            phonemes = word_to_phonemes(raw) if infl != InflectionTag.BARE else entry.phonemes
            pos = entry.pos_tags[0] if entry.pos_tags else POS.NOUN
            meaning = entry.primary_meaning
            synset_id = entry.synset_ids[0] if entry.synset_ids else -1
        else:
            phonemes = word_to_phonemes(raw)
            pos = POS.NOUN
            from autocorrupt.core.atoms.slice_00_types import MeaningVec
            meaning = MeaningVec()
            synset_id = -1

        slot_idx = len(self.tokens)
        node = TokenNode(
            slot_index=slot_idx,
            raw_typed=raw,
            current_text=raw,
            trailing_sep=trailing_sep,
            lemma=lemma,
            prefix=prefix,
            inflection=infl,
            case_profile=case_prof,
            pos=pos,
            phonemes=phonemes,
            intended_synset_id=synset_id,
            meaning_vec=meaning,
        )
        self.tokens.append(node)
        if len(self.tokens) > self.MAX_SLOTS:
            self.tokens.pop(0)
            for idx, t in enumerate(self.tokens):
                t.slot_index = idx
        return node

    def render_sentence(self) -> str:
        """Returns the exact text currently displayed on the user's screen."""
        parts = [t.current_text + t.trailing_sep for t in self.tokens]
        if self.active_chars:
            parts.append("".join(self.active_chars))
        return "".join(parts)

    def span_screen_text(self, start_slot: int) -> str:
        """Returns the exact screen text from `start_slot` to the current cursor."""
        parts = [t.current_text + t.trailing_sep for t in self.tokens[start_slot:]]
        if self.active_chars:
            parts.append("".join(self.active_chars))
        return "".join(parts)
