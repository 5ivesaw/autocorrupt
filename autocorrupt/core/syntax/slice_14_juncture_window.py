"""
Slice 14: Cross-Token Juncture & Oronym Window Scanner.
Extracts 2-token and 3-token boundary spans from the SentenceTape, fusing adjacent
words' orthography and phoneme streams (e.g., "lamp" + "on" -> "lampon",
"for" + "the" + "play" -> "fortheplay", "hors" + "d'oeuvres" -> phoneme stream)
so solvers can synthesize cross-boundary replacements without hardcoded phrase tables.
"""
from __future__ import annotations
from dataclasses import dataclass
from autocorrupt.core.syntax.slice_11_sentence_tape import SentenceTape


@dataclass(frozen=True, slots=True)
class JunctureSpan:
    slot_indices: tuple[int, ...]
    raw_words: tuple[str, ...]
    fused_ortho: str
    fused_phonemes: tuple[str, ...]


def extract_juncture_spans(tape: SentenceTape, end_slot: int | None = None) -> list[JunctureSpan]:
    """
    Returns the single-word, 2-word, and 3-word spans ending at `end_slot`
    (defaults to the last committed token on `tape`).
    Always uses `raw_typed` so an intermediate single-word mutation on slot `k-1`
    does not corrupt 2-word juncture recognition on `(k-1, k)`.
    """
    n = len(tape.tokens)
    if n == 0:
        return []
    k = (n - 1) if end_slot is None else end_slot
    if k < 0 or k >= n:
        return []

    spans: list[JunctureSpan] = []

    # 1-token span
    t0 = tape.tokens[k]
    spans.append(
        JunctureSpan(
            slot_indices=(k,),
            raw_words=(t0.raw_typed,),
            fused_ortho=t0.raw_typed.lower(),
            fused_phonemes=t0.phonemes,
        )
    )

    # 2-token span: (k-1, k)
    if k >= 1:
        tm1 = tape.tokens[k - 1]
        spans.append(
            JunctureSpan(
                slot_indices=(k - 1, k),
                raw_words=(tm1.raw_typed, t0.raw_typed),
                fused_ortho=(tm1.raw_typed + t0.raw_typed).lower(),
                fused_phonemes=tm1.phonemes + t0.phonemes,
            )
        )

    # 3-token span: (k-2, k-1, k) — e.g. "for the play"
    if k >= 2:
        tm2 = tape.tokens[k - 2]
        tm1 = tape.tokens[k - 1]
        mid_low = tm1.raw_typed.lower()
        fused_ortho = (
            (tm2.raw_typed + t0.raw_typed).lower()
            if mid_low in ("the", "a", "an", "of", "to")
            else (tm2.raw_typed + tm1.raw_typed + t0.raw_typed).lower()
        )
        fused_ph = (
            tm2.phonemes + t0.phonemes
            if mid_low in ("the", "a", "an")
            else tm2.phonemes + tm1.phonemes + t0.phonemes
        )
        spans.append(
            JunctureSpan(
                slot_indices=(k - 2, k - 1, k),
                raw_words=(tm2.raw_typed, tm1.raw_typed, t0.raw_typed),
                fused_ortho=fused_ortho,
                fused_phonemes=fused_ph,
            )
        )

    return spans
