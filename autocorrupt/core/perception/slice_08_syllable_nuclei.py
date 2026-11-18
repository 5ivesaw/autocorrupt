"""
Slice 08: Syllable Nuclei Segmenter & Onset-Rime Decomposer.
Segments any phoneme tuple into structured Syllable(onset, nucleus, coda) blocks
using the Maximal Onset Principle. Used by the Acoustic Beam Solver (for multi-word
mondegreen boundaries) and the Spoonerism Solver (for onset consonant cluster swapping).
"""
from __future__ import annotations
from dataclasses import dataclass
from autocorrupt.core.atoms.slice_01_articulatory_table import is_vowel_phoneme


@dataclass(frozen=True, slots=True)
class Syllable:
    onset: tuple[str, ...]
    nucleus: tuple[str, ...]
    coda: tuple[str, ...]


def segment_syllables(phonemes: tuple[str, ...]) -> tuple[Syllable, ...]:
    """
    Segments a phoneme tuple into Syllables around vowel nuclei.
    Applies the Maximal Onset Principle: intervocalic consonants attach to the
    following syllable's onset up to valid English 2-consonant onset clusters.
    """
    if not phonemes:
        return ()

    vowel_indices = [i for i, p in enumerate(phonemes) if is_vowel_phoneme(p)]
    if not vowel_indices:
        return (Syllable(onset=phonemes, nucleus=(), coda=()),)

    syllables: list[Syllable] = []
    num_v = len(vowel_indices)

    prev_coda_end = 0
    for idx_pos, v_idx in enumerate(vowel_indices):
        onset = phonemes[prev_coda_end:v_idx]
        nucleus = (phonemes[v_idx],)

        if idx_pos == num_v - 1:
            # Last vowel owns all remaining trailing consonants as its coda
            coda = phonemes[v_idx + 1:]
            prev_coda_end = len(phonemes)
        else:
            next_v_idx = vowel_indices[idx_pos + 1]
            inter = phonemes[v_idx + 1:next_v_idx]
            if len(inter) == 0:
                coda = ()
                prev_coda_end = v_idx + 1
            elif len(inter) == 1:
                # Single intervocalic consonant goes to next syllable's onset
                coda = ()
                prev_coda_end = v_idx + 1
            else:
                # Split cluster: keep all but last 1 (or 2 if valid cluster with R/L/W) in current coda
                if inter[-1] in ("R", "L", "W", "Y") and inter[-2] in ("P", "B", "T", "D", "K", "G", "F", "S"):
                    split_at = len(inter) - 2
                else:
                    split_at = len(inter) - 1
                coda = inter[:split_at]
                prev_coda_end = v_idx + 1 + split_at

        syllables.append(Syllable(onset=onset, nucleus=nucleus, coda=coda))

    return tuple(syllables)


def split_onset_rime(phonemes: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Splits a word's phoneme sequence into `(initial_onset_cluster, remaining_rime)`
    for deterministic Spoonerism generation.
    """
    for i, p in enumerate(phonemes):
        if is_vowel_phoneme(p):
            return (phonemes[:i], phonemes[i:])
    return (phonemes, ())
