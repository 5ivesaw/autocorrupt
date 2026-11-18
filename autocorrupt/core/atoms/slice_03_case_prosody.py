"""
Slice 03: Orthographic Case & Prosody Profile Transfer.
Extracts the capitalization bitmask of any input token (or multi-token span) and
deterministically projects that exact casing onto synthesized replacement words.
"""
from __future__ import annotations

CASE_LOWER = 0
CASE_UPPER = 1
CASE_TITLE = 2
CASE_MIXED = 3


def extract_case_profile(word: str) -> int:
    """Classifies a word's capitalization pattern into an integer profile."""
    alpha = [c for c in word if c.isalpha()]
    if not alpha:
        return CASE_LOWER
    if all(c.islower() for c in alpha):
        return CASE_LOWER
    if all(c.isupper() for c in alpha) and len(alpha) > 1:
        return CASE_UPPER
    if alpha[0].isupper() and all(c.islower() for c in alpha[1:]):
        return CASE_TITLE
    return CASE_MIXED


def apply_case_profile(replacement: str, profile: int, reference: str = "") -> str:
    """Applies the extracted capitalization pattern onto the synthesized word or phrase."""
    if not replacement:
        return replacement
    if profile == CASE_LOWER:
        return replacement.lower()
    if profile == CASE_UPPER:
        return replacement.upper()
    if profile == CASE_TITLE:
        # Capitalize the first alphabetic character, preserve rest lower
        chars = list(replacement.lower())
        for i, ch in enumerate(chars):
            if ch.isalpha():
                chars[i] = ch.upper()
                break
        return "".join(chars)
    if profile == CASE_MIXED and reference:
        # Project character-by-character bitmask from reference string
        out = []
        ref_len = len(reference)
        for i, ch in enumerate(replacement):
            if i < ref_len and reference[i].isupper():
                out.append(ch.upper())
            else:
                out.append(ch.lower())
        return "".join(out)
    return replacement
