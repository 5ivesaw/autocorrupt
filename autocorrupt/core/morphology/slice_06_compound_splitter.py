"""
Slice 06: Compound Word Decomposer.
Splits compound English words (e.g., "nutcracker" -> ("nut", "cracker"),
"birdseed" -> ("bird", "seed"), "bedroom" -> ("bed", "room")) by searching for
valid constituent dictionary lemmas of length >= 3.
"""
from __future__ import annotations


def split_compound(word: str, known_words: set[str]) -> tuple[str, str] | None:
    """
    If `word` (>= 6 chars) can be cleanly partitioned into two real dictionary words
    (each >= 3 chars), returns `(left_part, right_part)`; otherwise returns `None`.
    """
    w = word.lower()
    n = len(w)
    if n < 6:
        return None
    best_split: tuple[str, str] | None = None
    best_balance = 999
    for i in range(3, n - 2):
        left = w[:i]
        right = w[i:]
        if left in known_words and right in known_words:
            balance = abs(len(left) - len(right))
            if balance < best_balance:
                best_balance = balance
                best_split = (left, right)
    return best_split
