"""
Slice 01: Articulatory Phoneme Feature Vector Table & Mouth-Shape Distance Metric.
Encodes all 39 ARPAbet/IPA English phonemes as 16-bit physical mouth-shape bitmasks
so the GOFAI engine computes true acoustic/articulatory similarity without hardcoded pairs.
"""
from __future__ import annotations

def _cons(voiced: int, place: int, manner_bit: int, rounded: int = 0) -> int:
    return (voiced & 1) | ((place & 0x7) << 1) | (1 << (5 + manner_bit)) | ((rounded & 1) << 14)

def _vow(height: int, backness: int, rounded: int = 0, diph: int = 0) -> int:
    return 1 | (1 << 9) | ((height & 0x3) << 10) | ((backness & 0x3) << 12) | ((rounded & 1) << 14) | ((diph & 1) << 15)

ARTICULATORY_MAP: dict[str, int] = {
    "P":  _cons(0, 1, 0),
    "B":  _cons(1, 1, 0),
    "T":  _cons(0, 2, 0),
    "D":  _cons(1, 2, 0),
    "K":  _cons(0, 4, 0),
    "G":  _cons(1, 4, 0),
    "F":  _cons(0, 1, 1),
    "V":  _cons(1, 1, 1),
    "TH": _cons(0, 2, 1),
    "DH": _cons(1, 2, 1),
    "S":  _cons(0, 2, 1),
    "Z":  _cons(1, 2, 1),
    "SH": _cons(0, 3, 1),
    "ZH": _cons(1, 3, 1),
    "CH": _cons(0, 3, 1) | (1 << 5),
    "JH": _cons(1, 3, 1) | (1 << 5),
    "HH": _cons(0, 4, 1),
    "M":  _cons(1, 1, 2),
    "N":  _cons(1, 2, 2),
    "NG": _cons(1, 4, 2),
    "L":  _cons(1, 2, 3),
    "R":  _cons(1, 2, 3) | (1 << 15),
    "Y":  _cons(1, 3, 3),
    "W":  _cons(1, 1, 3, rounded=1),
    "IY": _vow(0, 0, 0, 0),
    "IH": _vow(0, 0, 0, 0),
    "EY": _vow(1, 0, 0, 1),
    "EH": _vow(1, 0, 0, 0),
    "AE": _vow(2, 0, 0, 0),
    "AA": _vow(2, 2, 0, 0),
    "AO": _vow(1, 2, 1, 0),
    "OW": _vow(1, 2, 1, 1),
    "UH": _vow(0, 2, 1, 0),
    "UW": _vow(0, 2, 1, 0),
    "AH": _vow(1, 1, 0, 0),
    "ER": _vow(1, 1, 1, 1),
    "AY": _vow(2, 1, 0, 1),
    "AW": _vow(2, 2, 1, 1),
    "OY": _vow(1, 2, 1, 1),
}

VOWEL_SET: frozenset[str] = frozenset(
    k for k, v in ARTICULATORY_MAP.items() if (v & (1 << 9))
)


def is_vowel_phoneme(p: str) -> bool:
    return p in VOWEL_SET


def phoneme_distance(p1: str, p2: str) -> float:
    """
    Returns continuous articulatory substitution cost in [0.0, 2.0].
    Models syllabic rhotic equivalence (ER <-> R = 0.15), nasal assimilation (N <-> NG = 0.25),
    and place-of-articulation zones for oral stops/fricatives.
    """
    if p1 == p2:
        return 0.0
    if (p1 == "ER" and p2 == "R") or (p1 == "R" and p2 == "ER"):
        return 0.15
    if (p1 == "N" and p2 == "NG") or (p1 == "NG" and p2 == "N"):
        return 0.25
    v1 = ARTICULATORY_MAP.get(p1, 0)
    v2 = ARTICULATORY_MAP.get(p2, 0)
    if not v1 or not v2:
        return 1.2
    vow1 = bool(v1 & (1 << 9))
    vow2 = bool(v2 & (1 << 9))
    if vow1 != vow2:
        if p1 in ("R", "L", "W", "Y") or p2 in ("R", "L", "W", "Y"):
            return 0.85
        return 1.75
    xor_bits = (v1 ^ v2).bit_count()
    base = xor_bits * 0.18
    if not vow1:
        place1 = (v1 >> 1) & 0x7
        place2 = (v2 >> 1) & 0x7
        if abs(place1 - place2) >= 2:
            base += 0.35
    return min(1.5, base)


def sequence_articulatory_distance(
    seq1: tuple[str, ...],
    seq2: tuple[str, ...],
    max_cutoff: float = 99.0,
) -> float:
    """Dynamic programming articulatory edit distance between two phoneme sequences."""
    n, m = len(seq1), len(seq2)
    if n == 0:
        return float(m)
    if m == 0:
        return float(n)
    if abs(n - m) * 0.45 > max_cutoff:
        return max_cutoff + 1.0

    prev = [0.0] * (m + 1)
    for j in range(1, m + 1):
        ins_cost = 0.45 if seq2[j - 1] in ("AH", "IH", "ER", "R", "L", "S", "Z") else 0.80
        prev[j] = prev[j - 1] + ins_cost

    for i in range(1, n + 1):
        p_i = seq1[i - 1]
        del_cost = 0.45 if p_i in ("AH", "IH", "ER", "R", "L", "S", "Z") else 0.80
        curr = [prev[0] + del_cost] + [0.0] * m
        row_min = curr[0]
        for j in range(1, m + 1):
            p_j = seq2[j - 1]
            ins_cost = 0.45 if p_j in ("AH", "IH", "ER", "R", "L", "S", "Z") else 0.80
            sub_cost = phoneme_distance(p_i, p_j)
            val = min(
                prev[j] + del_cost,
                curr[j - 1] + ins_cost,
                prev[j - 1] + sub_cost,
            )
            curr[j] = val
            if val < row_min:
                row_min = val
        if row_min > max_cutoff:
            return max_cutoff + 1.0
        prev = curr
    return prev[m]


def consonant_skeleton(word: str) -> str:
    """
    Extracts the perceptual consonant-and-initial-vowel skeleton of a word.
    Handles French loanword elision (d'oeuvres -> ovrs == ovaries -> ovrs).
    """
    w = word.lower().replace("'", "")
    if len(w) > 4 and w.startswith("d") and w[1] in "aeiou":
        w = w[1:]
    out: list[str] = []
    for i, ch in enumerate(w):
        if not ch.isalpha():
            continue
        if i == 0 or ch not in "aeiouy":
            if not out or out[-1] != ch:
                out.append(ch)
    return "".join(out)


def phonetic_hash_64(phonemes: tuple[str, ...]) -> int:
    """Coarse 64-bit articulatory skeleton hash for O(1) bucket indexing."""
    h = 0xCBF29CE484222325
    skeleton_count = 0
    last_cls = -1
    for p in phonemes:
        v = ARTICULATORY_MAP.get(p, 0)
        if v & (1 << 9):
            if skeleton_count == 0:
                cls = 5
            else:
                continue
        else:
            cls = (v >> 1) & 0x7
        if cls == last_cls:
            continue
        last_cls = cls
        h ^= cls
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        skeleton_count += 1
        if skeleton_count >= 6:
            break
    return h
