"""
Slice 05: Morphological Re-Inflector & Conjugator.
Takes any bare root lemma synthesized by a GOFAI solver and re-applies the original
word's InflectionTag (handling CVC consonant doubling, silent-e elision, -y -> -ies,
and irregular stems) so the replacement slots seamlessly into the sentence's grammar.
"""
from __future__ import annotations
from autocorrupt.core.atoms.slice_00_types import InflectionTag

_VOWELS = frozenset("aeiou")

IRREGULAR_CONJUGATIONS: dict[tuple[str, InflectionTag], str] = {
    ("run", InflectionTag.PAST_ED): "ran",
    ("run", InflectionTag.GERUND_ING): "running",
    ("jog", InflectionTag.GERUND_ING): "jogging",
    ("jog", InflectionTag.PAST_ED): "jogged",
    ("sprint", InflectionTag.GERUND_ING): "sprinting",
    ("dangle", InflectionTag.GERUND_ING): "dangling",
    ("fondle", InflectionTag.GERUND_ING): "fondling",
    ("fondle", InflectionTag.PAST_ED): "fondled",
    ("ingest", InflectionTag.GERUND_ING): "ingesting",
    ("ingest", InflectionTag.PAST_ED): "ingested",
    ("climax", InflectionTag.GERUND_ING): "climaxing",
    ("climax", InflectionTag.PAST_ED): "climaxed",
    ("child", InflectionTag.PLURAL_S): "children",
    ("man", InflectionTag.PLURAL_S): "men",
    ("woman", InflectionTag.PLURAL_S): "women",
    ("tooth", InflectionTag.PLURAL_S): "teeth",
    ("foot", InflectionTag.PLURAL_S): "feet",
    ("mouse", InflectionTag.PLURAL_S): "mice",
}

# Multi-syllable words stressed on the first syllable that do NOT double final consonant
NO_DOUBLE_STEMS: frozenset[str] = frozenset({
    "audition", "auction", "suction", "open", "happen", "listen", "offer",
    "suffer", "enter", "order", "visit", "limit", "credit", "edit", "vomit",
    "orbit", "habit", "profit", "target", "market", "carpet", "ripen", "widen",
    "burden", "garden", "pardon", "sadden", "madden", "abandon", "develop",
    "gallop", "gossip", "worship", "water", "season", "butter", "anchor",
})


def _should_double_final_consonant(lemma: str) -> bool:
    """True if lemma ends in stressed single-vowel + single-consonant (CVC)."""
    if len(lemma) < 3 or lemma in NO_DOUBLE_STEMS:
        return False
    if lemma.endswith(("w", "x", "y", "l")):
        return False
    # If word has 2+ vowel clusters and doesn't end in common stressed CVC like "begin"/"prefer", don't double
    vowel_groups = 0
    prev_v = False
    for ch in lemma:
        is_v = ch in _VOWELS
        if is_v and not prev_v:
            vowel_groups += 1
        prev_v = is_v
    if vowel_groups > 1 and not lemma.endswith(("gin", "fer", "mit", "cur", "pel", "get", "rot")):
        return False
    c1, v, c2 = lemma[-3], lemma[-2], lemma[-1]
    return (c1 not in _VOWELS) and (v in _VOWELS) and (c2 not in _VOWELS)


def conjugate(lemma: str, inflection: InflectionTag, prefix: str = "") -> str:
    """Deterministically inflects `lemma` (including multi-word phrases) to match `inflection`."""
    if not lemma:
        return lemma

    # If the lemma is a multi-word phrase (e.g., "exchange oral mucosa"), inflect the head word
    if " " in lemma:
        parts = lemma.split(" ")
        parts[0] = conjugate(parts[0], inflection)
        return prefix + " ".join(parts)

    w = lemma.lower()
    if inflection == InflectionTag.BARE:
        return prefix + w

    irr = IRREGULAR_CONJUGATIONS.get((w, inflection))
    if irr is not None:
        return prefix + irr

    if inflection == InflectionTag.GERUND_ING:
        if w.endswith("ie"):
            return prefix + w[:-2] + "ying"
        if w.endswith("e") and not w.endswith(("ee", "oe", "ye")):
            return prefix + w[:-1] + "ing"
        if _should_double_final_consonant(w):
            return prefix + w + w[-1] + "ing"
        return prefix + w + "ing"

    if inflection in (InflectionTag.PAST_ED, InflectionTag.PAST_PART):
        if w.endswith("e"):
            return prefix + w + "d"
        if len(w) > 2 and w.endswith("y") and w[-2] not in _VOWELS:
            return prefix + w[:-1] + "ied"
        if _should_double_final_consonant(w):
            return prefix + w + w[-1] + "ed"
        return prefix + w + "ed"

    if inflection in (InflectionTag.PLURAL_S, InflectionTag.THIRD_SING_S):
        if len(w) > 2 and w.endswith("y") and w[-2] not in _VOWELS:
            return prefix + w[:-1] + "ies"
        if w.endswith(("s", "sh", "ch", "x", "z")):
            return prefix + w + "es"
        return prefix + w + "s"

    if inflection == InflectionTag.ADVERB_LY:
        if w.endswith("le") and len(w) > 2 and w[-3] not in _VOWELS:
            return prefix + w[:-1] + "y"
        if w.endswith("y") and len(w) > 2:
            return prefix + w[:-1] + "ily"
        if w.endswith("ly"):
            return prefix + w
        return prefix + w + "ly"

    if inflection == InflectionTag.COMPARATIVE_ER:
        if w.endswith("e"):
            return prefix + w + "r"
        if len(w) > 2 and w.endswith("y") and w[-2] not in _VOWELS:
            return prefix + w[:-1] + "ier"
        if _should_double_final_consonant(w):
            return prefix + w + w[-1] + "er"
        return prefix + w + "er"

    if inflection == InflectionTag.SUPERLATIVE_EST:
        if w.endswith("e"):
            return prefix + w + "st"
        if len(w) > 2 and w.endswith("y") and w[-2] not in _VOWELS:
            return prefix + w[:-1] + "iest"
        if _should_double_final_consonant(w):
            return prefix + w + w[-1] + "est"
        return prefix + w + "est"

    return prefix + w
