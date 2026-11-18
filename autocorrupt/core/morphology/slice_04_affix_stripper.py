"""
Slice 04: Morphological Affix Stripper & Deterministic Lemmatizer.
Deconstructs any inflected English surface word into (prefix, bare_lemma, InflectionTag)
using phonological/orthographic stripping rules plus irregular stem resolution.
"""
from __future__ import annotations
from autocorrupt.core.atoms.slice_00_types import InflectionTag

# Irregular inflectional forms (strictly grammatical morphology, zero jokes)
IRREGULAR_LEMMAS: dict[str, tuple[str, InflectionTag]] = {
    "ran": ("run", InflectionTag.PAST_ED),
    "running": ("run", InflectionTag.GERUND_ING),
    "ate": ("eat", InflectionTag.PAST_ED),
    "eaten": ("eat", InflectionTag.PAST_PART),
    "eating": ("eat", InflectionTag.GERUND_ING),
    "left": ("leave", InflectionTag.PAST_ED),
    "leaving": ("leave", InflectionTag.GERUND_ING),
    "making": ("make", InflectionTag.GERUND_ING),
    "made": ("make", InflectionTag.PAST_ED),
    "taking": ("take", InflectionTag.GERUND_ING),
    "took": ("take", InflectionTag.PAST_ED),
    "taken": ("take", InflectionTag.PAST_PART),
    "shaking": ("shake", InflectionTag.GERUND_ING),
    "shook": ("shake", InflectionTag.PAST_ED),
    "sitting": ("sit", InflectionTag.GERUND_ING),
    "sat": ("sit", InflectionTag.PAST_ED),
    "waiting": ("wait", InflectionTag.GERUND_ING),
    "bringing": ("bring", InflectionTag.GERUND_ING),
    "brought": ("bring", InflectionTag.PAST_ED),
    "thinking": ("think", InflectionTag.GERUND_ING),
    "thought": ("think", InflectionTag.PAST_ED),
    "children": ("child", InflectionTag.PLURAL_S),
    "men": ("man", InflectionTag.PLURAL_S),
    "women": ("woman", InflectionTag.PLURAL_S),
    "teeth": ("tooth", InflectionTag.PLURAL_S),
    "feet": ("foot", InflectionTag.PLURAL_S),
    "mice": ("mouse", InflectionTag.PLURAL_S),
}

# Words ending in -s / -ing / -ed / -er / -ly that are actually bare lemmas
BARE_EXCEPTIONS: frozenset[str] = frozenset({
    "morning", "evening", "ceiling", "feeling", "wedding", "pudding", "nothing",
    "something", "everything", "anything", "string", "spring", "ring", "king",
    "wing", "sing", "bring", "thing", "fling", "cling", "sting", "swing",
    "dress", "stress", "press", "guess", "mess", "less", "bless", "chess",
    "glass", "class", "grass", "pass", "mass", "bass", "gas", "bus", "plus",
    "status", "virus", "focus", "bonus", "campus", "circus", "mucus", "fetus",
    "always", "sometimes", "perhaps", "towards", "regards", "hors", "d'oeuvres",
    "polenta", "placenta", "bed", "red", "fed", "led", "shed", "sled", "wed",
    "seed", "weed", "feed", "need", "speed", "bleed", "breed", "greed",
    "water", "father", "mother", "brother", "sister", "other", "weather",
    "feather", "leather", "finger", "shoulder", "dinner", "summer", "winter",
    "flower", "tower", "shower", "power", "anger", "danger", "stranger",
    "badger", "vader", "nutcracker", "butter", "letter", "matter", "better",
    "early", "only", "family", "belly", "jelly", "holy", "ugly", "friendly",
})

_VOWELS = frozenset("aeiou")


def _undouble_or_restore_e(stem: str, known_lemmas: set[str] | None = None) -> str:
    """Restores silent 'e' or undoubles doubled final consonant after stripping -ing/-ed/-er."""
    if known_lemmas:
        if stem in known_lemmas:
            return stem
        if len(stem) >= 3 and stem[-1] == stem[-2] and stem[:-1] in known_lemmas:
            return stem[:-1]
        if (stem + "e") in known_lemmas:
            return stem + "e"
        if (stem[:-1] if len(stem) >= 3 and stem[-1] == stem[-2] else stem) in known_lemmas:
            return stem[:-1]
    # Pure phonological fallback when dictionary set is not passed:
    if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in ("l", "s", "z", "f"):
        return stem[:-1]
    if len(stem) >= 2 and stem[-1] not in _VOWELS and (
        stem.endswith(("at", "it", "ot", "ut", "ak", "ik", "ok", "uk", "av", "iv", "ov"))
    ):
        return stem + "e"
    return stem


def strip_affixes(
    word: str,
    known_lemmas: set[str] | None = None,
) -> tuple[str, str, InflectionTag]:
    """
    Deconstructs `word` into `(prefix, root_lemma, InflectionTag)`.
    Guaranteed deterministic O(1) execution.
    """
    w = word.lower().strip()
    if not w or w in BARE_EXCEPTIONS:
        return ("", w, InflectionTag.BARE)

    if w in IRREGULAR_LEMMAS:
        lemma, tag = IRREGULAR_LEMMAS[w]
        return ("", lemma, tag)

    if known_lemmas and w in known_lemmas and not w.endswith(("ing", "ed", "ies", "s")):
        return ("", w, InflectionTag.BARE)

    # 1. Gerund / Present Participle (-ing)
    if len(w) > 4 and w.endswith("ing"):
        stem = w[:-3]
        if stem.endswith("y") and len(stem) >= 2:
            return ("", stem, InflectionTag.GERUND_ING)
        lemma = _undouble_or_restore_e(stem, known_lemmas)
        return ("", lemma, InflectionTag.GERUND_ING)

    # 2. Past Tense (-ed)
    if len(w) > 4 and w.endswith("ied"):
        return ("", w[:-3] + "y", InflectionTag.PAST_ED)
    if len(w) > 3 and w.endswith("ed"):
        stem = w[:-2]
        lemma = _undouble_or_restore_e(stem, known_lemmas)
        return ("", lemma, InflectionTag.PAST_ED)

    # 3. Adverb (-ly)
    if len(w) > 4 and w.endswith("ily"):
        return ("", w[:-3] + "y", InflectionTag.ADVERB_LY)
    if len(w) > 4 and w.endswith("ly"):
        stem = w[:-2]
        if not known_lemmas or stem in known_lemmas:
            return ("", stem, InflectionTag.ADVERB_LY)

    # 4. Superlative (-est) & Comparative (-er)
    if len(w) > 5 and w.endswith("iest"):
        return ("", w[:-4] + "y", InflectionTag.SUPERLATIVE_EST)
    if len(w) > 5 and w.endswith("est"):
        stem = _undouble_or_restore_e(w[:-3], known_lemmas)
        if not known_lemmas or stem in known_lemmas:
            return ("", stem, InflectionTag.SUPERLATIVE_EST)

    # 5. Plural / 3rd-Person Singular (-ies, -es, -s)
    if len(w) > 4 and w.endswith("ies"):
        return ("", w[:-3] + "y", InflectionTag.PLURAL_S)
    if len(w) > 4 and w.endswith(("ches", "shes", "xes", "zes", "sses")):
        return ("", w[:-2], InflectionTag.PLURAL_S)
    if len(w) > 2 and w.endswith("s") and not w.endswith(("ss", "us", "is", "os")):
        stem = w[:-1]
        if not known_lemmas or stem in known_lemmas:
            return ("", stem, InflectionTag.PLURAL_S)

    return ("", w, InflectionTag.BARE)
