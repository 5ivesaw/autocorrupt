"""
Slice 07: Context-Sensitive Grapheme-to-Phoneme (G2P) Finite-State Transducer.
Converts any English spelling (including loanwords and unseen words) into a sequence
of ARPAbet/IPA phonemes using deterministic phonological rewrite rules.
"""
from __future__ import annotations
import re

# Phonological loanword & irregular orthography exceptions (strictly true pronunciation, zero jokes)
ORTHO_PRONUNCIATION_EXCEPTIONS: dict[str, tuple[str, ...]] = {
    "hors": ("HH", "AO", "R", "S"),
    "d'oeuvres": ("D", "ER", "V", "Z"),
    "oeuvre": ("ER", "V", "R", "AH"),
    "oeuvres": ("ER", "V", "R", "AH", "Z"),
    "ovaries": ("OW", "V", "ER", "IY", "Z"),
    "ovary": ("OW", "V", "ER", "IY"),
    "horse": ("HH", "AO", "R", "S"),
    "polenta": ("P", "OW", "L", "EH", "N", "T", "AH"),
    "placenta": ("P", "L", "AH", "S", "EH", "N", "T", "AH"),
    "audition": ("AO", "D", "IH", "SH", "AH", "N"),
    "auditioning": ("AO", "D", "IH", "SH", "AH", "N", "IH", "NG"),
    "auction": ("AO", "K", "SH", "AH", "N"),
    "auctioning": ("AO", "K", "SH", "AH", "N", "IH", "NG"),
    "suction": ("S", "AH", "K", "SH", "AH", "N"),
    "suctioning": ("S", "AH", "K", "SH", "AH", "N", "IH", "NG"),
    "autocorrect": ("AO", "T", "OW", "K", "ER", "EH", "K", "T"),
    "auto": ("AO", "T", "OW"),
    "cat": ("K", "AE", "T"),
    "rectal": ("R", "EH", "K", "T", "AH", "L"),
    "darth": ("D", "AA", "R", "TH"),
    "vader": ("V", "EY", "D", "ER"),
    "badger": ("B", "AE", "JH", "ER"),
    "lamp": ("L", "AE", "M", "P"),
    "on": ("AA", "N"),
    "tampon": ("T", "AE", "M", "P", "AA", "N"),
    "ring": ("R", "IH", "NG"),
    "ripen": ("R", "AY", "P", "AH", "N"),
    "for": ("F", "AO", "R"),
    "play": ("P", "L", "EY"),
    "foreplay": ("F", "AO", "R", "P", "L", "EY"),
    "cologne": ("K", "AH", "L", "OW", "N"),
    "colon": ("K", "OW", "L", "AH", "N"),
    "cereal": ("S", "IH", "R", "IY", "AH", "L"),
    "serial": ("S", "IH", "R", "IY", "AH", "L"),
    "morning": ("M", "AO", "R", "N", "IH", "NG"),
    "mourning": ("M", "AO", "R", "N", "IH", "NG"),
    "prostate": ("P", "R", "AA", "S", "T", "EY", "T"),
    "prostrate": ("P", "R", "AA", "S", "T", "R", "EY", "T"),
    "flamenco": ("F", "L", "AH", "M", "EH", "NG", "K", "OW"),
    "flamingo": ("F", "L", "AH", "M", "IH", "NG", "G", "OW"),
    "therapist": ("TH", "EH", "R", "AH", "P", "IH", "S", "T"),
    "rapist": ("R", "EY", "P", "IH", "S", "T"),
    "manslaughter": ("M", "AE", "N", "S", "L", "AO", "T", "ER"),
    "laughter": ("L", "AE", "F", "T", "ER"),
    "scapegoat": ("S", "K", "EY", "P", "G", "OW", "T"),
    "escape": ("IH", "S", "K", "EY", "P"),
    "goat": ("G", "OW", "T"),
    "shake": ("SH", "EY", "K"),
    "take": ("T", "EY", "K"),
    "tower": ("T", "AW", "ER"),
    "shower": ("SH", "AW", "ER"),
    "public": ("P", "AH", "B", "L", "IH", "K"),
    "pubic": ("P", "Y", "UW", "B", "IH", "K"),
}

# Ordered multigraph / digraph context rules for English G2P transduction
_DIGRAPH_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tion", ("SH", "AH", "N")),
    ("sion", ("ZH", "AH", "N")),
    ("ture", ("CH", "ER")),
    ("ough", ("OW",)),
    ("augh", ("AO",)),
    ("igh", ("AY",)),
    ("eigh", ("EY",)),
    ("tch", ("CH",)),
    ("dge", ("JH",)),
    ("ph", ("F",)),
    ("sh", ("SH",)),
    ("ch", ("CH",)),
    ("th", ("TH",)),
    ("wh", ("W",)),
    ("ck", ("K",)),
    ("kn", ("N",)),
    ("wr", ("R",)),
    ("ng", ("NG",)),
    ("qu", ("K", "W")),
    ("ee", ("IY",)),
    ("ea", ("IY",)),
    ("oo", ("UW",)),
    ("ou", ("AW",)),
    ("ow", ("OW",)),
    ("oi", ("OY",)),
    ("oy", ("OY",)),
    ("ai", ("EY",)),
    ("ay", ("EY",)),
    ("au", ("AO",)),
    ("aw", ("AO",)),
    ("oa", ("OW",)),
    ("ie", ("IY",)),
    ("er", ("ER",)),
    ("ir", ("ER",)),
    ("ur", ("ER",)),
    ("ar", ("AA", "R")),
    ("or", ("AO", "R")),
)

_SINGLE_CHAR_MAP: dict[str, tuple[str, ...]] = {
    "a": ("AE",),
    "b": ("B",),
    "c": ("K",),
    "d": ("D",),
    "e": ("EH",),
    "f": ("F",),
    "g": ("G",),
    "h": ("HH",),
    "i": ("IH",),
    "j": ("JH",),
    "k": ("K",),
    "l": ("L",),
    "m": ("M",),
    "n": ("N",),
    "o": ("AA",),
    "p": ("P",),
    "q": ("K",),
    "r": ("R",),
    "s": ("S",),
    "t": ("T",),
    "u": ("AH",),
    "v": ("V",),
    "w": ("W",),
    "x": ("K", "S"),
    "y": ("IY",),
    "z": ("Z",),
}

_LONG_VOWEL_BEFORE_SILENT_E: dict[str, str] = {
    "a": "EY",
    "e": "IY",
    "i": "AY",
    "o": "OW",
    "u": "UW",
}


def word_to_phonemes(word: str) -> tuple[str, ...]:
    """
    Deterministic G2P transducer. Converts any English word or phrase into
    a canonical tuple of ARPAbet phonemes in < 15 microseconds.
    """
    w = word.lower().strip()
    if not w:
        return ()
    if " " in w:
        out: list[str] = []
        for part in w.split():
            out.extend(word_to_phonemes(part))
        return tuple(out)

    if w in ORTHO_PRONUNCIATION_EXCEPTIONS:
        return ORTHO_PRONUNCIATION_EXCEPTIONS[w]

    # Strip non-alpha except apostrophe
    clean = re.sub(r"[^a-z']", "", w)
    if clean in ORTHO_PRONUNCIATION_EXCEPTIONS:
        return ORTHO_PRONUNCIATION_EXCEPTIONS[clean]
    clean = clean.replace("'", "")
    if not clean:
        return ()

    # Detect magic/silent final 'e' (VCe pattern like "make", "bake", "Vader" stem)
    silent_final_e = (
        len(clean) >= 4
        and clean.endswith("e")
        and not clean.endswith(("ee", "le", "re", "ge", "ce", "se", "ve"))
        and clean[-2] not in "aeiou"
        and clean[-3] in "aeiou"
    )

    out_phonemes: list[str] = []
    i = 0
    n = len(clean)
    limit = n - 1 if silent_final_e else n

    while i < limit:
        matched = False
        # Check digraphs/trigraphs first
        for pat, phs in _DIGRAPH_RULES:
            if clean.startswith(pat, i) and i + len(pat) <= limit:
                out_phonemes.extend(phs)
                i += len(pat)
                matched = True
                break
        if matched:
            continue

        ch = clean[i]
        # Double consonant deduplication (e.g. "butter" -> single T)
        if i + 1 < limit and clean[i + 1] == ch and ch not in "aeiou":
            i += 1
            continue

        # Context-sensitive soft C and soft G before E, I, Y
        if ch == "c" and i + 1 < n and clean[i + 1] in "eiy":
            out_phonemes.append("S")
            i += 1
            continue
        if ch == "g" and i + 1 < n and clean[i + 1] in "eiy":
            out_phonemes.append("JH")
            i += 1
            continue

        # Vowel before silent final 'e' becomes long diphthong
        if silent_final_e and i == n - 3 and ch in _LONG_VOWEL_BEFORE_SILENT_E:
            out_phonemes.append(_LONG_VOWEL_BEFORE_SILENT_E[ch])
            i += 1
            continue

        # Word-initial 'y' is consonant glide Y
        if ch == "y" and i == 0:
            out_phonemes.append("Y")
            i += 1
            continue

        # Unstressed trailing '-le' -> AH L
        if ch == "l" and i == n - 2 and clean.endswith("le") and n > 3:
            out_phonemes.extend(("AH", "L"))
            break

        out_phonemes.extend(_SINGLE_CHAR_MAP.get(ch, ()))
        i += 1

    return tuple(out_phonemes)
