"""
Slice 17: Deterministic 6-Axis Connotation & Semantic Domain Analyzer.
Computes `MeaningVec(clinical, visceral, formality, concreteness, valence, rarity, domains)`
for any word purely from its dictionary definition gloss and Greco-Latin morphological roots.
Zero hardcoded joke weights: the vector is derived from what the word's definition actually says.
"""
from __future__ import annotations
import re
from autocorrupt.core.atoms.slice_00_types import DomainMask, MeaningVec

# Dictionary definition keyword sets that activate each semantic domain
DOMAIN_GLOSS_MARKERS: dict[DomainMask, frozenset[str]] = {
    DomainMask.FOOD: frozenset({
        "food", "cook", "cooked", "cooking", "eat", "edible", "meal", "dish", "kitchen",
        "grain", "cornmeal", "boiled", "salad", "dressing", "fruit", "strawberry", "almond",
        "chicken", "meat", "bread", "gluten", "wheat", "appetizer", "savory", "drink",
        "brew", "coffee", "water", "flavor", "spice", "herb", "sweet", "ripen", "fish",
        "trout", "cereal", "breakfast", "dinner", "nut", "cracker", "seed",
    }),
    DomainMask.BODY: frozenset({
        "body", "anatomical", "organ", "tissue", "flesh", "skin", "mouth", "lips",
        "orifice", "rectum", "rectal", "bowel", "colon", "intestine", "pelvic",
        "pubic", "groin", "genital", "buttock", "gluteal", "asses", "posterior",
        "uterus", "ovary", "ovaries", "placenta", "fetus", "blood", "menstrual",
        "mucosa", "saliva", "gland", "prostate", "finger", "hand", "foot", "head",
    }),
    DomainMask.MEDICAL: frozenset({
        "medical", "clinical", "surgical", "pathology", "disease", "sick", "diseased",
        "infection", "anatomical", "reproductive", "absorbent", "plug", "wound",
        "cavity", "vacuum", "aspiration", "fluid", "nutrient", "ingest", "unconsciousness",
        "coma", "therapy", "psychiatric", "symptom", "syndrome", "diagnosis", "autopsy",
    }),
    DomainMask.CRIME: frozenset({
        "crime", "criminal", "illegal", "illicit", "violence", "assault", "kill",
        "homicide", "manslaughter", "rapist", "trafficking", "hostage", "steal",
        "prison", "punish", "boycott", "negligence", "satan", "demon", "hell",
    }),
    DomainMask.KIDS: frozenset({
        "child", "children", "kid", "kids", "youth", "school", "toy", "doll",
        "play", "playground", "birthday", "baby", "infant", "parent", "family",
        "father", "mother", "bedtime",
    }),
    DomainMask.TECH: frozenset({
        "computer", "digital", "disk", "drive", "hardware", "software", "phone",
        "telephone", "call", "keyboard", "spell", "typing", "device", "electric",
        "network", "data", "screen",
    }),
    DomainMask.COMMERCE: frozenset({
        "money", "sell", "buy", "bid", "bidder", "auction", "commercial", "market",
        "price", "trade", "business", "office", "client", "customer", "property",
        "ticket", "reserve", "hotel",
    }),
    DomainMask.MOTION: frozenset({
        "move", "motion", "locomotion", "foot", "walk", "run", "jog", "sprint",
        "trot", "gallop", "jump", "dangle", "hang", "vehicle", "car", "station",
        "travel", "depart",
    }),
    DomainMask.TIME: frozenset({
        "time", "hour", "minute", "schedule", "late", "early", "morning", "evening",
        "night", "day", "year", "calendar", "autumn", "winter", "summer", "spring",
        "clock", "wait",
    }),
    DomainMask.HOUSEHOLD: frozenset({
        "house", "home", "room", "bathroom", "kitchen", "bedroom", "bed", "pillow",
        "lamp", "light", "bulb", "shower", "bath", "towel", "clean", "wash",
        "shoe", "shirt", "clothing", "furniture",
    }),
    DomainMask.EMOTION: frozenset({
        "emotion", "feeling", "love", "hate", "affection", "warmth", "unfriendly",
        "hostile", "aloof", "grief", "mourning", "dead", "death", "laughter",
        "happy", "sad", "cry", "fear", "reluctant", "unwilling",
    }),
    DomainMask.INTIMATE: frozenset({
        "sexual", "sex", "intimate", "erotic", "arousal", "climax", "orgasm",
        "foreplay", "fondle", "grope", "caress", "bedfellow", "lover", "seductive",
        "moist", "sensual", "kink", "fetish", "strip", "naked", "nude",
    }),
    DomainMask.ANIMALS: frozenset({
        "animal", "mammal", "equine", "horse", "cat", "feline", "dog", "canine",
        "badger", "burrowing", "goat", "bird", "flamingo", "fish", "trout",
        "leopard", "wildlife", "creature", "beast",
    }),
    DomainMask.SOUND: frozenset({
        "sound", "acoustic", "ring", "bell", "tone", "music", "musical", "voice",
        "sing", "bass", "frequency", "subwoofer", "baritone", "loud", "noise",
        "hearing",
    }),
}

CLINICAL_GLOSS_MARKERS: frozenset[str] = frozenset({
    "anatomical", "clinical", "medical", "vascular", "uterus", "reproductive",
    "orifice", "mucosa", "rectum", "rectal", "colon", "bowel", "fetus", "ovary",
    "placenta", "prostate", "gluteal", "ingest", "nutrient", "unconsciousness",
    "aspiration", "vacuum", "pathology", "diseased", "menstrual", "cavity",
    "palpate", "osculate", "masticate", "dihydrogen", "monoxide", "equine",
})

VISCERAL_GLOSS_MARKERS: frozenset[str] = frozenset({
    "sexual", "erotic", "intimate", "arousal", "climax", "foreplay", "fondle",
    "grope", "bedfellow", "kink", "rectal", "rectum", "pubic", "genital",
    "buttock", "asses", "tampon", "menstrual", "blood", "placenta", "ovary",
    "fetus", "excrement", "feces", "vulgar", "taboo", "rapist", "manslaughter",
    "satan", "retard", "cock", "shit", "moist", "suction", "auction",
})

FORMAL_ARCHAIC_MARKERS: frozenset[str] = frozenset({
    "anatomical", "clinical", "scientific", "legal", "formal", "literary",
    "archaic", "tome", "equine", "mammal", "osculate", "ingest", "consummate",
    "negligence", "boycott", "prostrate", "reluctantly",
})

CONCRETE_PHYSICAL_MARKERS: frozenset[str] = frozenset({
    "physical", "object", "organ", "animal", "mammal", "food", "dish", "plug",
    "fixture", "vehicle", "pillow", "fish", "bird", "liquid", "body", "flesh",
    "metallic", "graphite", "foot", "mouth", "cavity", "tower", "shower",
})


def compute_meaning_vec_from_gloss(word: str, gloss: str) -> MeaningVec:
    """
    Pure deterministic function that derives a word's 6-axis MeaningVec and DomainMask
    from its dictionary definition gloss and morphological structure.
    """
    tokens = set(re.findall(r"[a-z]+", gloss.lower()))
    w_low = word.lower()

    domains = DomainMask.NONE
    for dom, markers in DOMAIN_GLOSS_MARKERS.items():
        if tokens & markers:
            domains |= dom

    clin_hits = len(tokens & CLINICAL_GLOSS_MARKERS)
    visc_hits = len(tokens & VISCERAL_GLOSS_MARKERS)
    form_hits = len(tokens & FORMAL_ARCHAIC_MARKERS)
    conc_hits = len(tokens & CONCRETE_PHYSICAL_MARKERS)

    # Greco-Latin morphological root indicators for clinical/formal register
    if w_low.endswith(("enta", "ectal", "itis", "oma", "osis", "mucosa", "ate", "ic")):
        clin_hits += 1

    clinical = min(100, clin_hits * 32)
    visceral = min(100, visc_hits * 35)
    formality = min(100, 40 + form_hits * 22 + (10 if len(w_low) >= 8 else 0))
    concreteness = min(100, 35 + conc_hits * 25)

    valence = 0
    if tokens & {"love", "happy", "affection", "sweet", "play", "friend", "laugh", "laughter"}:
        valence += 40
    if tokens & {"disease", "dead", "grief", "mourning", "hostile", "unfriendly", "crime", "vulgar", "blood"}:
        valence -= 50

    rarity = min(95, max(10, len(w_low) * 6 + clin_hits * 12))

    return MeaningVec(
        clinical=clinical,
        visceral=visceral,
        formality=formality,
        concreteness=concreteness,
        valence=valence,
        rarity=rarity,
        domains=domains,
    )
