"""
Slice 00: Sub-Atomic Linguistic & Engine Data Contracts.
Pure immutable and __slots__ structs with zero external dependencies.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag


class POS(IntEnum):
    UNK = 0
    NOUN = 1
    VERB = 2
    ADJ = 3
    ADV = 4
    PREP = 5
    DET = 6
    PRON = 7
    CONJ = 8
    INTJ = 9


class InflectionTag(IntEnum):
    BARE = 0
    PLURAL_S = 1          # cats, ovaries
    THIRD_SING_S = 2      # runs, auctions
    GERUND_ING = 3        # running, auditioning
    PAST_ED = 4           # walked, auctioned
    PAST_PART = 5         # taken, broken
    COMPARATIVE_ER = 6    # bigger, faster
    SUPERLATIVE_EST = 7   # biggest, fastest
    ADVERB_LY = 8         # definitely, reluctantly


class DomainMask(IntFlag):
    NONE = 0
    FOOD = 1 << 0         # cooking, edible, kitchen, meal, grain, fruit
    BODY = 1 << 1         # anatomy, organ, tissue, fluid, orifice
    MEDICAL = 1 << 2      # clinical, pathology, surgery, disease, reproductive
    CRIME = 1 << 3        # law, violence, illicit, auction, trafficking
    KIDS = 1 << 4         # child, school, family, toy, play, youth
    TECH = 1 << 5         # computer, hardware, network, phone, device
    COMMERCE = 1 << 6     # money, trade, sell, bid, market, office
    MOTION = 1 << 7       # run, walk, physical locomotion, travel
    TIME = 1 << 8         # schedule, late, hour, morning, clock
    HOUSEHOLD = 1 << 9    # bathroom, lamp, bed, pillow, room, clean
    EMOTION = 1 << 10     # love, hate, happy, cry, fear, desire
    INTIMATE = 1 << 11    # sexual, suggestive, romance, bedfellow, arousal
    ANIMALS = 1 << 12     # horse, cat, badger, dog, mammal, bird
    SOUND = 1 << 13       # ring, music, voice, bass, call, noise
    SOCIAL = 1 << 14      # friend, companion, community, public, marriage


@dataclass(frozen=True, slots=True)
class MeaningVec:
    """6-Axis deterministic semantic & connotation vector."""
    clinical: int = 0       # 0..100: anatomical, forensic, Latinate medical register
    visceral: int = 0       # 0..100: taboo, bodily, suggestive, embarrassing shock
    formality: int = 50     # 0..100: 0=slang/vulgar, 50=everyday, 100=archaic/legal
    concreteness: int = 50  # 0..100: 0=abstract concept, 100=tangible physical object
    valence: int = 0        # -100..+100: negative to positive emotional polarity
    rarity: int = 20        # 0..100: inverse corpus frequency
    domains: DomainMask = DomainMask.NONE


@dataclass(frozen=True, slots=True)
class SynsetNode:
    """Single WordNet-style meaning node in the deterministic semantic graph."""
    synset_id: int
    pos: POS
    gloss: str
    lemmas: tuple[str, ...]
    hypernym_ids: tuple[int, ...] = ()
    hyponym_ids: tuple[int, ...] = ()
    antonym_lemmas: tuple[str, ...] = ()
    meaning_vec: MeaningVec = field(default_factory=MeaningVec)


@dataclass(slots=True)
class TokenNode:
    """Single word slot on the 64-token SentenceTape."""
    slot_index: int
    raw_typed: str
    current_text: str
    trailing_sep: str = " "
    lemma: str = ""
    prefix: str = ""
    inflection: InflectionTag = InflectionTag.BARE
    case_profile: int = 0
    pos: POS = POS.UNK
    phonemes: tuple[str, ...] = ()
    intended_synset_id: int = -1
    meaning_vec: MeaningVec = field(default_factory=MeaningVec)
    mutated_count: int = 0
    applied_solver: str = ""
    last_utility_score: float = 0.0
    locked: bool = False


@dataclass(frozen=True, slots=True)
class DependencyArc:
    """Syntactic relation linking two TokenNodes on the SentenceTape."""
    head_slot: int
    dep_slot: int
    rel_type: str  # SUBJ_VERB, VERB_OBJ, MOD_NOUN, PREP_OBJ, DETERMINER


@dataclass(frozen=True, slots=True)
class MutationHypo:
    """A candidate synthetic corruption produced by a Layer 5 Solver."""
    target_slots: tuple[int, ...]
    replacement_words: tuple[str, ...]
    solver_name: str
    form_tightness: float      # 0..100: phonetic, typo, or synonym closeness
    grammar_fit: float         # 0..100: POS & inflectional agreement
    collision_score: float     # 0..100: semantic/domain contrast shock
    total_utility: float       # weighted deterministic objective score
    explanation_trace: str


@dataclass(frozen=True, slots=True)
class RewriteAction:
    """Executable atomic edit plan emitted to the OS mutator port."""
    backspaces: int
    inject_text: str
    target_slots: tuple[int, ...]
    solver_name: str
    utility_score: float
    is_retroactive: bool
    trace: str
    latency_us: float = 0.0


@dataclass(frozen=True, slots=True)
class KeyEvent:
    """Cross-platform normalized hardware keystroke event."""
    key_code: int
    char: str
    is_press: bool
    shift: bool = False
    ctrl: bool = False
    alt: bool = False
    meta: bool = False
    timestamp_ns: int = 0
