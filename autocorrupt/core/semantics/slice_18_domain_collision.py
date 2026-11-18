"""
Slice 18: Semantic & Domain Collision Tensor.
Calculates the deterministic comedic/cursed shock score (`collision_score` in 0..100)
when injecting a candidate replacement with `cand_vec` into a sentence with `context_domains`
in place of `orig_vec`.
"""
from __future__ import annotations
from autocorrupt.core.atoms.slice_00_types import DomainMask, MeaningVec

# High-shock domain collision pairs: (Sentence Context Domain, Injected Word Domain) -> bonus
HIGH_SHOCK_COLLISIONS: tuple[tuple[DomainMask, DomainMask, float], ...] = (
    (DomainMask.FOOD, DomainMask.BODY, 42.0),       # e.g. polenta/chicken -> placenta, salad -> ovaries
    (DomainMask.FOOD, DomainMask.MEDICAL, 38.0),    # e.g. salad -> ovaries, eat -> ingest
    (DomainMask.KIDS, DomainMask.COMMERCE, 44.0),   # e.g. auditioning kids -> auctioning kids
    (DomainMask.KIDS, DomainMask.MEDICAL, 40.0),    # e.g. kids -> suctioning kids
    (DomainMask.KIDS, DomainMask.INTIMATE, 48.0),   # e.g. for the play -> foreplay
    (DomainMask.KIDS, DomainMask.ANIMALS, 35.0),    # e.g. Darth Vader pillow -> Darth badger pillow
    (DomainMask.HOUSEHOLD, DomainMask.MEDICAL, 45.0),  # e.g. lamp on in bathroom -> tampon
    (DomainMask.HOUSEHOLD, DomainMask.BODY, 42.0),  # e.g. shirt -> shit, clock -> cock
    (DomainMask.TECH, DomainMask.BODY, 42.0),       # e.g. autocorrect -> auto cat rectal
    (DomainMask.TECH, DomainMask.FOOD, 36.0),       # e.g. phone waiting to ring -> ripen (banana phone)
    (DomainMask.TIME, DomainMask.MOTION, 34.0),     # e.g. running late -> jogging late
    (DomainMask.FOOD, DomainMask.EMOTION, 34.0),    # e.g. cold brew -> unfriendly brew
    (DomainMask.TECH, DomainMask.EMOTION, 32.0),    # e.g. hard drive -> difficult drive
    (DomainMask.COMMERCE, DomainMask.INTIMATE, 44.0),  # e.g. condo -> condom, client -> clitoris
    (DomainMask.SOCIAL, DomainMask.INTIMATE, 40.0),
)


def compute_domain_collision(
    orig_vec: MeaningVec,
    cand_vec: MeaningVec,
    sentence_domain_mask: DomainMask,
) -> float:
    """
    Computes a deterministic score in [0.0, 100.0] measuring how strongly `cand_vec`
    derails or subverts the sentence's register and semantic domain while contrasting
    with `orig_vec`.
    """
    score = 15.0

    # 1. Visceral / Taboo & Clinical Escalation delta
    visc_delta = max(0, cand_vec.visceral - orig_vec.visceral)
    clin_delta = max(0, cand_vec.clinical - orig_vec.clinical)
    score += visc_delta * 0.48 + clin_delta * 0.38

    # 2. Cross-Domain Incongruity against the active sentence context
    ctx_doms = sentence_domain_mask | orig_vec.domains
    cand_doms = cand_vec.domains

    # If the candidate brings in a completely foreign domain not in the original word
    new_doms = cand_doms & ~orig_vec.domains
    if new_doms:
        score += 18.0

    for ctx_flag, cand_flag, bonus in HIGH_SHOCK_COLLISIONS:
        if (ctx_doms & ctx_flag) and (cand_doms & cand_flag) and not (orig_vec.domains & cand_flag):
            score += bonus

    # 3. Valence Polarity Inversion (turning positive/neutral into negative/awkward or vice versa)
    valence_shift = abs(cand_vec.valence - orig_vec.valence)
    score += valence_shift * 0.22

    # 4. Concreteness Clash (e.g. replacing an abstract time/state verb with a hyper-concrete physical noun/verb)
    conc_shift = abs(cand_vec.concreteness - orig_vec.concreteness)
    score += conc_shift * 0.18

    return min(100.0, score)
