"""
Slice 10: Deterministic Levenshtein-1 & SymSpell Automaton.
Intersects the SymSpell symmetric-delete index and physical Keyboard Geometry (Slice 02)
to discover all real dictionary words within 1 edit (substitution, deletion, insertion,
or adjacent transposition) of any single word or fused 2-word juncture in under 40 microseconds.
"""
from __future__ import annotations
from dataclasses import dataclass
from autocorrupt.core.atoms.slice_02_key_geometry import key_distance
from autocorrupt.core.perception.slice_09_mmap_trie import MmapLexiconStore


@dataclass(frozen=True, slots=True)
class OrthoNeighbor:
    word: str
    word_id: int
    edit_type: str       # "key_slip", "delete_1", "insert_1", "transpose", "sub_1"
    form_similarity: float  # 0..100


def generate_1_deletes(word: str) -> set[str]:
    """Generates the word itself plus all 1-character deletions."""
    w = word.lower()
    out = {w}
    for i in range(len(w)):
        out.add(w[:i] + w[i + 1:])
    return out


def find_orthographic_neighbors(
    query: str,
    store: MmapLexiconStore,
    allow_same_length_sub: bool = True,
) -> list[OrthoNeighbor]:
    """
    Finds all real English words in `store` within Levenshtein/Damerau distance 1
    of `query` using O(1) SymSpell symmetric delete intersection.
    Zero linear scan of the dictionary.
    """
    q = query.lower().strip()
    if len(q) < 2:
        return []

    candidate_ids: set[int] = set()
    for d_key in generate_1_deletes(q):
        ids = store.symspell_index.get(d_key)
        if ids:
            candidate_ids.update(ids)

    neighbors: list[OrthoNeighbor] = []
    q_len = len(q)

    for wid in candidate_ids:
        entry = store.entries[wid]
        cand = entry.word
        if cand == q:
            continue
        c_len = len(cand)
        if abs(c_len - q_len) > 1:
            continue

        # Case 1: Same length (1 substitution or 1 adjacent transposition)
        if c_len == q_len and allow_same_length_sub:
            diffs = [i for i in range(q_len) if q[i] != cand[i]]
            if len(diffs) == 1:
                idx = diffs[0]
                k_dist = key_distance(q[idx], cand[idx])
                if k_dist <= 1.55:
                    # Physical adjacent key slip on QWERTY (e.g., sec->sex, regards->retards, kind->kink)
                    sim = max(78.0, 96.0 - k_dist * 8.0)
                    neighbors.append(OrthoNeighbor(cand, wid, "key_slip", sim))
                else:
                    # Single letter substitution (e.g., lampon -> tampon, Vader -> badger stem)
                    neighbors.append(OrthoNeighbor(cand, wid, "sub_1", 82.0))
            elif len(diffs) == 2 and diffs[1] == diffs[0] + 1:
                i0, i1 = diffs[0], diffs[1]
                if q[i0] == cand[i1] and q[i1] == cand[i0]:
                    # Adjacent letter transposition (e.g., santa->satan, marital->martial, united->untied)
                    neighbors.append(OrthoNeighbor(cand, wid, "transpose", 91.0))

        # Case 2: 1 character deleted from query (e.g., public->pubic, shirt->shit, clock->cock, canal->anal)
        elif c_len == q_len - 1:
            # Verify cand is a subsequence of q missing exactly 1 character
            i = j = 0
            skips = 0
            while i < q_len and j < c_len:
                if q[i] == cand[j]:
                    i += 1
                    j += 1
                else:
                    skips += 1
                    i += 1
                    if skips > 1:
                        break
            if skips <= 1:
                neighbors.append(OrthoNeighbor(cand, wid, "delete_1", 88.0))

        # Case 3: 1 character inserted into query (e.g., condo->condom, pen->peen)
        elif c_len == q_len + 1:
            i = j = 0
            skips = 0
            while i < q_len and j < c_len:
                if q[i] == cand[j]:
                    i += 1
                    j += 1
                else:
                    skips += 1
                    j += 1
                    if skips > 1:
                        break
            if skips <= 1:
                neighbors.append(OrthoNeighbor(cand, wid, "insert_1", 86.0))

    return neighbors
