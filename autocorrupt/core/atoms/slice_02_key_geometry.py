"""
Slice 02: Physical Keyboard Geometry & Euclidean Finger-Slip Metric.
Maps every key on physical QWERTY/Colemak layouts into 2D coordinates (in key pitch units,
including row stagger: +0.25 on home row, +0.75 on bottom row) for synthetic fat-finger analysis.
"""
from __future__ import annotations
import math

# Physical key coordinates (x_col_staggered, y_row) for standard ANSI/ISO QWERTY
QWERTY_COORDS: dict[str, tuple[float, float]] = {
    # Top letter row (y = 0.0)
    "q": (0.0, 0.0), "w": (1.0, 0.0), "e": (2.0, 0.0), "r": (3.0, 0.0), "t": (4.0, 0.0),
    "y": (5.0, 0.0), "u": (6.0, 0.0), "i": (7.0, 0.0), "o": (8.0, 0.0), "p": (9.0, 0.0),
    # Home letter row (y = 1.0, staggered +0.25 right)
    "a": (0.25, 1.0), "s": (1.25, 1.0), "d": (2.25, 1.0), "f": (3.25, 1.0), "g": (4.25, 1.0),
    "h": (5.25, 1.0), "j": (6.25, 1.0), "k": (7.25, 1.0), "l": (8.25, 1.0),
    # Bottom letter row (y = 2.0, staggered +0.75 right)
    "z": (0.75, 2.0), "x": (1.75, 2.0), "c": (2.75, 2.0), "v": (3.75, 2.0), "b": (4.75, 2.0),
    "n": (5.75, 2.0), "m": (6.75, 2.0),
}


def key_distance(c1: str, c2: str) -> float:
    """Euclidean physical key distance in key-cap units (1.0 = adjacent horizontal key)."""
    k1 = c1.lower()
    k2 = c2.lower()
    if k1 == k2:
        return 0.0
    p1 = QWERTY_COORDS.get(k1)
    p2 = QWERTY_COORDS.get(k2)
    if p1 is None or p2 is None:
        return 10.0
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def is_adjacent_key(c1: str, c2: str, radius: float = 1.55) -> bool:
    """Returns True if c1 and c2 are physical neighbor keys on the keyboard."""
    d = key_distance(c1, c2)
    return 0.0 < d <= radius


# Precompute O(1) adjacency lookup table at module load
ADJACENT_KEYS: dict[str, tuple[str, ...]] = {
    ch: tuple(
        other for other in QWERTY_COORDS
        if other != ch and is_adjacent_key(ch, other)
    )
    for ch in QWERTY_COORDS
}
