"""The 4 continuation strategies of Trick 1 (depth-limit leaf valuation).
Pluribus idea → "at each leaf, every player picks one of 4 continuation
strategies" — blueprint + fold/call/raise-biased · doc §2 (Trick 1), §9.

A continuation strategy is just a function ``(state) -> σ`` used to play out the
rest of the hand below a depth-limit leaf. The blueprint is one; the three
biased ones are built by taking the blueprint's σ at a node and multiplying the
probability of a chosen action family (fold / call / raise) by a constant, then
renormalizing. Cheap to build, and giving the *opponent* the choice among them
at the leaf is what makes a shallow cut honest (the search can no longer assume a
docile opponent). See search.py for how the choice is folded into the CFR solve.
"""
from __future__ import annotations

import numpy as np

from .config import FOLD, CHECK_CALL, ALL_IN, NUM_ACTIONS

# Action "families" a bias can emphasize. RAISE = every raise/all-in slot.
_RAISE_SLOTS = tuple(range(2, NUM_ACTIONS))   # RAISE_25 .. ALL_IN


def bias_strategy(sigma: np.ndarray, family: str, factor: float) -> np.ndarray:
    """Multiply the probability mass of `family` by `factor`, renormalize.

    `family` ∈ {"blueprint", "fold", "call", "raise"}. "blueprint" returns σ
    unchanged. The result keeps σ's support (no action is zeroed), so beliefs
    never collapse — a deviation the search can still see."""
    if family == "blueprint" or sigma.sum() <= 0:
        return sigma
    if family == "fold":
        slots = (FOLD,)
    elif family == "call":
        slots = (CHECK_CALL,)
    elif family == "raise":
        slots = _RAISE_SLOTS
    else:
        raise ValueError(f"unknown bias family {family!r}")
    out = sigma.astype(np.float64).copy()
    for a in slots:
        out[a] *= factor
    z = out.sum()
    return out / z if z > 0 else sigma


# The canonical 4 continuations of the paper (order fixed for reproducibility).
CONTINUATIONS: tuple[str, ...] = ("blueprint", "fold", "call", "raise")


class ContinuationSet:
    """The K continuation policies derived from a blueprint, as σ-providers.

    Each continuation `k` is a callable ``state -> σ`` (NUM_ACTIONS), namely the
    blueprint's average strategy at `state`, biased toward family `families[k]`
    by `factor`. Used to roll out / value the hand below a depth-limit leaf."""

    def __init__(self, blueprint, factor: float = 5.0,
                 families: tuple[str, ...] = CONTINUATIONS):
        self.blueprint = blueprint
        self.factor = factor
        self.families = families

    def __len__(self) -> int:
        return len(self.families)

    def sigma(self, k: int, state) -> np.ndarray:
        base = self.blueprint.strategy(state)
        return bias_strategy(base, self.families[k], self.factor)
