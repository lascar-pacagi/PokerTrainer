"""Policy abstraction for evaluation matches.

Every policy answers a single question: given an observation, which index
into `obs.legal` do you play? That's the only thing the match runner calls.

Concrete policies:
  * RandomPolicy         — uniform over legal actions.
  * CallingStationPolicy — never folds, never raises. Call/check whenever
                           possible; if all-in-or-fold, call. (A classic
                           "passive fish" baseline.)
  * CheckFoldPolicy      — folds to any bet, checks when no bet. A punching
                           bag baseline; any trained net should crush it.
  * ModelPolicy          — wraps a DMCNet, greedy over legal-action values.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

import pokertrainer_engine as pte

from dmc.models import DMCNet


class Policy(Protocol):
    name: str
    def choose(self, obs, rng: np.random.Generator) -> int: ...


class RandomPolicy:
    name = "random"

    def choose(self, obs, rng: np.random.Generator) -> int:
        return int(rng.integers(0, len(obs.legal)))


class CallingStationPolicy:
    """Never folds, never raises. Picks CHECK_CALL if legal, else the lowest
    chip-cost non-fold action (which will be CHECK_CALL when facing no bet,
    and ALL_IN only when call is not legal — which in HU can't happen as long
    as both players have chips).
    """
    name = "calling_station"

    def choose(self, obs, rng: np.random.Generator) -> int:
        legal = list(obs.legal)
        if pte.ActionType.CHECK_CALL in legal:
            return legal.index(pte.ActionType.CHECK_CALL)
        # Fallback: if check/call isn't in legal (shouldn't happen in HU NLHE
        # except terminal-all-in races), pick the first non-FOLD action.
        for i, a in enumerate(legal):
            if a != pte.ActionType.FOLD:
                return i
        return 0


class CheckFoldPolicy:
    """Check when possible, fold to any bet. Dead money — a sanity lower bound."""
    name = "check_fold"

    def choose(self, obs, rng: np.random.Generator) -> int:
        legal = list(obs.legal)
        # If no to_call (check is legal), take CHECK_CALL.
        # to_call == 0 iff CHECK_CALL costs 0 iff we're "checking". We detect
        # that by inspecting the action row's bet_to_bb vs our already-invested
        # this-street amount. Simpler: if FOLD is NOT legal, there's no bet to
        # call → CHECK_CALL acts as check.
        if pte.ActionType.FOLD not in legal:
            return legal.index(pte.ActionType.CHECK_CALL)
        return legal.index(pte.ActionType.FOLD)


class ModelPolicy:
    """Greedy policy over a DMCNet's legal-action values. No exploration."""

    def __init__(self, net: DMCNet, device: torch.device, name: str = "model"):
        self.net = net
        self.device = device
        self.name = name
        self.net.train(False)

    @torch.no_grad()
    def choose(self, obs, rng: np.random.Generator) -> int:
        if len(obs.legal) == 1:
            return 0
        x = torch.from_numpy(obs.x).to(self.device)
        a = torch.from_numpy(obs.a).to(self.device)
        vals = self.net.score_legal(x, a)
        return int(torch.argmax(vals).item())
