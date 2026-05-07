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
from cfr.models import AdvNet, PolicyNet
from cfr.regret_matching import regret_matching_np


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


# ─── Deep CFR policies ──────────────────────────────────────────────────────

def _legal_mask_from_obs(obs) -> np.ndarray:
    """Build a NUM_ACTIONS-dim 0/1 mask aligned to ActionType integer index."""
    mask = np.zeros(int(pte.NUM_ACTIONS), dtype=np.float32)
    for at in obs.legal:
        mask[int(at)] = 1.0
    return mask


def _legal_local_idx_from_action_type(obs, sampled_at: int) -> int:
    """Map ActionType integer → local index in obs.legal. Falls back to 0."""
    for i, at in enumerate(obs.legal):
        if int(at) == sampled_at:
            return i
    return 0


class CFRAdvPolicy:
    """Deep CFR policy from a per-player pair of AdvNets, via regret matching.

    This is the "current strategy" — what the traversal uses internally. It is
    NOT the deployable average strategy; for that, use CFRPolicyNetPolicy below.

    `match.py` dispatches to `choose_with_seat` automatically (via hasattr),
    passing the integer seat (0=SB, 1=BB) so the right per-player AdvNet is used.

    Supports both greedy (argmax of regret-matched strategy) and stochastic
    (sample from regret-matched strategy) action selection.
    """

    def __init__(self,
                 adv_net_per_player: list[AdvNet],
                 device: torch.device,
                 stochastic: bool = True,
                 name: str = "cfr_adv"):
        assert len(adv_net_per_player) == 2
        self.adv_net = adv_net_per_player
        for n in self.adv_net:
            n.train(False)
        self.device = device
        self.stochastic = stochastic
        self.name = name

    @torch.no_grad()
    def choose_with_seat(self, obs, seat: int, rng: np.random.Generator) -> int:
        if len(obs.legal) == 1:
            return 0
        mask = _legal_mask_from_obs(obs)
        x = torch.from_numpy(obs.x).to(self.device).unsqueeze(0)
        regrets = self.adv_net[seat](x).squeeze(0).cpu().numpy()
        sigma = regret_matching_np(regrets, mask)         # NUM_ACTIONS-dim
        legal_int = np.array([int(at) for at in obs.legal], dtype=np.int64)
        legal_probs = sigma[legal_int]
        z = legal_probs.sum()
        if z <= 0:
            legal_probs = np.full(len(obs.legal), 1.0 / len(obs.legal),
                                  dtype=np.float32)
        else:
            legal_probs = legal_probs / z
        if self.stochastic:
            return int(rng.choice(len(obs.legal), p=legal_probs))
        return int(legal_probs.argmax())


class CFRPolicyNetPolicy:
    """Deployable Deep CFR policy: samples actions from the trained PolicyNet.

    PolicyNet's `forward(x, mask)` returns a probability distribution over
    `NUM_ACTIONS`. We map back to local legal-action indices and sample
    (or argmax) from that distribution.
    """

    def __init__(self,
                 policy_net: PolicyNet,
                 device: torch.device,
                 stochastic: bool = True,
                 name: str = "cfr_policy"):
        self.net = policy_net
        self.net.train(False)
        self.device = device
        self.stochastic = stochastic
        self.name = name

    @torch.no_grad()
    def choose(self, obs, rng: np.random.Generator) -> int:
        if len(obs.legal) == 1:
            return 0
        mask = _legal_mask_from_obs(obs)
        x = torch.from_numpy(obs.x).to(self.device).unsqueeze(0)
        m = torch.from_numpy(mask).to(self.device).unsqueeze(0)
        probs = self.net(x, m).squeeze(0).cpu().numpy()    # NUM_ACTIONS-dim
        legal_int = np.array([int(at) for at in obs.legal], dtype=np.int64)
        legal_probs = probs[legal_int]
        z = legal_probs.sum()
        if z <= 0:
            legal_probs = np.full(len(obs.legal), 1.0 / len(obs.legal),
                                  dtype=np.float32)
        else:
            legal_probs = legal_probs / z
        if self.stochastic:
            return int(rng.choice(len(obs.legal), p=legal_probs))
        return int(legal_probs.argmax())


