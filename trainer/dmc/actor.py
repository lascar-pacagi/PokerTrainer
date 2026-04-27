"""Rollout loop: play hands with epsilon-greedy policy, write transitions to a buffer.

DMC target: pure Monte-Carlo — each transition's target value is the chip
delta in BB realised at the end of the *same* hand for the *same* player.
Because HU NLHE hands are short (<= ~20 decisions) and the reward is only
at terminal, we buffer per-player decisions during the hand and flush them
with the terminal return once the hand ends.

Exploration:
  * With probability epsilon, pick uniformly from legal actions.
  * Else, argmax of the network's value over legal actions.

A multi-process variant that writes to shared-memory buffers is a swap-in
upgrade — see DouZero dmc/dmc.py for the pattern.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

import pokertrainer_engine as pte

from .buffers import MCBuffer
from .config import ActorConfig
from .models import DMCNet


def schedule_epsilon(step: int, cfg: ActorConfig) -> float:
    """Linear decay from `epsilon_start` to `epsilon_end` over the first
    `epsilon_decay_steps` learner grad-steps. Flat at `epsilon_end` afterward.
    """
    if cfg.epsilon_decay_steps <= 0 or step >= cfg.epsilon_decay_steps:
        return cfg.epsilon_end
    frac = step / cfg.epsilon_decay_steps
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def schedule_reward_clip(step: int,
                         start: float | None,
                         end: float | None,
                         decay_steps: int) -> float | None:
    """Linear interpolation of the MC return clip over `decay_steps`.

    Use cases:
      * `start=25, end=None, decay_steps=50_000` — tighten early, then unclip
        for the late phase (the policy has learned aggression; let it now see
        full reward magnitudes to refine value estimates).
      * `start=25, end=25, decay_steps=*` — constant clip (back-compat).
      * `start=None, end=*` — same as no clip throughout (no schedule active).

    Returns the clip value to pass into `MCBuffer.set_reward_clip()`. None
    means "no clipping at this step".
    """
    if start is None:
        return None
    if decay_steps <= 0 or end is None or step >= decay_steps:
        # Past the decay window: hold at `end` (or `start` if no end given).
        return end if end is not None else start
    frac = step / decay_steps
    return start + frac * (end - start)


@dataclass
class PlayerTrajectory:
    xs: List[np.ndarray]
    as_: List[np.ndarray]  # the A_DIM vector of the action TAKEN (not all legal)

    def __init__(self):
        self.xs, self.as_ = [], []

    def add(self, x: np.ndarray, a: np.ndarray) -> None:
        self.xs.append(x.copy())
        self.as_.append(a.copy())

    def __len__(self) -> int:
        return len(self.xs)


def choose_action(obs, net: DMCNet, epsilon: float, device: torch.device,
                  rng: np.random.Generator,
                  ban_actions=()) -> int:
    """Return the chosen index into obs.legal.

    `ban_actions` is an iterable of `pte.ActionType` values to exclude from
    BOTH the random and greedy choice. Used by curriculum schedules
    (e.g. forbid ALL_IN preflop for the first N steps so the rollout buffer
    accumulates non-shove transitions and the network can learn
    bet-sizing-conditional EVs without the high-variance terminal escape).
    If every legal action is banned, fall through to argmax over all legal
    (curriculum can't override engine-required moves).
    """
    n_legal = len(obs.legal)
    if ban_actions:
        allowed = [i for i in range(n_legal) if obs.legal[i] not in ban_actions]
    else:
        allowed = list(range(n_legal))
    if not allowed:
        allowed = list(range(n_legal))

    if rng.random() < epsilon or len(allowed) == 1:
        return int(allowed[rng.integers(0, len(allowed))])

    x = torch.from_numpy(obs.x).to(device)
    a = torch.from_numpy(obs.a[allowed]).to(device)
    vals = net.score_legal(x, a)   # (len(allowed),)
    return int(allowed[int(torch.argmax(vals).item())])


def run_hand(env: pte.Env, net: DMCNet, buf: MCBuffer,
             epsilon: float, device: torch.device,
             rng: np.random.Generator,
             opponent_policy=None,
             learner_seat: pte.Player | None = None) -> dict:
    """Play one hand end-to-end; flush trajectory at terminal.

    Self-play (default): both seats use the network. Both trajectories go
    into the buffer.

    Train-against-baseline (opponent_policy is not None): the learner_seat
    uses the network; the other seat uses opponent_policy.choose(obs, rng).
    Only the learner's trajectory is buffered. The baseline's actions
    don't train the network, but they shape the state distribution the
    learner sees, which is the whole point.
    """
    traj = {pte.Player.SB: PlayerTrajectory(),
            pte.Player.BB: PlayerTrajectory()}

    while not env.is_terminal():
        actor = env.to_act()
        obs   = env.observation()
        if opponent_policy is not None and actor != learner_seat:
            idx = opponent_policy.choose(obs, rng)
        else:
            idx = choose_action(obs, net, epsilon, device, rng)
        taken_a_vec = obs.a[idx]
        # Only buffer the learner's transitions. In self-play that's both
        # seats; in train-against-baseline it's just the learner's seat.
        if opponent_policy is None or actor == learner_seat:
            traj[actor].add(obs.x, taken_a_vec)
        env.step(idx)

    payoffs = env.payoffs_bb()
    for p in (pte.Player.SB, pte.Player.BB):
        t = traj[p]
        if len(t) == 0:
            continue
        buf.append_trajectory(t.xs, t.as_, payoffs[int(p)])

    return {
        "len_sb": len(traj[pte.Player.SB]),
        "len_bb": len(traj[pte.Player.BB]),
        "payoff_sb_bb": payoffs[0],
    }


def rollout(env: pte.Env, net: DMCNet, buf: MCBuffer,
            n_hands: int, epsilon: float, device: torch.device,
            rng: np.random.Generator,
            opponent_policy=None) -> dict:
    """Play n_hands and append all transitions. Returns simple rollup stats.

    If opponent_policy is provided, the learner's seat alternates each
    hand (so it experiences both SB and BB equally). Otherwise self-play.
    """
    net.train(False)   # inference mode for rollouts
    stats = {"hands": 0, "transitions": 0, "sb_bb_sum": 0.0}
    for h in range(n_hands):
        env.reset()
        learner_seat = (pte.Player.SB if (h % 2 == 0) else pte.Player.BB) \
                       if opponent_policy is not None else None
        info = run_hand(env, net, buf, epsilon, device, rng,
                        opponent_policy=opponent_policy,
                        learner_seat=learner_seat)
        stats["hands"] += 1
        stats["transitions"] += info["len_sb"] + info["len_bb"]
        stats["sb_bb_sum"]   += info["payoff_sb_bb"]
    return stats
