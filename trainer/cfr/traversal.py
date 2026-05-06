"""External-sampling Monte-Carlo CFR traversal (Brown et al. 2019).

One traversal:
    Pick a `traverser` ∈ {SB, BB} for this whole tree.
    Recurse from the current Env state, returning the traverser's
    *expected utility* at that state.

At each node:

  * Terminal:
        Return the traverser's payoff from `env.payoffs_bb()`.

  * Traverser-to-act:
        Compute the current strategy σ via regret matching from
        AdvNet(traverser).
        For EACH legal action a, recurse on env.clone().step(a) → v(s, a).
        Compute expected value V(s) = Σ σ(a) · v(s, a).
        Compute regrets r(s, a) = v(s, a) - V(s) for all legal a.
        Insert (x, full_regret_vec, t) into adv_buf[traverser].
        Insert (x, full_strategy_vec, t) into policy_buf as well — the
        average strategy must be sampled from EVERY state the traverser
        visits along the way (this is the strategy-collection rule that
        makes the policy net converge to the average strategy).
        Return V(s) to caller.

  * Opponent-to-act (or chance):
        Sample one action a from σ (regret-matched from AdvNet(opponent)).
        Recurse on env.step(a). Return the recursive value.
        We do NOT add a regret entry here (only the traverser's regrets are
        accumulated this iteration). We DO add to policy_buf — Deep CFR
        collects strategies for the policy network from BOTH players'
        decisions across all traversals (so the policy net sees both seats).

Notes:
    * The traversal returns a *float* (the traverser's expected utility),
      not a vector. The vector is used only at the traverser's decision
      point to compute the regret target.
    * Linear CFR weighting is applied at training time (the iteration `t`
      is stored alongside each entry; the trainer multiplies the loss by
      `t / Σ_t t'` per sample). The traversal itself only records `t`.
    * `env` is mutated by step() — we always clone before stepping into
      a branch we'll need to leave behind. The opponent branch (single
      sample) doesn't need cloning; we step the env in place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import pokertrainer_engine as pte

from .buffers import ReservoirBuffer
from .regret_matching import regret_matching_np


NUM_ACTIONS  = 11
NUM_PLAYERS  = 2


@dataclass
class TraversalNets:
    """Per-player advantage networks. Both must be in inference mode and on `device`.

    The recursion calls the net once per visited state. Single-state CPU
    forward is the throughput bottleneck of single-process Deep CFR — at
    laptop scale that's ~5k forwards per hand. The optional TorchScript
    compilation in `build()` gives ~1.5-2x speedup by removing Python
    interpreter overhead on small MLPs.
    """
    adv_net_per_player: list[torch.nn.Module]
    device: torch.device

    @classmethod
    def build(cls, nets: list[torch.nn.Module], device: torch.device,
              script: bool = True) -> "TraversalNets":
        """Move nets to device, set inference mode, optionally TorchScript them."""
        out: list[torch.nn.Module] = []
        for n in nets:
            n.train(False)
            n.to(device)
            if script:
                try:
                    n = torch.jit.script(n)
                except Exception:
                    pass   # fall back to eager — not critical
            out.append(n)
        return cls(adv_net_per_player=out, device=device)


def _legal_mask_from_obs(obs) -> np.ndarray:
    """Build a NUM_ACTIONS-dim 0/1 float mask from the engine's legal list."""
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for at in obs.legal:
        mask[int(at)] = 1.0
    return mask


@torch.no_grad()
def _predict_regrets(net: torch.nn.Module,
                     x: np.ndarray,
                     device: torch.device) -> np.ndarray:
    """Single-state forward of an AdvNet. Returns NUM_ACTIONS-dim float32."""
    xt = torch.from_numpy(x).to(device).unsqueeze(0)
    r = net(xt).squeeze(0).cpu().numpy().astype(np.float32, copy=False)
    return r


def _full_strategy_vec(legal_idx_to_sigma: dict, mask: np.ndarray) -> np.ndarray:
    """Pack per-legal-action probabilities back into a NUM_ACTIONS-dim vector
    aligned with the action-type integer index (so illegal slots are 0)."""
    out = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for at, p in legal_idx_to_sigma.items():
        out[at] = p
    # Sanity: should sum to 1 over legal entries (or 0 if every legal action's
    # regret was non-positive AND the fallback-uniform put mass on legal).
    return out


# Default depth cap. Matches the engine's HIST_MAX (encoder truncates beyond
# this anyway, so the obs.x at deeper states is lossy — recursing further is
# both expensive AND uses incomplete state). Override per-run via the trainer
# CLI knob `--max-depth`.
DEFAULT_MAX_DEPTH = 34


# Regret-target scaling. Terminal HU NLHE payoffs are in BB units (±100 BB on
# shove paths), so raw regret targets r(s,a) = v(s,a) - V(s) reach magnitudes
# of order 100. With those raw targets the AdvNet's MSE loss sits at ~1000
# and gradient norms regularly hit 1500-5000 — the grad clip at 10 fires
# every step, biasing each Adam update's direction.
#
# We divide stored regrets by REGRET_SCALE so the network learns 1/100-scale
# values; loss drops by 10^4, grads drop by 10^4, and the clip stops firing.
# Regret matching (`regret_matching_np`) is invariant under positive rescaling
# of its input, so the σ derived from the AdvNet is unchanged. Only the
# depth-cap value bootstrap needs to multiply by REGRET_SCALE to reconvert
# the scaled prediction to chip units before mixing with terminal payoffs.
REGRET_SCALE = 100.0


def traverse(env: pte.Env,
             traverser: int,
             nets: TraversalNets,
             adv_buf_traverser: ReservoirBuffer,
             policy_buf: ReservoirBuffer,
             t: int,
             rng: np.random.Generator,
             max_depth: int = DEFAULT_MAX_DEPTH) -> float:
    """Recursive external-sampling traversal. Returns traverser's expected utility.

    `traverser` is 0 (SB) or 1 (BB). The recursion mutates `env` only along
    the opponent branch; traverser branches use `env.clone()` so we can
    revisit each one.

    `max_depth` caps recursion based on the engine's history size. Reaching
    the cap returns the AdvNet-estimated value σ-weighted-average of predicted
    regrets, which is the standard depth-limited-solving baseline. This is
    what stops a min-raise-war traversal from ballooning to 10^15 leaves
    during early-iteration training when σ is near-uniform — the per-traversal
    wall time goes from "indistinguishable from hung" to "bounded by B^max_depth".
    """
    if env.is_terminal():
        return float(env.payoffs_bb()[traverser])

    obs       = env.observation()
    actor     = int(env.to_act())
    x         = obs.x.astype(np.float32, copy=False)
    legal     = list(obs.legal)                    # list[ActionType]
    n_legal   = len(legal)
    mask      = _legal_mask_from_obs(obs)
    adv_net   = nets.adv_net_per_player[actor]

    pred_r    = _predict_regrets(adv_net, x, nets.device)
    sigma     = regret_matching_np(pred_r, mask)   # full NUM_ACTIONS vector

    # Always record the strategy at this state into the policy buffer:
    # Deep CFR's average strategy is collected from EVERY decision point,
    # both seats, every traversal. This is what makes the final PolicyNet
    # converge to the time-averaged strategy.
    policy_buf.add(x, sigma, t)

    # Depth cap: substitute the network's value estimate for further recursion.
    # We don't store an adv_buf entry here — the regret target would be 0 by
    # construction (action_values = pred_r → V(s) = Σσ·pred_r → r(a) = pred_r-V(s);
    # the network is being asked to predict a target that's a function of
    # itself, which is degenerate). Skipping the write is the safe choice.
    #
    # pred_r is in scaled-regret units (see REGRET_SCALE comment). The value
    # we return must be in chip units so it integrates correctly with terminal
    # payoffs at higher recursion levels — multiply back by REGRET_SCALE.
    if env.state().history_size >= max_depth:
        return float((sigma * pred_r * mask).sum()) * REGRET_SCALE

    if actor == traverser:
        # Branch every legal action, compute v(s, a), accumulate regret.
        action_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for ai, at in enumerate(legal):
            child = env.clone()
            # step by ActionType (more robust than legal_idx after clone).
            child.step_action(at)
            v_a = traverse(child, traverser, nets,
                           adv_buf_traverser, policy_buf, t, rng,
                           max_depth=max_depth)
            action_values[int(at)] = v_a

        # V(s) = Σ σ(a) · v(s, a)  — only legal actions contribute (σ is 0 elsewhere).
        v_state = float((sigma * action_values).sum())
        # Regrets only defined for legal actions; illegal slots remain 0.
        # Scale by 1/REGRET_SCALE before storing — see comment on REGRET_SCALE.
        regrets = (action_values - v_state) * mask / REGRET_SCALE
        adv_buf_traverser.add(x, regrets.astype(np.float32, copy=False), t)
        return v_state

    # Opponent: sample one action from σ (restricted to legal indices).
    # We renormalize over legal positions to be safe against numerical drift.
    legal_int = np.array([int(at) for at in legal], dtype=np.int64)
    legal_probs = sigma[legal_int]
    z = legal_probs.sum()
    if z <= 0.0:
        # Fallback: uniform legal (shouldn't happen since regret_matching_np
        # already falls back, but guard against fp underflow).
        legal_probs = np.full(n_legal, 1.0 / n_legal, dtype=np.float32)
    else:
        legal_probs = legal_probs / z
    chosen_local_idx = int(rng.choice(n_legal, p=legal_probs))

    env.step(chosen_local_idx)   # in-place: opponent branch is single-path
    return traverse(env, traverser, nets,
                    adv_buf_traverser, policy_buf, t, rng,
                    max_depth=max_depth)


def run_traversals(env_seed: int,
                   n_traversals: int,
                   traverser: int,
                   nets: TraversalNets,
                   adv_buf_traverser: ReservoirBuffer,
                   policy_buf: ReservoirBuffer,
                   t: int,
                   rng: np.random.Generator,
                   max_depth: int = DEFAULT_MAX_DEPTH) -> dict:
    """Run K traversals as `traverser` (0=SB, 1=BB). Each starts with a fresh
    env.reset() drawing a new hand. Returns a stats dict.
    """
    env = pte.Env(seed=env_seed)
    sum_v = 0.0
    for _ in range(n_traversals):
        env.reset()
        v = traverse(env, traverser, nets,
                     adv_buf_traverser, policy_buf, t, rng,
                     max_depth=max_depth)
        sum_v += v
    return {
        "n_traversals":    n_traversals,
        "mean_traverser_v": sum_v / max(1, n_traversals),
        "adv_buf_n_seen":   adv_buf_traverser.n_seen,
        "adv_buf_n_filled": adv_buf_traverser.n_filled,
        "pol_buf_n_seen":   policy_buf.n_seen,
        "pol_buf_n_filled": policy_buf.n_filled,
    }
