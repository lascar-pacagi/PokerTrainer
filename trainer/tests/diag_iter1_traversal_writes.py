"""Replay a handful of iter-1 traversals with random-init AdvNets and
count adv_writes per traverser-side. Goal: localize the 46x buffer
asymmetry observed in the cluster log.

The driver here is essentially the cfr_coro inner loop but with N=1
coroutine (no concurrency) and an explicit per-traversal report.

Run:
    PYTHONPATH=engine/build:trainer python trainer/tests/diag_iter1_traversal_writes.py
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

import pokertrainer_engine as pte

from cfr.config import CFRConfig
from cfr.models import AdvNet
from cfr.regret_matching import regret_matching_np
from cfr.traversal import DEFAULT_MAX_DEPTH


NUM_ACTIONS = int(pte.NUM_ACTIONS)


def _legal_mask(obs) -> np.ndarray:
    m = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for a in obs.legal:
        m[int(a)] = 1.0
    return m


def traverse(env, traverser: int, adv_nets, rng,
             adv_writes: list, pol_writes: list,
             slot_picks: list,
             depth: int = 0,
             max_depth: int = DEFAULT_MAX_DEPTH) -> float:
    """Single-process, non-coroutine version of cfr_coro.traverse_coro.

    Records adv_writes (only at traverser-actor nodes), pol_writes (at
    every visited node), and a `slot_picks` list of (actor, slot, depth)
    for the opponent's sampled actions — useful for understanding *how*
    the tree shrinks under degenerate opponent sigma.
    """
    if env.is_terminal():
        return float(env.payoffs_bb()[traverser])

    obs = env.observation()
    actor = int(env.to_act())
    mask = _legal_mask(obs)
    x = obs.x.astype(np.float32, copy=False).copy()
    legal = list(obs.legal)
    n_legal = len(legal)

    with torch.no_grad():
        xt = torch.from_numpy(x).unsqueeze(0)
        pred_r = adv_nets[actor](xt).squeeze(0).numpy()
    sigma = regret_matching_np(pred_r, mask)
    pol_writes.append((x, sigma))

    if env.state().history_size >= max_depth:
        return float((sigma * pred_r * mask).sum()) * 100.0  # REGRET_SCALE

    if actor == traverser:
        action_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for at in legal:
            child = env.clone()
            child.step_action(at)
            v_a = traverse(child, traverser, adv_nets, rng,
                           adv_writes, pol_writes, slot_picks,
                           depth + 1, max_depth)
            action_values[int(at)] = v_a
        v_state = float((sigma * action_values).sum())
        regrets = (action_values - v_state) * mask / 100.0
        adv_writes.append((x, regrets.astype(np.float32, copy=False)))
        return v_state

    # opponent samples
    legal_int = np.array([int(at) for at in legal], dtype=np.int64)
    probs = sigma[legal_int]
    z = probs.sum()
    if z > 0.0:
        probs = probs / z
    else:
        probs = np.full(n_legal, 1.0 / n_legal, dtype=np.float32)
    chosen = int(rng.choice(n_legal, p=probs))
    slot_picks.append((actor, int(legal[chosen]), depth))
    env.step(chosen)
    return traverse(env, traverser, adv_nets, rng,
                    adv_writes, pol_writes, slot_picks,
                    depth + 1, max_depth)


def main() -> None:
    cfg = CFRConfig()
    cfg.model.hidden = 1024
    cfg.model.n_layers = 10
    torch.manual_seed(42)
    advs = [AdvNet(cfg.model), AdvNet(cfg.model)]
    for net in advs:
        net.train(False)

    # Run a handful of traversals per traverser side and report stats.
    N_PER_SIDE = 8
    rng = np.random.default_rng(7)

    for traverser in (0, 1):
        side = "SB" if traverser == 0 else "BB"
        print(f"\n=== traverser={traverser} ({side}) — {N_PER_SIDE} traversals ===")
        total_writes = 0
        per_traversal_writes = []
        all_picks = []
        for k in range(N_PER_SIDE):
            env = pte.Env(0xC0FFEE + k)
            env.reset(k)
            adv_writes: list = []
            pol_writes: list = []
            slot_picks: list = []
            _v = traverse(env, traverser, advs, rng,
                          adv_writes, pol_writes, slot_picks)
            per_traversal_writes.append(len(adv_writes))
            total_writes += len(adv_writes)
            all_picks.extend(slot_picks)
        print(f"  total adv_writes        = {total_writes}")
        print(f"  per-traversal min/avg/max = "
              f"{min(per_traversal_writes)}/"
              f"{total_writes / N_PER_SIDE:.1f}/"
              f"{max(per_traversal_writes)}")
        # Summarize what the OPPONENT (not the traverser) sampled, by
        # action slot.  This is the "degenerate σ in action" view.
        opp_picks = [p for p in all_picks if p[0] == (1 - traverser)]
        from collections import Counter
        slot_count = Counter(p[1] for p in opp_picks)
        labels = ("F", "C", "R25", "R33", "R50", "R75",
                  "R100", "R150", "R200", "R300", "AI")
        print(f"  opponent picks (n={len(opp_picks)}):")
        for sl, lab in enumerate(labels):
            c = slot_count.get(sl, 0)
            if c > 0:
                print(f"    {lab:<5} {c}")


if __name__ == "__main__":
    main()
