"""Stage-1 training validation: the net's push/fold ranges vs the Nash oracle.

═══════════════════════════════════════════════════════════════════════════════
WHAT THIS ANSWERS
═══════════════════════════════════════════════════════════════════════════════

The AA/72o-vs-random probe (cfr/probe.py) can only tell you the policy is
*hand-aware* in a coarse, two-point way. At a 10 bb push/fold stage we can do
much better: there is a known-correct equilibrium (evaluate/pushfold_solver.py),
so we can read the net's *entire* jam/call range over all 169 starting hands and
score it against the solved Nash oracle.

Two things are read directly from the AdvNets (no self-play, no sampling noise):

  SB open-jam:  at the 10 bb SB-open node, σ[ALL_IN] for each of 169 hands.
  BB call-vs-jam: force SB all-in, then 1 − σ[FOLD] at the BB node for each hand.

These are printed as 13×13 grids (so you literally watch the jam frontier form
across iterations) plus three scalars per side: combo-weighted decision agreement
with the oracle, mean-absolute frequency error, and the net's range size vs the
oracle's. A flat or speckled grid ⇒ still hand-agnostic; convergence shows up as
the grid sharpening toward the oracle and agreement climbing toward 100%.

Cost: 169 + 169 single-state forwards, batched into two GPU calls. Negligible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import pokertrainer_engine as pte

from .regret_matching import regret_matching_np
from .tokenize import tokenize_state, pad_batch

from evaluate.preflop_equity import (N_CLASSES, CLASS_LABELS, COMBOS, class_of,
                                     grid_cell_class)
from evaluate.pushfold_solver import PushFoldOracle, format_grid

NUM_ACTIONS = 11
_FOLD, _CALL, _ALL_IN = 0, 1, 10


# ═══════════════════════════════════════════════════════════════════════════
# SEED TABLES — one dealt seed per starting-hand class
# ═══════════════════════════════════════════════════════════════════════════
#
# The hole cards produced by env.reset(seed) depend only on the seed, not on
# the starting stack, so we can find a representative seed per class ONCE and
# reuse it across every probe interval (and across stacks). We need one table
# per seat: the SB-jam grid wants class c in seat 0, the BB-call grid wants
# class c in seat 1 (with SB's hand irrelevant — SB is forced all-in). Cached.

_SEED_CACHE: dict[int, np.ndarray] = {}


def _build_seed_table(seat: int, max_find: int = 400_000) -> np.ndarray:
    """seeds[c] = a seed whose seat-`seat` hole is class c. -1 if not found."""
    seeds = np.full(N_CLASSES, -1, dtype=np.int64)
    remaining = N_CLASSES
    env = pte.Env(0x5EED)
    s = 0
    while remaining > 0 and s < max_find:
        env.reset(s)
        hole = env.state().hole[seat]
        c = class_of(int(hole[0]), int(hole[1]))
        if seeds[c] < 0:
            seeds[c] = s
            remaining -= 1
        s += 1
    if remaining > 0:
        missing = [CLASS_LABELS[i] for i in range(N_CLASSES) if seeds[i] < 0]
        raise RuntimeError(f"no seat-{seat} seed for {remaining} classes: {missing[:8]}…")
    return seeds


def _seed_table(seat: int) -> np.ndarray:
    if seat not in _SEED_CACHE:
        _SEED_CACHE[seat] = _build_seed_table(seat)
    return _SEED_CACHE[seat]


# ═══════════════════════════════════════════════════════════════════════════
# READ THE NET'S RANGES
# ═══════════════════════════════════════════════════════════════════════════


def _masked_sigmas(net: torch.nn.Module,
                   tok_states: list,
                   masks: list[np.ndarray],
                   device: torch.device) -> np.ndarray:
    """One batched forward → regret-matched σ per row. Returns (n, NUM_ACTIONS)."""
    tokens_np, pad_mask_np, dpos_np = pad_batch(tok_states)
    with torch.no_grad():
        tokens   = torch.from_numpy(tokens_np).to(device)
        pad_mask = torch.from_numpy(pad_mask_np).to(device)
        dpos     = torch.from_numpy(dpos_np).to(device)
        regrets = net(tokens, pad_mask, dpos).cpu().numpy()
    return np.stack([regret_matching_np(regrets[k], masks[k])
                     for k in range(len(tok_states))], axis=0)


def _legal_mask(legal, allowed) -> np.ndarray:
    m = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for at in legal:
        a = int(at)
        if allowed is None or a in allowed:
            m[a] = 1.0
    return m


@dataclass
class PushFoldReport:
    net_jam: np.ndarray      # (169,) SB jam freq
    net_call: np.ndarray     # (169,) BB call freq
    oracle: PushFoldOracle

    def _stats(self, net: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
        w = COMBOS.astype(np.float64)
        wsum = w.sum()
        decision_agree = float((w * ((net >= 0.5) == (ref >= 0.5))).sum() / wsum)
        mae = float((w * np.abs(net - ref)).sum() / wsum)
        net_pct = float((w * net).sum() / wsum)
        return decision_agree, mae, net_pct

    def summary_lines(self, iter_t: int | None = None) -> list[str]:
        prefix = f"[pushfold iter={iter_t}]" if iter_t is not None else "[pushfold]"
        sj_agree, sj_mae, sj_pct = self._stats(self.net_jam, self.oracle.sb_jam)
        bc_agree, bc_mae, bc_pct = self._stats(self.net_call, self.oracle.bb_call)
        return [
            f"  {prefix} SB-jam : agree={sj_agree*100:5.1f}%  mae={sj_mae:.3f}  "
            f"net={sj_pct*100:4.1f}% vs oracle={self.oracle.sb_jam_pct*100:4.1f}%",
            f"  {' '*len(prefix)} BB-call: agree={bc_agree*100:5.1f}%  mae={bc_mae:.3f}  "
            f"net={bc_pct*100:4.1f}% vs oracle={self.oracle.bb_call_pct*100:4.1f}%",
        ]

    def grids(self) -> str:
        g = np.array([self.net_jam[grid_cell_class(r, c)]
                      for r in range(13) for c in range(13)]).reshape(13, 13)
        gc = np.array([self.net_call[grid_cell_class(r, c)]
                       for r in range(13) for c in range(13)]).reshape(13, 13)
        return (format_grid(g, "  net SB open-jam %:") + "\n\n"
                + format_grid(gc, "  net BB call-vs-jam %:"))


def run_pushfold_validation(adv_nets: list[torch.nn.Module],
                            device: torch.device,
                            oracle: PushFoldOracle,
                            *,
                            starting_stack_chips: int,
                            allowed: frozenset | None) -> PushFoldReport:
    """Read the AdvNets' SB-jam and BB-call ranges over all 169 hands."""
    sb_seeds = _seed_table(0)     # class c in seat 0 (SB)
    bb_seeds = _seed_table(1)     # class c in seat 1 (BB)
    env = pte.Env(0x5EED, starting_stack_chips)

    sb_tok, sb_masks = [], []
    bb_tok, bb_masks = [], []
    for c in range(N_CLASSES):
        # ── SB open node: SB (seat 0) holds class c, empty history ─────────
        env.reset(int(sb_seeds[c]))
        obs = env.observation()
        sb_tok.append(tokenize_state(env.state(), hero_seat=0))
        sb_masks.append(_legal_mask(obs.legal, allowed))

        # ── BB-vs-jam node: BB (seat 1) holds class c; force SB all-in ─────
        env.reset(int(bb_seeds[c]))
        obs = env.observation()
        ai_local = [int(a) for a in obs.legal].index(_ALL_IN)
        env.step(ai_local)
        obs_bb = env.observation()
        bb_tok.append(tokenize_state(env.state(), hero_seat=1))
        bb_masks.append(_legal_mask(obs_bb.legal, allowed))

    was_training = [n.training for n in adv_nets]
    for n in adv_nets:
        n.train(False)
    sb_sigma = _masked_sigmas(adv_nets[0], sb_tok, sb_masks, device)
    bb_sigma = _masked_sigmas(adv_nets[1], bb_tok, bb_masks, device)
    for n, t in zip(adv_nets, was_training):
        n.train(t)

    net_jam = sb_sigma[:, _ALL_IN]
    net_call = 1.0 - bb_sigma[:, _FOLD]      # only FOLD/CALL legal vs a jam
    return PushFoldReport(net_jam=net_jam, net_call=net_call, oracle=oracle)
