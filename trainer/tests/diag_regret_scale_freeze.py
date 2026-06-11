"""Why did BB freeze at uniform? Isolate regret-scale vs weight_decay vs trunk.

The iter-17 cluster run had BB stuck at exactly 50% call on every hand. The
question (a good one): regret matching AND Adam are both scale-invariant, so
dividing regrets by 100 instead of 10 should not, on its own, change learning.
The ONLY thing that breaks scale-invariance is weight_decay (a fixed absolute
pull to zero that doesn't scale with the targets).

This builds a realistic BB-vs-jam regret buffer — for each of the 169 BB hands,
many noisy (call_value, fold_value) samples against a uniform-random SB jam —
and trains a fresh AdvNet under several (regret_scale, weight_decay) settings,
then reports whether the net learns to DISCRIMINATE hands (call AA, fold 72o)
or stays flat. That tells us which knob actually unfreezes BB.

Run:
    PYTHONPATH=engine/build:trainer python trainer/tests/diag_regret_scale_freeze.py
"""
from __future__ import annotations

import time
import numpy as np
import torch
from torch import nn

import os
import pokertrainer_engine as pte

torch_threads = os.cpu_count() or 4

from cfr.config import CFRModelConfig
from cfr.models import AdvNet
from cfr.regret_matching import regret_matching_np
from cfr.tokenize import tokenize_state, pad_batch
from cfr.pushfold_validation import _seed_table
from evaluate.preflop_equity import N_CLASSES, LABEL_TO_ID, COMBOS, class_of
from evaluate.pushfold_solver import solve_pushfold
from evaluate.preflop_equity import build_equity_matrix

_FOLD, _CALL, _ALL_IN = 0, 1, 10
NUM_ACTIONS = 11
STACK_BB = 10.0


def build_bb_buffer(samples_per_hand: int, seed: int = 7):
    """For each BB class: the BB-vs-jam tok_state + many noisy regret targets.

    SB is forced all-in (a uniform-random hand); BB chooses FOLD/CALL with the
    current (uniform) strategy. Per sample, with v = 0.5*call+0.5*fold:
        regret[FOLD] = fold_value - v ;  regret[CALL] = call_value - v
    call_value = +10/-10/0 (showdown, matched 10bb), fold_value = -1 (blind).
    Returns (tokens, dpos, raw_regrets) with regrets UNSCALED (divide later).
    """
    rng = np.random.default_rng(seed)
    ev = pte.HandEvaluator.load_or_generate("")
    bb_seeds = _seed_table(1)
    env = pte.Env(0x5EED, int(STACK_BB * 100))

    tok_list, raw_targets, cls_list = [], [], []
    for c in range(N_CLASSES):
        env.reset(int(bb_seeds[c]))
        bb_hole = [int(x) for x in env.state().hole[1]]
        obs = env.observation()
        ai = [int(a) for a in obs.legal].index(_ALL_IN)
        env.step(ai)
        ts = tokenize_state(env.state(), hero_seat=1)
        # sample SB hands + boards avoiding BB's two cards
        for _ in range(samples_per_hand):
            deck = [x for x in range(52) if x not in bb_hole]
            pick = rng.choice(len(deck), size=7, replace=False)
            cards = [deck[i] for i in pick]
            sb = cards[:2]; board = cards[2:]
            r_bb = ev.evaluate7([bb_hole[0], bb_hole[1], *board])
            r_sb = ev.evaluate7([sb[0], sb[1], *board])
            call_value = 0.0 if r_bb == r_sb else (10.0 if r_bb < r_sb else -10.0)
            fold_value = -1.0
            v = 0.5 * call_value + 0.5 * fold_value
            reg = np.zeros(NUM_ACTIONS, dtype=np.float32)
            reg[_FOLD] = fold_value - v
            reg[_CALL] = call_value - v
            tok_list.append(ts); raw_targets.append(reg); cls_list.append(c)
    return tok_list, np.stack(raw_targets), np.array(cls_list)


def train(tok_list, raw_targets, *, regret_scale, weight_decay, steps=2000,
          batch=256, lr=1e-3, d_model=64, n_layers=2, seed=0):
    torch.manual_seed(seed)
    cfg = CFRModelConfig(); cfg.d_model = d_model; cfg.n_layers = n_layers
    cfg.n_heads = 4; cfg.d_ff = 4 * d_model
    net = AdvNet(cfg); net.train(True)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    targets = (raw_targets / regret_scale).astype(np.float32)
    n = len(tok_list)
    rng = np.random.default_rng(seed)
    for step in range(steps):
        idx = rng.integers(0, n, size=batch)
        toks, pad, dpos = pad_batch([tok_list[i] for i in idx])
        pred = net(torch.from_numpy(toks), torch.from_numpy(pad),
                   torch.from_numpy(dpos))
        tgt = torch.from_numpy(targets[idx])
        loss = ((pred - tgt) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return net, cfg


def read_call_freq(net, cfg):
    """Net's call freq (1-σ[FOLD]) per BB class, one batched forward."""
    bb_seeds = _seed_table(1)
    env = pte.Env(0x5EED, int(STACK_BB * 100))
    toks_in, masks = [], []
    for c in range(N_CLASSES):
        env.reset(int(bb_seeds[c]))
        obs = env.observation()
        ai = [int(a) for a in obs.legal].index(_ALL_IN)
        env.step(ai)
        toks_in.append(tokenize_state(env.state(), hero_seat=1))
        m = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for a in env.observation().legal:
            if int(a) in (_FOLD, _CALL):
                m[int(a)] = 1.0
        masks.append(m)
    net.train(False)
    t, p, d = pad_batch(toks_in)
    with torch.no_grad():
        r = net(torch.from_numpy(t), torch.from_numpy(p), torch.from_numpy(d)).numpy()
    sig = np.stack([regret_matching_np(r[k], masks[k]) for k in range(N_CLASSES)])
    return 1.0 - sig[:, _FOLD]


def main():
    torch.set_num_threads(torch_threads)
    print(f"[diag] torch threads={torch_threads}")
    print("[diag] building BB-vs-jam buffer (169 hands)…", flush=True)
    t0 = time.time()
    tok_list, raw_targets, cls = build_bb_buffer(samples_per_hand=60)
    print(f"  buffer: {len(tok_list)} samples in {time.time()-t0:.0f}s. "
          f"raw |regret| mean={np.abs(raw_targets).mean():.2f} "
          f"(/100 -> {np.abs(raw_targets).mean()/100:.3f}, /10 -> {np.abs(raw_targets).mean()/10:.3f})")

    # Reference: oracle BB call range (so we can score discrimination).
    ev = pte.HandEvaluator.load_or_generate("")
    E, C = build_equity_matrix(ev, 2_000_000, seed=99)
    oracle = solve_pushfold(E, C, STACK_BB, iters=3000)
    orc_call = oracle.bb_call
    w = COMBOS.astype(float)

    def score(call):
        agree = float((w * ((call >= 0.5) == (orc_call >= 0.5))).sum() / w.sum())
        return agree

    hands = ["AA", "KK", "QQ", "AKo", "ATo", "KJs", "T9s", "72o", "32o"]
    configs = [
        ("scale=100 wd=1e-3  (the iter-17 setting)", dict(regret_scale=100, weight_decay=1e-3)),
        ("scale=10  wd=1e-3  (my committed fix)   ", dict(regret_scale=10,  weight_decay=1e-3)),
        ("scale=100 wd=0     (isolate weight_decay)", dict(regret_scale=100, weight_decay=0.0)),
    ]
    print(f"\n{'config':44s} | agree | " + " ".join(f"{h:>4}" for h in hands))
    print("-" * 110)
    for label, kw in configs:
        t1 = time.time()
        net, cfg = train(tok_list, raw_targets, **kw)
        call = read_call_freq(net, cfg)
        row = " ".join(f"{call[LABEL_TO_ID[h]]*100:4.0f}" for h in hands)
        spread = call.max() - call.min()
        print(f"{label:44s} | {score(call)*100:4.0f}% | {row}   "
              f"[spread={spread*100:.0f}pts, {time.time()-t1:.0f}s]", flush=True)
    print("\noracle target:" + " " * 31 + " |       | "
          + " ".join(f"{orc_call[LABEL_TO_ID[h]]*100:4.0f}" for h in hands))


if __name__ == "__main__":
    main()
