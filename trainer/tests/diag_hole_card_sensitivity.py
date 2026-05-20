"""Diagnostic: does a trained AdvNet differentiate hole cards at SB-open?

PROBE QUESTION
    For a fixed preflop-state-0 chip configuration (SB to act, 100bb stacks,
    blinds in, empty history), feed the trained AdvNet a sequence of x's
    that differ ONLY in the hole-card region. If the regret-matched sigma
    varies meaningfully across hole cards, the network IS reading hole
    cards and convergence is just slow. If sigma is essentially flat across
    hole cards, the network is ignoring them and no amount of training
    iterations will fix it.

WHY THIS IS DIAGNOSTIC
    obs.x is 816-dim; only 52 bits encode the hole cards (~6% of input).
    A trunk that doesn't amplify those 52 bits cannot produce hole-card-
    conditioned regret estimates regardless of how long you train. Looking
    at sigma spread across hands is the cheapest way to confirm/falsify.

RUN
    PYTHONPATH=engine/build:trainer python trainer/tests/diag_hole_card_sensitivity.py \
        --ckpt /path/to/cfr_iter_XXXX.ckpt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import pokertrainer_engine as pte

from cfr.models import AdvNet
from cfr.regret_matching import regret_matching_np
from cfr.tokenize import tokenize_state, pad_batch


# Hands to scan, grouped by category — gives us a clean comparison across
# strength tiers. Each entry: (label, target predicate over hole_pair).
def _ranks(hole_pair):
    return tuple(sorted((int(hole_pair[0]) >> 2,
                         int(hole_pair[1]) >> 2), reverse=True))

def _suited(hole_pair):
    return (int(hole_pair[0]) & 3) == (int(hole_pair[1]) & 3)

def _make_predicate(spec: str):
    """spec like 'AA', 'AKs', 'AKo', '72o', etc.

    rank chars: 2..9 T J Q K A   (T=10, A=12)
    's' = suited, 'o' = offsuit (and for pairs no suffix).
    """
    rank_chr_to_idx = {c: i for i, c in enumerate("23456789TJQKA")}
    r0 = rank_chr_to_idx[spec[0]]
    r1 = rank_chr_to_idx[spec[1]]
    if r0 == r1:                              # pocket pair
        def pred(hole_pair):
            return _ranks(hole_pair) == (r0, r0)
        return pred
    suited = (len(spec) >= 3 and spec[2] == 's')
    hi, lo = max(r0, r1), min(r0, r1)
    def pred(hole_pair):
        if _ranks(hole_pair) != (hi, lo):
            return False
        return _suited(hole_pair) == suited
    return pred


# A representative sweep spanning strong → trash.
HAND_SPECS = (
    "AA", "KK", "QQ", "JJ", "TT", "99", "66", "33", "22",
    "AKs", "AQs", "AJs", "A9s", "A5s", "A2s",
    "AKo", "AQo", "AJo", "A2o",
    "KQs", "KJs", "QJs", "JTs", "T9s", "98s", "76s", "54s",
    "KQo", "QJo", "T9o",
    "K2o", "Q2o", "J2o", "T2o",
    "72s", "72o", "32o", "23o",
)


def _find_seed(env, predicate, max_tries: int = 5000) -> int:
    """Rejection-sample seeds until SB hole matches predicate."""
    for s in range(max_tries):
        env.reset(s)
        if predicate(env.state().hole[0]):
            return s
    raise RuntimeError(f"no seed in {max_tries} matched predicate")


def _legal_mask(obs) -> np.ndarray:
    m = np.zeros(int(pte.NUM_ACTIONS), dtype=np.float32)
    for a in obs.legal:
        m[int(a)] = 1.0
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path,
                   help="path to a cfr_iter_XXXX.ckpt or cfr_final.ckpt")
    p.add_argument("--device", default="cpu",
                   help="diagnostic is fast on CPU; no reason to load GPU")
    args = p.parse_args()

    sd = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = sd["cfg"]
    print(f"[diag] loaded {args.ckpt}  iteration={sd['iteration']}  "
          f"cfg.model={cfg.model}")

    adv_sb = AdvNet(cfg.model).to(args.device)
    adv_sb.load_state_dict(sd["adv_net_sb"])
    adv_sb.train(False)

    env = pte.Env(0xD1A6)

    # Compute sigma at SB-open for each hand spec.
    rows = []
    for spec in HAND_SPECS:
        pred = _make_predicate(spec)
        seed = _find_seed(env, pred)
        env.reset(seed)
        assert int(env.to_act()) == 0, "SB must be to-act at state 0"
        obs = env.observation()
        mask = _legal_mask(obs)
        # Tokenize from the SB perspective; pad to a batch of 1 for the
        # transformer forward, then read the regret head's output.
        tok_state = tokenize_state(env.state(), hero_seat=0)
        tokens_np, pad_mask_np, dpos_np = pad_batch([tok_state])
        with torch.no_grad():
            tokens   = torch.from_numpy(tokens_np).to(args.device)
            pad_mask = torch.from_numpy(pad_mask_np).to(args.device)
            dpos     = torch.from_numpy(dpos_np).to(args.device)
            pred_r = adv_sb(tokens, pad_mask, dpos).squeeze(0).cpu().numpy()
        sigma = regret_matching_np(pred_r, mask)
        rows.append((spec, sigma, pred_r))

    # Print per-hand sigma table.
    labels = ("F", "C", "R25", "R33", "R50", "R75",
              "R100", "R150", "R200", "R300", "AI")
    header = "hand   " + " ".join(f"{lab:>5}" for lab in labels)
    print()
    print("=== per-hand sigma at SB-open ===")
    print(header)
    for spec, sigma, _ in rows:
        cells = " ".join(f"{p:5.2f}" for p in sigma)
        print(f"{spec:<6} {cells}")

    # Per-slot range across hands: how much does sigma move?
    sigmas = np.stack([r[1] for r in rows], axis=0)  # (n_hands, NUM_ACTIONS)
    per_slot_min = sigmas.min(axis=0)
    per_slot_max = sigmas.max(axis=0)
    per_slot_range = per_slot_max - per_slot_min

    print()
    print("=== sigma spread across hands (max - min) ===")
    print(header)
    cells = " ".join(f"{r:5.2f}" for r in per_slot_range)
    print(f"range  {cells}")

    print()
    overall_max_range = float(per_slot_range.max())
    print(f"[diag] overall max sigma-range across slots = {overall_max_range:.3f}")

    # Also check: pure regret-output range (before regret-matching) tells
    # whether the network is producing different INTERNAL representations
    # even when regret-matching collapses small differences to similar sigmas.
    preds = np.stack([r[2] for r in rows], axis=0)
    pred_max_range = float((preds.max(axis=0) - preds.min(axis=0)).max())
    print(f"[diag] overall max raw-regret range across slots = {pred_max_range:.3f}")

    print()
    if overall_max_range > 0.30:
        print("  >>> sigma varies strongly across hands; net IS conditioning. "
              "Convergence likely just slow.")
    elif overall_max_range > 0.10:
        print("  >>> sigma varies moderately. Some conditioning present but weak.")
    elif pred_max_range > 0.5:
        print("  >>> raw regret varies across hands, but regret-matching is "
              "collapsing the differences. The signal is in the trunk but "
              "small relative to the dominant slot. Should sharpen with "
              "training.")
    else:
        print("  >>> sigma AND raw regrets nearly flat across hands. The "
              "AdvNet is essentially ignoring the hole-card region of x. "
              "More training won't fix this; needs an encoder/trunk change.")


if __name__ == "__main__":
    main()
