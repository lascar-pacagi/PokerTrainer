"""Play a trained model (or a fresh net) vs simple rule-based baselines.

Two opponents, chosen via --opponent:
  calling_station — never folds, never raises. A classic passive fish.
  check_fold      — folds to any bet, otherwise checks. Dead money.

Usage:
    PYTHONPATH=engine/build:trainer python -m evaluate.vs_rulebased \\
        --ckpt runs/latest/weights_final_00020.ckpt \\
        --opponent calling_station --n-hands 2000

Any competent policy should crush check_fold (+~50_000 mbb/100 just by
betting when checked to). calling_station is a slightly harder test — the
net has to learn to value-bet thin and stop bluffing, because the opponent
calls everything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import pokertrainer_engine as pte  # noqa: F401

from dmc.config import DMCConfig
from dmc.models import DMCNet

from .match import play_match
from .policies import CallingStationPolicy, CheckFoldPolicy, ModelPolicy


OPPONENTS = {
    "calling_station": CallingStationPolicy,
    "check_fold":      CheckFoldPolicy,
}


def load_model(ckpt_path: Path | None, device: torch.device) -> DMCNet:
    cfg = DMCConfig()
    net = DMCNet(cfg.model).to(device)
    if ckpt_path is not None:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(blob["model"])
        print(f"[vs_rulebased] loaded ckpt: {ckpt_path}  step={blob.get('step', '?')}")
    else:
        print("[vs_rulebased] no ckpt — evaluating a freshly initialized network")
    return net


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--opponent", type=str, default="calling_station",
                   choices=sorted(OPPONENTS.keys()))
    p.add_argument("--n-hands", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-duplicated", action="store_true",
                   help="disable duplicate-dealing (use seat-alternation only)")
    args = p.parse_args()

    device = torch.device(args.device)
    net = load_model(Path(args.ckpt) if args.ckpt else None, device)

    model = ModelPolicy(net, device, name="model")
    opp   = OPPONENTS[args.opponent]()

    rng = np.random.default_rng(args.seed)
    result = play_match(model, opp,
                        n_hands=args.n_hands,
                        base_seed=args.seed,
                        rng=rng,
                        duplicated=not args.no_duplicated)
    print(result.summary())
    print(result.slot_summary(per_street=True))
    sys.exit(0)


if __name__ == "__main__":
    main()
