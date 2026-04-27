"""Play a trained model (or a fresh net) vs a uniform-random policy.

Usage:
    # Evaluate a checkpoint.
    PYTHONPATH=engine/build:trainer python -m evaluate.vs_random \\
        --ckpt runs/latest/weights_final_00020.ckpt --n-hands 2000

    # No checkpoint → fresh random net (sanity check: ~0 mbb/100 expected).
    PYTHONPATH=engine/build:trainer python -m evaluate.vs_random --n-hands 2000

Exit code is 0 regardless of outcome; this is a measurement tool. A trained
net should post a positive mbb/100 well clear of its ~standard error. Against
a uniform-random opponent, even a very weak learned policy typically clears
+10_000 mbb/100, because a random player folds aces one time in ten and shoves
bottom pair six times in ten.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import pokertrainer_engine as pte  # noqa: F401  (ensures module loads before DMCNet)

from dmc.config import DMCConfig
from dmc.models import DMCNet

from .match import play_match
from .policies import ModelPolicy, RandomPolicy


def load_model(ckpt_path: Path | None, device: torch.device) -> DMCNet:
    cfg = DMCConfig()
    net = DMCNet(cfg.model).to(device)
    if ckpt_path is not None:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(blob["model"])
        print(f"[vs_random] loaded ckpt: {ckpt_path}  step={blob.get('step', '?')}")
    else:
        print("[vs_random] no ckpt — evaluating a freshly initialized network")
    return net


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None, help="path to a .ckpt file")
    p.add_argument("--n-hands", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-duplicated", action="store_true",
                   help="disable duplicate-dealing (use seat-alternation only)")
    args = p.parse_args()

    device = torch.device(args.device)
    net = load_model(Path(args.ckpt) if args.ckpt else None, device)

    model  = ModelPolicy(net, device, name="model")
    random = RandomPolicy()

    rng = np.random.default_rng(args.seed)
    result = play_match(model, random,
                        n_hands=args.n_hands,
                        base_seed=args.seed,
                        rng=rng,
                        duplicated=not args.no_duplicated)
    print(result.summary())
    print(result.slot_summary(per_street=True))
    sys.exit(0)


if __name__ == "__main__":
    main()
