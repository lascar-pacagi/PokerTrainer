"""Unified high-N evaluation for a single checkpoint.

Existing `vs_random.py` / `vs_rulebased.py` each evaluate against ONE opponent.
For comparing checkpoints across all baselines at consistent settings — and
especially at high hand counts (5k–20k) where we actually trust the SE — it's
easier to drive everything from one tool.

Usage:
    PYTHONPATH=engine/build:trainer python -m evaluate.eval_ckpt \\
        --ckpt runs/gpu_100k_clip10/weights_final_00100000.ckpt \\
        --n-hands 5000

    PYTHONPATH=engine/build:trainer python -m evaluate.eval_ckpt \\
        --ckpt path/to.ckpt \\
        --opponents check_fold,calling_station --n-hands 20000

Output is a one-line-per-opponent rollup with mbb/100, SE, and σ-from-zero,
followed by the model's per-street slot summary on each match.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.serialization as _ts

import pokertrainer_engine as pte  # noqa: F401

from dmc.config import (
    ActorConfig, DMCConfig, LearnerConfig, ModelConfig, OptimConfig, RunConfig,
)
from dmc.models import DMCNet

from .match import play_match
from .policies import (
    CallingStationPolicy, CheckFoldPolicy, ModelPolicy, RandomPolicy,
)


_ts.add_safe_globals([
    DMCConfig, ModelConfig, OptimConfig,
    ActorConfig, LearnerConfig, RunConfig,
])


OPPONENTS = {
    "random":          RandomPolicy,
    "check_fold":      CheckFoldPolicy,
    "calling_station": CallingStationPolicy,
}


def load_model(ckpt_path: Path, device: torch.device) -> DMCNet:
    blob = torch.load(ckpt_path, map_location=device)
    cfg = blob.get("cfg")
    if cfg is not None and hasattr(cfg, "model"):
        mcfg = cfg.model
        if not isinstance(mcfg, ModelConfig):
            mcfg = ModelConfig(
                x_dim=int(mcfg.x_dim), a_dim=int(mcfg.a_dim),
                mlp_hidden=int(mcfg.mlp_hidden), mlp_layers=int(mcfg.mlp_layers),
                arch=getattr(mcfg, "arch", "mlp_v1"),
                mlp_expansion=int(getattr(mcfg, "mlp_expansion", 4)),
            )
    else:
        mcfg = DMCConfig().model
    net = DMCNet(mcfg).to(device)
    net.load_state_dict(blob["model"])
    net.train(False)
    print(f"[eval_ckpt] loaded {ckpt_path}  step={blob.get('step', '?')}  "
          f"x={mcfg.x_dim} a={mcfg.a_dim} hidden={mcfg.mlp_hidden}")
    return net


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--opponents", type=str,
                   default="random,check_fold,calling_station",
                   help="comma-separated subset of " + ",".join(OPPONENTS))
    p.add_argument("--n-hands", type=int, default=5000,
                   help="hands per match. 5k tightens vs-station SE to ~70k; "
                        "20k tightens it to ~35k.")
    p.add_argument("--seed", type=int, default=0xEEEE)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-duplicated", action="store_true")
    p.add_argument("--no-slot-summary", action="store_true",
                   help="skip the per-street slot breakdown printout")
    args = p.parse_args()

    requested = [s.strip() for s in args.opponents.split(",") if s.strip()]
    unknown = [n for n in requested if n not in OPPONENTS]
    if unknown:
        print(f"[eval_ckpt] unknown opponent(s): {unknown}; "
              f"known: {list(OPPONENTS)}")
        return 2

    device = torch.device(args.device)
    net = load_model(Path(args.ckpt), device)
    model = ModelPolicy(net, device, name="model")

    print(f"[eval_ckpt] running {args.n_hands} hands × {len(requested)} "
          f"opponent{'s' if len(requested) != 1 else ''} "
          f"(duplicated={'no' if args.no_duplicated else 'yes'})")
    print()

    # Header — column-aligned for readable diff between runs.
    print(f"  {'opponent':<18} {'mbb/100':>11} {'SE':>10} {'σ':>6}  "
          f"{'mode':>4} {'hands':>7} {'wall':>6}")
    print(f"  {'-'*18} {'-'*11} {'-'*10} {'-'*6}  {'-'*4} {'-'*7} {'-'*6}")

    results: dict[str, dict] = {}
    t_total = time.time()
    for i, name in enumerate(requested):
        opp = OPPONENTS[name]()
        per_match_seed = (args.seed + 1_000_003 * i) & 0xFFFFFFFFFFFFFFFF
        rng = np.random.default_rng(per_match_seed)
        r = play_match(model, opp,
                       n_hands=args.n_hands,
                       base_seed=per_match_seed,
                       rng=rng,
                       duplicated=not args.no_duplicated)
        sigma = (abs(r.a_mbb_per_100) /
                 max(r.a_std_err_mbb_per_100, 1.0))
        mode = "dup" if r.duplicated else "seq"
        print(f"  {name:<18} {r.a_mbb_per_100:>+11.0f} "
              f"{r.a_std_err_mbb_per_100:>10.0f} {sigma:>6.1f}  "
              f"{mode:>4} {r.n_hands:>7d} {r.elapsed_s:>5.1f}s")
        results[name] = {
            "mbb_per_100": r.a_mbb_per_100,
            "se": r.a_std_err_mbb_per_100,
            "n_hands": r.n_hands,
            "result": r,
        }

    if not args.no_slot_summary:
        print()
        for name in requested:
            r = results[name]["result"]
            print(f"  vs {name}:")
            print(r.slot_summary(per_street=True))

    print()
    print(f"[eval_ckpt] total wall: {time.time() - t_total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
