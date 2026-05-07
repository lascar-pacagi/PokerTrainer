"""Evaluate a Deep CFR checkpoint against baselines.

Usage:
    PYTHONPATH=engine/build:trainer python -m evaluate.cfr_ckpt \
        --ckpt runs/cfr_smoke/cfr_final.ckpt \
        --hands 1000 --opponents random,check_fold,calling_station

Loads both AdvNets and the PolicyNet, runs each through play_match against
the named opponents, and prints the standard mbb/100 + slot-summary report.

Two policy variants are evaluated by default:
    cfr_policy : the deployable average strategy (PolicyNet sampled stochastically)
    cfr_adv    : the per-iteration current strategy (AdvNets + regret matching)
The PolicyNet result is the one that matters for "is this a good GTO bot?"
since average strategy is what CFR converges to.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cfr.config import CFRConfig
from cfr.models import AdvNet, PolicyNet
from evaluate.match import play_match
from evaluate.policies import (
    CallingStationPolicy, CheckFoldPolicy, RandomPolicy,
    CFRAdvPolicy, CFRPolicyNetPolicy,
)


OPPONENT_FACTORIES = {
    "random":          RandomPolicy,
    "check_fold":      CheckFoldPolicy,
    "calling_station": CallingStationPolicy,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--hands", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0xCFEDED)
    p.add_argument("--opponents", type=str,
                   default="random,check_fold,calling_station")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-duplicated", action="store_true")
    p.add_argument("--variant", type=str, default="both",
                   choices=("both", "policy", "adv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: CFRConfig = sd["cfg"]
    print(f"loaded {args.ckpt}  iter={sd['iteration']}  cfg.model={cfg.model}")

    adv_sb = AdvNet(cfg.model); adv_sb.load_state_dict(sd["adv_net_sb"])
    adv_bb = AdvNet(cfg.model); adv_bb.load_state_dict(sd["adv_net_bb"])
    policy_net = PolicyNet(cfg.model); policy_net.load_state_dict(sd["policy_net"])
    adv_sb.to(device); adv_bb.to(device); policy_net.to(device)

    opp_names = [s.strip() for s in args.opponents.split(",") if s.strip()]
    bad = [n for n in opp_names if n not in OPPONENT_FACTORIES]
    if bad:
        raise SystemExit(f"unknown opponent(s): {bad}")

    duplicated = not args.no_duplicated

    variants_to_run = []
    if args.variant in ("both", "policy"):
        variants_to_run.append(("cfr_policy",
                                CFRPolicyNetPolicy(policy_net, device, stochastic=True)))
    if args.variant in ("both", "adv"):
        variants_to_run.append(("cfr_adv",
                                CFRAdvPolicy([adv_sb, adv_bb], device, stochastic=True)))

    for label, model_pol in variants_to_run:
        print(f"\n────────  {label}  ────────")
        for i, name in enumerate(opp_names):
            opp = OPPONENT_FACTORIES[name]()
            opp.name = name
            seed = (args.seed + 1_000_003 * i) & 0xFFFFFFFFFFFFFFFF
            r = play_match(model_pol, opp, n_hands=args.hands,
                           base_seed=seed, duplicated=duplicated)
            mode = "dup" if r.duplicated else "seq"
            print(f"vs {name:<16} {r.a_mbb_per_100:+9.0f} mbb/100 "
                  f"(SE {r.a_std_err_mbb_per_100:.0f})  "
                  f"{r.n_hands} hands [{mode}] in {r.elapsed_s:.1f}s")
            print(r.slot_summary(per_street=True))


if __name__ == "__main__":
    main()
