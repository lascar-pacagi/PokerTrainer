"""Cluster blueprint trainer: CPU actor pool + additive table merge (NO GPU).
Pluribus idea → "8 days on a 64-core server, no GPU": tabular MCCFR is CPU/RAM
bound and parallelizes by summing per-worker regret/strategy increments · doc §11.

Synchronous data-parallel CFR. Each ROUND: ship the current master tables to W
workers; each worker deep-copies them, runs a batch of external-sampling Linear
MCCFR traversals (adapting locally), and returns only its INCREMENTS (Δregret,
Δstrat); the master sums all workers' increments. Because regret and
strategy-sum are additive, summing |W| workers' increments ≈ doing |W|× the
traversals of sequential CFR — exact in the limit, and embarrassingly parallel
across the 64 cores with no GPU. Linear-CFR weight = the round index.

Memory: tables are shipped per round, so this scales to "fits in a worker's
RAM". The 700 GB-class run shards the infoset keyspace across workers (each owns
a hash bucket; no per-round broadcast) — the documented extension; the additive
merge here is the primitive it builds on. The BLAS pin (package __init__) makes
one thread per worker × W workers the right shape for a 64-core box.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import numpy as np

from .abstraction import build_abstraction
from .config import NUM_ACTIONS, PluribusConfig
from .infoset import Tables
from . import mccfr


# ─── worker ──────────────────────────────────────────────────────────────────

_W: dict = {}   # per-worker persistent context (abstraction, env, cfg)


def _init_worker(cfg: PluribusConfig):
    import pokertrainer_engine as pte
    _W["cfg"] = cfg
    _W["abstraction"] = build_abstraction(cfg)
    _W["env"] = pte.Env(0, cfg.stack_bb * 100)


def _run_round(args):
    """One worker batch: start from the broadcast tables, run `n` traversals,
    return the INCREMENTS (Δregret, Δstrat) accumulated this round."""
    regret0, strat0, weight, n, seed = args
    cfg = _W["cfg"]
    abst = _W["abstraction"]
    env = _W["env"]
    rng = np.random.default_rng(seed)

    tables = Tables(NUM_ACTIONS)
    tables.regret = {k: v.copy() for k, v in regret0.items()}
    tables.strat = {k: v.copy() for k, v in strat0.items()}

    for _ in range(n):
        for traverser in (0, 1):
            env.reset(int(rng.integers(1, 2**63 - 1)))
            mccfr.traverse(env, traverser, weight, tables, abst, rng,
                           allin_samples=cfg.allin_equity_samples)

    d_regret = {k: v - regret0.get(k, 0.0) for k, v in tables.regret.items()}
    d_strat = {k: v - strat0.get(k, 0.0) for k, v in tables.strat.items()}
    return d_regret, d_strat


# ─── master ──────────────────────────────────────────────────────────────────

def train_blueprint_mp(cfg: PluribusConfig, *, n_workers: int, rounds: int,
                       traversals_per_worker: int, ckpt_dir: str = "",
                       ckpt_every: int = 50, log_every: int = 10,
                       start_method: str = "fork") -> Tables:
    """Run the parallel blueprint trainer; returns the master Tables."""
    master = Tables(NUM_ACTIONS)
    ctx = mp.get_context(start_method)
    t0 = time.time()
    with ctx.Pool(n_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for r in range(1, rounds + 1):
            weight = float(r) if cfg.linear_cfr else 1.0
            # Broadcast a snapshot; workers copy it and return increments.
            reg, strat = master.regret, master.strat
            seeds = [int(cfg.seed + r * 100003 + w) for w in range(n_workers)]
            jobs = [(reg, strat, weight, traversals_per_worker, s) for s in seeds]
            for d_regret, d_strat in pool.imap_unordered(_run_round, jobs):
                for k, v in d_regret.items():
                    cur = master.regret.get(k)
                    master.regret[k] = v.copy() if cur is None else cur + v
                for k, v in d_strat.items():
                    cur = master.strat.get(k)
                    master.strat[k] = v.copy() if cur is None else cur + v

            if r % log_every == 0 or r == rounds:
                dt = time.time() - t0
                trav = r * n_workers * traversals_per_worker * 2
                print(f"[pluribus.mp] round {r:>6,}/{rounds:,} | "
                      f"infosets={master.n_infosets():>9,} | "
                      f"{trav/max(dt,1e-9):,.0f} trav/s", flush=True)
            if ckpt_dir and (r % ckpt_every == 0 or r == rounds):
                _save(master, cfg, os.path.join(ckpt_dir, f"blueprint_r{r}"))
    return master


def _save(master: Tables, cfg: PluribusConfig, path: str) -> None:
    from .blueprint import Blueprint
    abst = build_abstraction(cfg)
    Blueprint(abst, master.strat).save(path)
    print(f"[pluribus.mp] checkpoint → {path}.*", flush=True)


# ─── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Pluribus parallel blueprint trainer")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    p.add_argument("--rounds", type=int, default=2000)
    p.add_argument("--traversals-per-worker", type=int, default=256)
    p.add_argument("--stack-bb", type=int, default=10)
    p.add_argument("--action-preset", default="pushfold")
    p.add_argument("--allin-equity-samples", type=int, default=8)
    p.add_argument("--bucket-cache", default="")
    p.add_argument("--ckpt-dir", default="")
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-method", default="fork", choices=["fork", "forkserver", "spawn"])
    args = p.parse_args()

    cfg = PluribusConfig(stack_bb=args.stack_bb, action_preset=args.action_preset,
                         allin_equity_samples=args.allin_equity_samples,
                         bucket_cache=args.bucket_cache or "runs/pluribus_buckets",
                         seed=args.seed)
    if args.ckpt_dir:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"[pluribus.mp] {args.workers} workers × {args.traversals_per_worker} "
          f"traversals × {args.rounds} rounds  (preset={args.action_preset}, "
          f"{args.stack_bb}bb)", flush=True)
    train_blueprint_mp(cfg, n_workers=args.workers, rounds=args.rounds,
                       traversals_per_worker=args.traversals_per_worker,
                       ckpt_dir=args.ckpt_dir, ckpt_every=args.ckpt_every,
                       log_every=args.log_every, start_method=args.start_method)


if __name__ == "__main__":
    main()
