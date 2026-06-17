"""Phase-1 validation of the tabular blueprint (no search).
Pluribus idea → the hard correctness gate: a tabular Linear-MCCFR blueprint of
the 10bb push/fold game must reproduce the known Nash equilibrium · doc §6, §10.

The push/fold game is the cleanest ground truth available: at short stacks the
preflop decision collapses to jam-or-fold (SB) and call-or-fold (BB), and
``evaluate.pushfold_solver`` already solves its Nash equilibrium exactly (via
fictitious play on the 169-class equity/conflict matrices). If our
external-sampling MCCFR over the *abstracted* game (preset "pushfold", 169
lossless preflop classes) converges to the same SB-jam / BB-call ranges, the
whole blueprint stack — abstraction keying, regret matching, Linear-CFR
weighting, external sampling — is correct.

Run:
    PYTHONPATH=engine/build:trainer python -m pluribus.validate
    PYTHONPATH=engine/build:trainer python -m pluribus.validate --iters 400000
"""
from __future__ import annotations

import argparse

import numpy as np

from evaluate.preflop_equity import N_CLASSES, COMBOS, CLASS_LABELS
from evaluate.pushfold_solver import build_oracle, format_grid, _to_grid
from .config import FOLD, CHECK_CALL, ALL_IN, PluribusConfig
from .blueprint import train_blueprint


def _extract_pushfold(tables) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pull per-class SB-jam and BB-call frequencies out of the trained tables.

    SB open infoset:  (street=0, to_act=0, bucket=class, hist=())          → P(ALL_IN)
    BB vs a jam:       (street=0, to_act=1, bucket=class, hist=(ALL_IN,))   → P(CALL)
    """
    sb_jam = np.full(N_CLASSES, np.nan)
    bb_call = np.full(N_CLASSES, np.nan)
    diag = {"sb_keys": 0, "bb_keys": 0}
    for key, s in tables.strat.items():
        street, to_act, bucket, hist = key
        if street != 0:
            continue
        if to_act == 0 and len(hist) == 0:
            tot = s[FOLD] + s[ALL_IN]
            if tot > 0:
                sb_jam[bucket] = s[ALL_IN] / tot
            diag["sb_keys"] += 1
        elif to_act == 1 and hist == (ALL_IN,):
            tot = s[FOLD] + s[CHECK_CALL]
            if tot > 0:
                bb_call[bucket] = s[CHECK_CALL] / tot
            diag["bb_keys"] += 1
    # Unseen classes (never sampled) default to fold (0.0), as in the oracle's
    # tail; this only affects the rarest classes and a handful of combos.
    sb_jam = np.nan_to_num(sb_jam, nan=0.0)
    bb_call = np.nan_to_num(bb_call, nan=0.0)
    return sb_jam, bb_call, diag


def _metrics(bp: np.ndarray, oracle: np.ndarray) -> dict:
    """Compare a blueprint frequency vector to the oracle's, combo-weighted.

    Reports several views, because a *sampled* MCCFR blueprint vs an *exact*
    Nash oracle will always differ in the thin indifferent band (hands the
    equilibrium mixes), where exact frequencies have irreducible sampling
    variance — that is NOT a correctness failure. The substantive checks are:
      • gross   — combo-weight that is GROSSLY misplayed (|Δ| > 0.5, e.g. the
                  blueprint folds 100% a hand the oracle jams 100%). A correct
                  blueprint mass-misplays almost nothing; a broken one a lot.
      • size    — aggregate range size (combo-weighted %); must track the oracle.
      • mae     — combo-weighted mean abs error (info; dominated by the band).
    """
    w = COMBOS / COMBOS.sum()
    # "decisive" hands: ones the oracle plays (almost) purely (freq <0.1 or >0.9).
    # A correct blueprint must play THOSE on the right side; the mixed band
    # (0.1..0.9) is where a sampled method legitimately differs from an exact one.
    decisive = (oracle < 0.1) | (oracle > 0.9)
    wrong_side = decisive & ((bp >= 0.5) != (oracle >= 0.5))
    wd = (w * decisive).sum()
    return {
        "mae": float((w * np.abs(bp - oracle)).sum()),
        "gross": float((w * (np.abs(bp - oracle) > 0.5)).sum()),
        "decisive_misplay": float((w * wrong_side).sum() / wd) if wd > 0 else 0.0,
        "size": float((w * bp).sum()) * 100.0,
    }


def validate_pushfold(iters: int = 400_000, stack_bb: int = 10, seed: int = 0,
                      show_grids: bool = True, max_misplay: float = 0.03,
                      max_size_gap: float = 4.0) -> bool:
    print(f"── {stack_bb}bb push/fold blueprint vs Nash oracle "
          f"(MCCFR iters={iters:,}) ──", flush=True)
    cfg = PluribusConfig(stack_bb=stack_bb, action_preset="pushfold",
                         iters=iters, seed=seed, log_every=max(1, iters // 10),
                         allin_equity_samples=32)
    tables, _ = train_blueprint(cfg, progress=True)
    bp_jam, bp_call, diag = _extract_pushfold(tables)
    print(f"  extracted {diag['sb_keys']} SB-open + {diag['bb_keys']} BB-vs-jam "
          f"infosets (expect ~{N_CLASSES} each)")

    oracle = build_oracle(stack_bb=stack_bb, progress=False)
    jam = _metrics(bp_jam, oracle.sb_jam)
    call = _metrics(bp_call, oracle.bb_call)
    print(f"  SB jam : blueprint {jam['size']:5.1f}%  vs oracle "
          f"{oracle.sb_jam_pct*100:5.1f}%  | decisive-misplay "
          f"{jam['decisive_misplay']*100:4.1f}%  MAE {jam['mae']*100:4.1f}%")
    print(f"  BB call: blueprint {call['size']:5.1f}%  vs oracle "
          f"{oracle.bb_call_pct*100:5.1f}%  | decisive-misplay "
          f"{call['decisive_misplay']*100:4.1f}%  MAE {call['mae']*100:4.1f}%")

    if show_grids:
        print()
        print(format_grid(_to_grid(bp_jam), "blueprint SB jam %"))
        print(format_grid(oracle.jam_grid(), "oracle    SB jam %"))
        print()
        print(format_grid(_to_grid(bp_call), "blueprint BB call %"))
        print(format_grid(oracle.call_grid(), "oracle    BB call %"))

    jam_gap = abs(jam["size"] - oracle.sb_jam_pct * 100)
    call_gap = abs(call["size"] - oracle.bb_call_pct * 100)
    ok = (jam["decisive_misplay"] <= max_misplay
          and call["decisive_misplay"] <= max_misplay
          and jam_gap <= max_size_gap and call_gap <= max_size_gap)
    print(f"\n  push/fold gate: {'PASS' if ok else 'FAIL'} "
          f"(decisive-hand misplay ≤ {max_misplay*100:.0f}% AND range-size gap "
          f"≤ {max_size_gap:.0f}%)")
    return ok


def validate_river_resolve(seeds=(7, 11, 23, 42), stack_bb: int = 20,
                           iters: int = 600, thresh: float = 0.05) -> bool:
    """Phase-2 gate: a river re-solve over the WHOLE 1326-hand range (Trick 2)
    must drive the subgame to ~0 exploitability — the search machinery is exact
    on the river (true terminals, no continuation leaves)."""
    import pokertrainer_engine as pte
    from rebel_py.subgame_solver import CfrSolver
    from rebel_py.hand_index import board_free_mask
    from .abstraction import build_abstraction
    from .search import build_engine_subgame, subgame_exploitability

    print(f"── river re-solve (whole-range Linear CFR, {stack_bb}bb) ──", flush=True)
    abst = build_abstraction(PluribusConfig(action_preset="discrete"))
    ok = True
    for seed in seeds:
        env = pte.Env(seed, stack_bb * 100)
        env.reset(seed)
        while not env.is_terminal() and int(env.state().street) < 3:
            legal = [int(a) for a in env.state().legal_actions()]
            env.step(legal.index(1))
        if env.is_terminal():
            continue
        board = [int(c) for c in env.state().board][:5]
        sg = build_engine_subgame(env, abst, depth_limit=1)
        free = board_free_mask(board).astype(float)
        ranges = (free / free.sum(), free / free.sum())
        sv = CfrSolver(sg, ranges, num_iters=iters, linear=True)
        sv.multistep()
        e = subgame_exploitability(sg, sv.average_strategy(), ranges)
        flag = "ok" if e < thresh else "HIGH"
        print(f"  seed {seed:>3}: {len(sg.nodes):>4} nodes | exploitability "
              f"{e:.4f} bb/hand  [{flag}]")
        ok = ok and e < thresh
    print(f"\n  river re-solve gate: {'PASS' if ok else 'FAIL'} "
          f"(exploitability < {thresh} bb/hand)")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description="Pluribus validation gates")
    p.add_argument("--phase", choices=["1", "2", "all"], default="all",
                   help="1=push/fold blueprint, 2=river re-solve, all=both")
    p.add_argument("--iters", type=int, default=400_000)
    p.add_argument("--stack-bb", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-grids", action="store_true")
    p.add_argument("--max-misplay", type=float, default=0.03,
                   help="max combo-weighted misplay among oracle-DECISIVE hands")
    p.add_argument("--max-size-gap", type=float, default=4.0,
                   help="max aggregate range-size gap vs the oracle (percentage pts)")
    args = p.parse_args()
    ok = True
    if args.phase in ("1", "all"):
        ok &= validate_pushfold(iters=args.iters, stack_bb=args.stack_bb,
                                seed=args.seed, show_grids=not args.no_grids,
                                max_misplay=args.max_misplay,
                                max_size_gap=args.max_size_gap)
    if args.phase in ("2", "all"):
        print()
        ok &= validate_river_resolve()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
