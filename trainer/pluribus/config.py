"""Pluribus configuration — the knobs of the offline blueprint and online search.
Pluribus idea → the dials behind Listing "Pluribus, schematically" · doc §2, §8.
"""
from __future__ import annotations

from dataclasses import dataclass


# Engine ActionType ints (engine/src/action.h). Re-exported for readability.
FOLD, CHECK_CALL = 0, 1
RAISE_25, RAISE_33, RAISE_50, RAISE_75 = 2, 3, 4, 5
RAISE_100, RAISE_150, RAISE_200, RAISE_300 = 6, 7, 8, 9
ALL_IN = 10
NUM_ACTIONS = 11


@dataclass
class PluribusConfig:
    # ─── game / abstraction ──────────────────────────────────────────────
    stack_bb: int = 10
    # Action abstraction preset:
    #   "pushfold"  — preflop jam/fold (SB) + call/fold (BB facing a jam); the
    #                 Phase-1 Nash-oracle gate, reproduces evaluate.pushfold_solver.
    #   "discrete"  — a per-street pot-fraction grid mapped onto the engine's 11
    #                 ActionType slots (preflop denser, postflop coarse).
    action_preset: str = "pushfold"
    # For "discrete": which engine ActionType ints are allowed per street.
    # FOLD/CHECK_CALL/ALL_IN are always force-included (the tree must be closeable).
    preflop_actions: tuple[int, ...] = (FOLD, CHECK_CALL, RAISE_75, RAISE_150, ALL_IN)
    postflop_actions: tuple[int, ...] = (FOLD, CHECK_CALL, RAISE_50, RAISE_100, ALL_IN)
    # Information abstraction: postflop equity-histogram k-means bucket counts.
    # Preflop is lossless (169 strategic classes), so it is not bucketed.
    flop_buckets: int = 200
    turn_buckets: int = 200
    river_buckets: int = 200
    equity_hist_bins: int = 30        # histogram resolution for the k-means feature
    bucket_cache: str = "runs/pluribus_buckets"

    # ─── blueprint MCCFR ─────────────────────────────────────────────────
    # T iterations; each iteration runs one traverser=SB pass and one
    # traverser=BB pass (so both players' regret AND strategy tables fill).
    iters: int = 200_000
    traversals_per_iter: int = 1
    linear_cfr: bool = True           # Linear CFR: weight iteration t by t
    # All-in showdown variance reduction: 0 → use the single board the engine
    # dealt (pure external sampling); K>0 → Monte-Carlo all-in equity over K
    # sampled runouts (same equilibrium, far less variance — a called all-in
    # swings ±stack per board, the dominant noise source at short stacks).
    allin_equity_samples: int = 0
    # Regret-based pruning (Pluribus speedup): with prob `prune_prob`, after
    # `prune_after_iter`, skip branching actions whose regret < `prune_threshold`.
    # 0 / negative threshold disabled by default (exact, slower).
    prune_threshold: float = -3.0e6
    prune_prob: float = 0.0
    prune_after_iter: int = 0
    log_every: int = 10_000

    # ─── online re-solve (search) ────────────────────────────────────────
    depth_limit: int = 1              # betting rounds below the root → leaf
    search_cfr_iters: int = 256
    continuation_bias: float = 5.0    # multiply the biased action's prob ×this
    # The 4 continuation strategies of Trick 1 (blueprint + 3 biases).
    n_continuations: int = 4

    # ─── checkpointing ───────────────────────────────────────────────────
    ckpt_path: str = ""               # "" → no checkpoint written
    seed: int = 0

    def __post_init__(self):
        for fld in ("preflop_actions", "postflop_actions"):
            v = getattr(self, fld)
            if isinstance(v, list):
                setattr(self, fld, tuple(v))
