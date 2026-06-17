"""The blueprint: offline tabular Linear-MCCFR + the queryable strategy artifact.
Pluribus idea → "Stage 1 (offline): the blueprint" — fill regret tables by
external-sampling Linear MCCFR, then read off the average strategy · doc §2, §6.

`train_blueprint(cfg)` runs the MCCFR loop (one SB-traverser pass + one
BB-traverser pass per iteration, Linear-CFR weight = t) and returns the filled
`Tables` + the `Abstraction`. `Blueprint` wraps the resulting average strategy
for play / validation, with save/load.
"""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np

from .abstraction import Abstraction, ActionAbstraction, InfoAbstraction, build_abstraction
from .config import NUM_ACTIONS, PluribusConfig
from .infoset import Tables, infoset_key
from . import mccfr

CHIPS_PER_BB = 100


# ════════════════════════════════════════════════════════════════════════════
# TRAINER
# ════════════════════════════════════════════════════════════════════════════

def train_blueprint(cfg: PluribusConfig, *, tables: Tables | None = None,
                    abstraction: Abstraction | None = None,
                    rng: np.random.Generator | None = None,
                    progress: bool = True) -> tuple[Tables, Abstraction]:
    """Fill regret/strategy tables by external-sampling Linear MCCFR."""
    import pokertrainer_engine as pte

    tables = tables or Tables(NUM_ACTIONS)
    abstraction = abstraction or build_abstraction(cfg)
    rng = rng or np.random.default_rng(cfg.seed)
    stack_chips = cfg.stack_bb * CHIPS_PER_BB
    env = pte.Env(cfg.seed, stack_chips)

    t0 = time.time()
    for t in range(1, cfg.iters + 1):
        weight = float(t) if cfg.linear_cfr else 1.0
        prune = (-np.inf if (cfg.prune_prob <= 0 or t < cfg.prune_after_iter
                             or rng.random() >= cfg.prune_prob)
                 else cfg.prune_threshold)
        for traverser in (0, 1):
            for _ in range(cfg.traversals_per_iter):
                env.reset(int(rng.integers(1, 2**63 - 1)))
                mccfr.traverse(env, traverser, weight, tables, abstraction, rng,
                               prune_threshold=prune,
                               allin_samples=cfg.allin_equity_samples)
        if progress and (t % cfg.log_every == 0 or t == cfg.iters):
            dt = time.time() - t0
            print(f"[pluribus.blueprint] iter {t:>9,}/{cfg.iters:,} | "
                  f"infosets={tables.n_infosets():>8,} | "
                  f"{t / max(dt, 1e-9):,.0f} it/s", flush=True)

    if cfg.ckpt_path:
        Blueprint.from_tables(tables, abstraction).save(cfg.ckpt_path)
        if progress:
            print(f"[pluribus.blueprint] saved → {cfg.ckpt_path}")
    return tables, abstraction


# ════════════════════════════════════════════════════════════════════════════
# QUERYABLE ARTIFACT
# ════════════════════════════════════════════════════════════════════════════

class Blueprint:
    """The trained average strategy, queryable by live engine state.

    Holds the (linearly weighted) strategy-sum dict + the abstraction; computes
    the normalized average strategy on demand, masked to the abstract-legal
    actions of the queried state. Unseen infosets fall back to uniform-legal."""

    def __init__(self, abstraction: Abstraction, strat: dict[tuple, np.ndarray],
                 n_actions: int = NUM_ACTIONS):
        self.abstraction = abstraction
        self.strat = strat
        self.n_actions = n_actions

    @staticmethod
    def from_tables(tables: Tables, abstraction: Abstraction) -> "Blueprint":
        return Blueprint(abstraction, tables.strat, tables.n_actions)

    # ── query ─────────────────────────────────────────────────────────────────
    def _mask(self, state) -> np.ndarray:
        legal = [int(a) for a in state.legal_actions()]
        abs_legal = self.abstraction.action.actions(state, legal)
        m = np.zeros(self.n_actions, dtype=np.float64)
        for a in abs_legal:
            m[a] = 1.0
        return m

    def strategy(self, state) -> np.ndarray:
        """Average strategy σ over NUM_ACTIONS for the player to act (0 in
        illegal/abstracted-out slots, sums to 1 over the abstract-legal set)."""
        mask = self._mask(state)
        key = infoset_key(state, self.abstraction.info)
        s = self.strat.get(key)
        if s is None:
            n = max(1.0, float(mask.sum()))
            return (mask / n)
        pos = s * mask
        z = pos.sum()
        if z > 0:
            return pos / z
        n = max(1.0, float(mask.sum()))
        return mask / n

    def sample(self, state, rng: np.random.Generator) -> int:
        """Sample an engine ActionType int from the blueprint at `state`."""
        sigma = self.strategy(state)
        idx = np.nonzero(sigma)[0]
        return int(rng.choice(idx, p=sigma[idx] / sigma[idx].sum()))

    # ── persistence ───────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        # The strategy-sum dict has tuple keys + ndarray values (not JSON-native),
        # so it is pickled. This is a TRUSTED, self-produced checkpoint (written
        # and read by this trainer only), the same trust model as the repo's other
        # .npz/checkpoint artifacts — do not unpickle a blueprint from an untrusted
        # source.
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path + ".strat.pkl", "wb") as f:
            pickle.dump(self.strat, f, protocol=pickle.HIGHEST_PROTOCOL)
        act = self.abstraction.action
        meta = {"preset": act.preset, "preflop": list(act.preflop),
                "postflop": list(act.postflop), "n_actions": self.n_actions}
        with open(path + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        if self.abstraction.info._buckets:
            self.abstraction.info.save(path + ".buckets.npz")

    @staticmethod
    def load(path: str) -> "Blueprint":
        with open(path + ".meta.json") as f:
            meta = json.load(f)
        with open(path + ".strat.pkl", "rb") as f:
            strat = pickle.load(f)
        cfg = PluribusConfig(action_preset=meta["preset"],
                             preflop_actions=tuple(meta["preflop"]),
                             postflop_actions=tuple(meta["postflop"]))
        action = ActionAbstraction(cfg)
        info = (InfoAbstraction.load(path + ".buckets.npz")
                if os.path.exists(path + ".buckets.npz") else InfoAbstraction())
        return Blueprint(Abstraction(action, info), strat, meta["n_actions"])
