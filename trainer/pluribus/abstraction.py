"""Action + information abstraction — the distinctive Pluribus offline machinery.
Pluribus idea → "information abstraction (k-means on equity distributions) +
action abstraction (a handful of bet sizes)" · doc §2 (Stage 1), §3.

Two independent abstractions shrink the game so a *table* can hold it:

  • ACTION abstraction (`ActionAbstraction`): at each node, restrict the engine's
    11-action set to a small, context-dependent grid. Betting is hole-independent,
    so this is a function of the public state only.

  • INFORMATION abstraction (`InfoAbstraction.bucket`): map a (street, board,
    hole) situation to a small bucket id. Preflop is LOSSLESS — the 169 strategic
    classes (``evaluate.preflop_equity.class_of``). Postflop, situations are
    clustered by the *shape* of their equity distribution over runouts (k-means),
    so "top pair" and "flush draw" of equal raw equity still separate. Centroids
    are learned on a sample of boards (``learn_street_buckets``) and every
    situation is assigned to its nearest centroid (a standard scalable
    approximation of the full per-situation precompute).

The infoset that the MCCFR tables are keyed on is then
``(street, to_act, bucket, betting-history-token)`` — see infoset.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .config import FOLD, CHECK_CALL, ALL_IN, PluribusConfig

# Lossless preflop classes (169) come straight from the push/fold oracle's
# machinery so the blueprint and the oracle share one indexing.
from evaluate.preflop_equity import class_of

N_PREFLOP_CLASSES = 169
# Street ints (engine Street enum): PREFLOP=0 FLOP=1 TURN=2 RIVER=3.
PREFLOP, FLOP, TURN, RIVER = 0, 1, 2, 3
_N_VISIBLE = {PREFLOP: 0, FLOP: 3, TURN: 4, RIVER: 5}


# ════════════════════════════════════════════════════════════════════════════
# ACTION ABSTRACTION
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Abstraction:
    """The two abstractions bundled, as the MCCFR loop and search consume them."""
    action: "ActionAbstraction"
    info: "InfoAbstraction"


def build_abstraction(cfg: PluribusConfig,
                      bucket_cache: str | None = None) -> Abstraction:
    """Action abstraction from `cfg`; info abstraction from a learned bucket
    cache if given/exists (postflop k-means centroids), else preflop-only."""
    info = InfoAbstraction()
    cache = bucket_cache or cfg.bucket_cache
    if cache and os.path.exists(cache):
        info = InfoAbstraction.load(cache)
    return Abstraction(action=ActionAbstraction(cfg), info=info)


class ActionAbstraction:
    """Restrict the engine's legal actions to the abstraction's grid.

    ``actions(state, legal_ints)`` returns the allowed engine ActionType ints
    (a subset of the engine-legal set). The result is never empty: CHECK_CALL is
    always legal in HU NLHE, and the discrete grid force-includes FOLD/CALL/ALL_IN
    so the betting tree always closes.
    """

    def __init__(self, cfg: PluribusConfig):
        self.preset = cfg.action_preset
        # Always allow the three "structural" actions so a tree can terminate.
        force = (FOLD, CHECK_CALL, ALL_IN)
        self.preflop = tuple(sorted(set(cfg.preflop_actions) | set(force)))
        self.postflop = tuple(sorted(set(cfg.postflop_actions) | set(force)))

    def actions(self, state, legal_ints: list[int]) -> list[int]:
        if self.preset == "pushfold":
            # Preflop jam/fold + call/fold: forbid limping (CHECK_CALL on the
            # open) so the abstract game IS the push/fold game the Nash oracle
            # solves. When facing an all-in there is no raise, so allow call/fold.
            opp = 1 - int(state.to_act)
            facing_allin = bool(state.all_in[opp])
            allow = (FOLD, CHECK_CALL) if facing_allin else (FOLD, ALL_IN)
            out = [a for a in legal_ints if a in allow]
            return out or [a for a in legal_ints if a == CHECK_CALL]
        # "discrete" pot-fraction grid, per street.
        allow = self.preflop if int(state.street) == PREFLOP else self.postflop
        out = [a for a in legal_ints if a in allow]
        return out or [a for a in legal_ints if a == CHECK_CALL] or list(legal_ints)


# ════════════════════════════════════════════════════════════════════════════
# INFORMATION ABSTRACTION
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class StreetBuckets:
    """Learned k-means centroids for one postflop street (feature = equity hist)."""
    street: int
    centroids: np.ndarray            # (K, n_bins) float
    n_bins: int


class InfoAbstraction:
    """Map a live engine state to its information bucket.

    Preflop → 0..168 (lossless class). Postflop → 0..K-1 nearest-centroid bucket
    if centroids for that street are loaded, else a documented coarse FALLBACK:
    the preflop class of the hole cards (board-blind). The fallback keeps a full
    "discrete" game runnable before the (expensive) k-means precompute; the
    Phase-1 push/fold gate is preflop-only and never reaches it.
    """

    def __init__(self, street_buckets: dict[int, StreetBuckets] | None = None,
                 hist_samples: int = 400):
        self._buckets = street_buckets or {}
        self._hist_samples = hist_samples
        self._feat_cache: dict[tuple, int] = {}   # (street, board, hand) → bucket
        self._warned = False
        self._ev = None

    # ── public ──────────────────────────────────────────────────────────────
    def bucket(self, state) -> int:
        """Bucket of the player-to-act's hand in a live engine state."""
        street = int(state.street)
        to_act = int(state.to_act)
        hole = state.hole[to_act]
        board = [int(c) for c in state.board][:_N_VISIBLE[street]]
        return self.bucket_cards(street, board, int(hole[0]), int(hole[1]))

    def bucket_cards(self, street: int, board: list[int], c0: int, c1: int) -> int:
        """Bucket of a (street, board, hole) directly — no engine state needed
        (used by ranges.py to vectorize over all 1326 hands)."""
        if street == PREFLOP:
            return class_of(c0, c1)
        sb = self._buckets.get(street)
        if sb is None:
            if not self._warned:
                print("[pluribus.abstraction] WARNING: no postflop buckets loaded "
                      f"for street={street}; using coarse preflop-class fallback. "
                      "Run learn_street_buckets / load a bucket cache for a real "
                      "postflop abstraction.")
                self._warned = True
            return class_of(c0, c1)
        return self._assign(street, board, (c0, c1), sb)

    # ── nearest-centroid assignment (cached per situation) ───────────────────
    def _assign(self, street: int, board: list[int], hole: tuple[int, int],
                sb: StreetBuckets) -> int:
        key = (street, tuple(sorted(board)), tuple(sorted(hole)))
        hit = self._feat_cache.get(key)
        if hit is not None:
            return hit
        feat = equity_histogram(board, hole, sb.n_bins,
                                self._hist_samples, _rng(), self._evaluator())
        d = ((sb.centroids - feat[None, :]) ** 2).sum(axis=1)
        b = int(np.argmin(d))
        self._feat_cache[key] = b
        return b

    def _evaluator(self):
        if self._ev is None:
            import pokertrainer_engine as pte
            self._ev = pte.HandEvaluator.load_or_generate("")
        return self._ev

    # ── persistence ───────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        arrs: dict[str, np.ndarray] = {}
        for st, sb in self._buckets.items():
            arrs[f"centroids_{st}"] = sb.centroids
            arrs[f"nbins_{st}"] = np.array([sb.n_bins])
        np.savez(path, streets=np.array(sorted(self._buckets)), **arrs)

    @staticmethod
    def load(path: str, hist_samples: int = 400) -> "InfoAbstraction":
        sbs: dict[int, StreetBuckets] = {}
        with np.load(path) as z:
            for st in [int(s) for s in z["streets"]]:
                sbs[st] = StreetBuckets(st, z[f"centroids_{st}"],
                                        int(z[f"nbins_{st}"][0]))
        return InfoAbstraction(sbs, hist_samples=hist_samples)


# ════════════════════════════════════════════════════════════════════════════
# EQUITY-DISTRIBUTION FEATURE + K-MEANS (the postflop bucket learner)
# ════════════════════════════════════════════════════════════════════════════

_RNG = None


def _rng():
    global _RNG
    if _RNG is None:
        _RNG = np.random.default_rng(0xB10E)
    return _RNG


def equity_histogram(board: list[int], hole: tuple[int, int], n_bins: int,
                     n_samples: int, rng: np.random.Generator, ev) -> np.ndarray:
    """Feature vector: histogram of the hand's equity across sampled runouts.

    For each sampled completion of the board, estimate the hand's equity as the
    fraction of a sampled batch of opponent holdings it beats (ties = ½). The
    *shape* of this histogram is what distinguishes made hands (mass near 1)
    from draws (bimodal) — the property k-means clusters on. On the river there
    is no runout, so the histogram concentrates at the hand's single equity.
    """
    h0, h1 = hole
    used = set(board) | {h0, h1}
    deck = [c for c in range(52) if c not in used]
    need = 5 - len(board)                       # board cards still to come
    n_runouts = 1 if need == 0 else max(1, n_samples // 16)
    n_opp = max(8, n_samples // max(1, n_runouts))
    eqs = np.empty(n_runouts, dtype=np.float64)
    for r in range(n_runouts):
        d = deck
        if need > 0:
            run_idx = rng.choice(len(d), size=need, replace=False)
            run = [d[i] for i in run_idx]
            d = [c for c in d if c not in run]
        else:
            run = []
        full = board + run
        hero = ev.evaluate7([h0, h1, full[0], full[1], full[2], full[3], full[4]])
        wins = 0.0
        for _ in range(n_opp):
            oi = rng.choice(len(d), size=2, replace=False)
            o0, o1 = d[oi[0]], d[oi[1]]
            vill = ev.evaluate7([o0, o1, full[0], full[1], full[2], full[3], full[4]])
            wins += 1.0 if hero < vill else (0.5 if hero == vill else 0.0)
        eqs[r] = wins / n_opp
    hist, _ = np.histogram(eqs, bins=n_bins, range=(0.0, 1.0))
    s = hist.sum()
    return (hist / s).astype(np.float64) if s > 0 else hist.astype(np.float64)


def _kmeans(X: np.ndarray, k: int, iters: int = 25,
            rng: np.random.Generator | None = None) -> np.ndarray:
    """Plain Lloyd's k-means → (k, d) centroids. k-means++ seeding."""
    rng = rng or np.random.default_rng(0)
    n = X.shape[0]
    k = min(k, n)
    # k-means++ seeding.
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        centers.append(X[rng.choice(n, p=probs)])
    C = np.array(centers)
    for _ in range(iters):
        assign = np.argmin(((X[:, None, :] - C[None, :, :]) ** 2).sum(2), axis=1)
        newC = np.array([X[assign == j].mean(0) if np.any(assign == j) else C[j]
                         for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C


def learn_street_buckets(street: int, n_boards: int, k: int, *,
                         n_bins: int = 30, hist_samples: int = 400,
                         seed: int = 0) -> StreetBuckets:
    """Learn k centroids for `street` from a random sample of (board, hand)
    equity histograms. This is the offline information-abstraction precompute;
    cluster on a sample, then assign all situations at run time (InfoAbstraction).
    """
    import pokertrainer_engine as pte
    ev = pte.HandEvaluator.load_or_generate("")
    rng = np.random.default_rng(seed)
    nvis = _N_VISIBLE[street]
    feats = []
    for _ in range(n_boards):
        cards = rng.choice(52, size=nvis + 2, replace=False)
        board = [int(c) for c in cards[:nvis]]
        hole = (int(cards[nvis]), int(cards[nvis + 1]))
        feats.append(equity_histogram(board, hole, n_bins, hist_samples, rng, ev))
    X = np.array(feats)
    C = _kmeans(X, k, rng=rng)
    return StreetBuckets(street=street, centroids=C, n_bins=n_bins)
