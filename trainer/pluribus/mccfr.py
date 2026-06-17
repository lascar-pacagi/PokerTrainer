"""External-sampling Linear MCCFR — one blueprint traversal.
Pluribus idea → "Linear MCCFR (external-sampling traversals, linear t-weighting)"
= Listing 1 (CFR) run with sampling, tables instead of nets · doc §2, §5.

This is the same external-sampling scheme the repo's Deep CFR traversal uses
(trainer/cfr/cfr_coro.traverse_coro), with two differences: (1) regrets and the
average strategy live in *tables* (infoset.py) keyed by the *abstracted* infoset
(abstraction.py), not in a neural net; (2) there is no GPU batching — a traversal
is pure CPU recursion over the engine.

The scheme, for a traverser p and Linear-CFR weight w = t:
  • terminal node      → return p's payoff (bb).
  • TRAVERSER's node    → branch every abstract-legal action, value each child,
                          v = Σ σ(a)·v(a); accumulate regret += w·(v(a) − v).
  • OPPONENT's node     → accumulate average strategy += w·σ, then SAMPLE one
                          action from σ and recurse (external sampling: the
                          opponent's reach is realized by the sampling, the
                          chance deal by reset()).
σ is stored ONLY at opponent nodes (Brown 2019, Alg. 1): the opponent samples,
so its nodes are visited ∝ its own reach — exactly the weighting the average
strategy needs. Over a full iteration both traverser=SB and traverser=BB run, so
every infoset is filled (regret when its owner traverses, strategy when the
other does).
"""
from __future__ import annotations

import numpy as np

from .config import NUM_ACTIONS
from .infoset import infoset_key

CHIPS_PER_BB = 100
_N_VISIBLE = {0: 0, 1: 3, 2: 4, 3: 5, 4: 5}   # board cards visible per street
_EVALUATOR = None


def _evaluator():
    global _EVALUATOR
    if _EVALUATOR is None:
        import pokertrainer_engine as pte
        _EVALUATOR = pte.HandEvaluator.load_or_generate("")
    return _EVALUATOR


def _allin_equity_payoff(state, traverser: int, rng: np.random.Generator,
                         k: int) -> float:
    """Traverser's bb payoff at an all-in showdown via Monte-Carlo equity.

    The committed board is whatever was visible when the last chips went in; the
    remaining cards are dealt `k` times and the hands compared, so the value is
    the (variance-reduced) equity-weighted pot share rather than one coin-flip
    board. Independent of the push/fold oracle (uses the engine's evaluator)."""
    h = state.hole
    sb = (int(h[0][0]), int(h[0][1]))
    bb = (int(h[1][0]), int(h[1][1]))
    committed = _N_VISIBLE[int(state.history[-1].street)] if state.history_size else 0
    board0 = [int(c) for c in state.board][:committed]
    used = {*sb, *bb, *board0}
    deck = [c for c in range(52) if c not in used]
    need = 5 - committed
    inv_sb = int(state.invested_prior_streets[0] + state.invested_this_street[0])
    inv_bb = int(state.invested_prior_streets[1] + state.invested_this_street[1])
    pot = inv_sb + inv_bb
    ev = _evaluator()
    wins0 = 0.0
    for _ in range(k):
        run = ([deck[i] for i in rng.choice(len(deck), size=need, replace=False)]
               if need > 0 else [])
        full = board0 + run
        r_sb = ev.evaluate7([sb[0], sb[1], *full])
        r_bb = ev.evaluate7([bb[0], bb[1], *full])
        wins0 += 1.0 if r_sb < r_bb else (0.5 if r_sb == r_bb else 0.0)
    eq0 = wins0 / k
    net_chips = (eq0 * pot - inv_sb) if traverser == 0 else ((1 - eq0) * pot - inv_bb)
    return net_chips / CHIPS_PER_BB


def traverse(env, traverser: int, weight: float, tables, abstraction,
             rng: np.random.Generator,
             prune_threshold: float = -np.inf,
             allin_samples: int = 0) -> float:
    """One external-sampling MCCFR recursion; returns the traverser's value (bb).

    `env` is a live engine Env at a (possibly terminal) node. `weight` is the
    Linear-CFR iteration weight (t); pass 1.0 for vanilla CFR. `prune_threshold`
    (Pluribus regret pruning) skips branching the traverser's actions whose
    cumulative regret is below it — keep at -inf to disable (exact).
    `allin_samples` (>0) values all-in showdowns by Monte-Carlo equity over that
    many runouts instead of the single dealt board (variance reduction)."""
    import pokertrainer_engine as pte

    if env.is_terminal():
        if allin_samples > 0:
            state = env.state()
            if int(state.terminal) == 2 and state.all_in[0] and state.all_in[1]:
                return _allin_equity_payoff(state, traverser, rng, allin_samples)
        return float(env.payoffs_bb()[traverser])

    state = env.state()
    actor = int(env.to_act())
    # state.legal_actions() (not env.observation()) — observation() also builds
    # the 816-dim encoder tensor, which this tabular traversal never needs.
    legal = [int(a) for a in state.legal_actions()]
    abs_legal = abstraction.action.actions(state, legal)

    mask = np.zeros(NUM_ACTIONS, dtype=np.float64)
    for a in abs_legal:
        mask[a] = 1.0

    key = infoset_key(state, abstraction.info)
    regret = tables.regret_vec(key)
    sigma = tables.strategy(key, mask)

    if actor != traverser:
        # ── OPPONENT NODE: accumulate avg strategy (×weight), then sample ─────
        tables.add_strat(key, weight * sigma)
        probs = sigma[abs_legal]
        z = probs.sum()
        probs = probs / z if z > 0 else np.full(len(abs_legal), 1.0 / len(abs_legal))
        a = int(rng.choice(abs_legal, p=probs))
        env.step_action(pte.ActionType(a))
        return traverse(env, traverser, weight, tables, abstraction, rng,
                        prune_threshold, allin_samples)

    # ── TRAVERSER NODE: branch every abstract-legal action ────────────────────
    action_values = np.zeros(NUM_ACTIONS, dtype=np.float64)
    explored = np.zeros(NUM_ACTIONS, dtype=bool)
    for a in abs_legal:
        if regret[a] < prune_threshold:
            continue                      # regret pruning (skip a hopeless action)
        child = env.clone()
        child.step_action(pte.ActionType(a))
        action_values[a] = traverse(child, traverser, weight, tables,
                                    abstraction, rng, prune_threshold,
                                    allin_samples)
        explored[a] = True

    # σ over explored actions only (pruned actions carry no value this pass).
    use = mask * explored
    su = sigma * use
    zz = su.sum()
    if zz > 0:
        v_state = float((su * action_values).sum() / zz)
    else:
        v_state = float((sigma * action_values).sum())
    instant = (action_values - v_state) * use
    tables.add_regret(key, weight * instant)
    return v_state
