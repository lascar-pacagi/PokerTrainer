"""Bayes the betting history into per-player ranges, assuming blueprint play.
Pluribus idea → "Problem 1: the position is not a state" → Bayes turns the line
into a range for every player · doc §2 (Stage 2, step 2), §7.

A poker subgame's root is not a single state: its value depends on what each
player might hold. If everyone is assumed to have played a known strategy so far
(the blueprint), then every public action is evidence, and Bayes' rule turns the
betting line into a probability distribution over the 1326 hole combos for each
player:

    range_p[h] ∝ Π_{p's actions a in the line}  σ_blueprint(a | infoset of p holding h)

Hands that the blueprint would (almost) never play this way get ~0 mass; hands
that fit the line get boosted. Card removal: only board-free combos carry mass.
The result feeds the search root beliefs (search.py). Pluribus knows its OWN
strategy exactly, so its own range is exact; opponents are assumed to play the
blueprint (the honest caveat is in the tutorial's Q2).
"""
from __future__ import annotations

import numpy as np

from rebel_py.hand_index import NUM_HANDS, HAND_CARDS, board_free_mask
from .config import NUM_ACTIONS
from .infoset import infoset_key  # noqa: F401  (kept for parity / external use)

_NO_CARD = 255
_N_VISIBLE = {0: 0, 1: 3, 2: 4, 3: 5, 4: 5}


def bayes_ranges(state, blueprint, *, replay_seed: int = 0xBA1E5
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Return (range_sb, range_bb): normalized beliefs over the 1326 combos at
    `state`, computed by replaying its public history under `blueprint`."""
    import pokertrainer_engine as pte

    abst = blueprint.abstraction
    hist = list(state.history)
    board_full = [int(c) for c in state.board if int(c) != _NO_CARD]
    free = board_free_mask(board_full).astype(np.float64)
    ranges = [free.copy(), free.copy()]

    # Replay a placeholder-hole env to recover, at each historical decision, the
    # hole-INDEPENDENT public facts: who acts, the street, and the abstract-legal
    # mask (needed to normalize the blueprint's stored strategy-sum there).
    stack = int(state.starting_stacks[0])
    env = pte.Env(replay_seed, stack)
    env.reset(replay_seed)
    prefix: list[int] = []
    for act in hist:
        st = env.state()
        if st.is_terminal():
            break
        p = int(st.to_act)
        street = int(st.street)
        a = int(act.type)
        legal = [int(x) for x in st.legal_actions()]
        abs_legal = abst.action.actions(st, legal)
        mask = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for x in abs_legal:
            mask[x] = 1.0
        board = board_full[:_N_VISIBLE[street]]
        key_prefix = tuple(prefix)

        # Per-hand bucket → per-bucket blueprint σ → multiply this action's prob.
        buckets = np.full(NUM_HANDS, -1, dtype=np.int64)
        for h in range(NUM_HANDS):
            if ranges[p][h] > 0.0:
                buckets[h] = abst.info.bucket_cards(
                    street, board, int(HAND_CARDS[h, 0]), int(HAND_CARDS[h, 1]))
        for b in np.unique(buckets):
            if b < 0:
                continue
            key = (street, p, int(b), key_prefix)
            sig = _avg_sigma(blueprint.strat.get(key), mask)
            ranges[p][buckets == b] *= sig[a]

        env.step_action(pte.ActionType(a))
        prefix.append(a)

    for p in (0, 1):
        s = ranges[p].sum()
        if s > 0:
            ranges[p] /= s
    return ranges[0], ranges[1]


def _avg_sigma(strat_sum: np.ndarray | None, mask: np.ndarray) -> np.ndarray:
    """Normalized average strategy from a stored strategy-sum (uniform fallback)."""
    if strat_sum is None:
        n = max(1.0, float(mask.sum()))
        return mask / n
    pos = strat_sum * mask
    z = pos.sum()
    if z > 0:
        return pos / z
    n = max(1.0, float(mask.sum()))
    return mask / n
