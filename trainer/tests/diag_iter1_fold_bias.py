"""Diagnostic: confirm that the iter-1 buffer asymmetry comes from
degenerate regret-matching sigma on the SB preflop-open state.

Hypothesis: a freshly-initialized AdvNet produces output where one slot
is randomly the highest. Regret matching of such an output is degenerate
(all probability on the single positive slot). If that slot happens to
be FOLD for SB at preflop-state-0 across most hole cards, then when
traverser=BB, the opponent SB folds at the root and the traversal
produces zero BB-side adv writes - explaining the 46x buffer imbalance
observed in the first cluster log.

Run:
    PYTHONPATH=engine/build:trainer python trainer/tests/diag_iter1_fold_bias.py
"""
from __future__ import annotations

import numpy as np
import torch

import pokertrainer_engine as pte

from cfr.config import CFRConfig
from cfr.models import AdvNet
from cfr.regret_matching import regret_matching_np


NUM_ACTIONS = int(pte.NUM_ACTIONS)


def _legal_mask(obs) -> np.ndarray:
    m = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for a in obs.legal:
        m[int(a)] = 1.0
    return m


def main() -> None:
    # Build TWO independently-initialized AdvNets - same shape the cluster
    # ran (hidden=1024, layers=10). For the diagnostic we don't need GPU.
    cfg = CFRConfig()
    cfg.model.hidden = 1024
    cfg.model.n_layers = 10
    torch.manual_seed(42)
    advs = [AdvNet(cfg.model), AdvNet(cfg.model)]
    for net in advs:
        net.train(False)

    # Sample N random deals, look only at the very first state (SB to act
    # preflop, blinds in). Run adv_net[0] (the SB net used during BB-
    # traverser traversals - that's what samples the SB action).
    N = 500
    fold_count = 0
    fold_prob_sum = 0.0
    argmax_hist = np.zeros(NUM_ACTIONS, dtype=np.int64)
    env = pte.Env(0xC0FFEE)

    for seed in range(N):
        env.reset(seed)
        # Sanity: SB should always be to-act at state 0.
        assert int(env.to_act()) == 0
        obs = env.observation()
        mask = _legal_mask(obs)
        with torch.no_grad():
            x = torch.from_numpy(obs.x).unsqueeze(0)
            r = advs[0](x).squeeze(0).numpy()           # (NUM_ACTIONS,)
        sigma = regret_matching_np(r, mask)
        # FOLD is slot 0.
        fold_prob_sum += float(sigma[0])
        if sigma[0] > 0.99:
            fold_count += 1
        argmax_hist[int(np.argmax(sigma))] += 1

    print(f"[diag] N={N} random preflop-state-0 SB decisions, fresh AdvNet[0]")
    print(f"[diag]  mean sigma(FOLD)                     = {fold_prob_sum / N:.4f}")
    print(f"[diag]  fraction with sigma(FOLD) > 0.99      = {fold_count / N:.4f}")
    print(f"[diag]  argmax-sigma histogram across slots:")
    labels = ("F", "C", "R25", "R33", "R50", "R75",
              "R100", "R150", "R200", "R300", "AI")
    for lab, c in zip(labels, argmax_hist):
        print(f"    {lab:<5} {c:>4} ({100*c/N:.1f}%)")

    print()
    print("[diag] Interpretation:")
    if fold_count / N > 0.5:
        print("  >>> Most random initializations land on FOLD for SB preflop-open.")
        print("  >>> This matches the 46x buffer asymmetry: when traverser=BB,")
        print("      opponent SB folds at state 0 -> 0 BB writes per such traversal.")
    elif fold_count / N > 0.1:
        print("  >>> Some random initializations land on FOLD, others elsewhere.")
        print("      Buffer asymmetry has a more complex cause; check non-FOLD")
        print("      degenerate behaviour too.")
    else:
        print("  >>> FOLD does not dominate. Asymmetry must come from elsewhere.")


if __name__ == "__main__":
    main()
