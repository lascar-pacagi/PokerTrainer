"""Smoke test for the CFR learning probes (cfr/probe.py).

Why this exists:
    cfr_coro.py now calls `run_default_probes` every iteration. The probe
    code depends on a small but non-trivial surface: pybind's state.hole /
    invested_this_street, evaluate.policies.CFRAdvPolicy, and the
    AdvNet/regret-matching pipeline. A single import-or-API regression
    silently turns "useful diagnostic every iter" into "training crashes
    after the first refit on the cluster". We catch that here, at .sif
    build time.

What it checks (CPU-only, no CUDA required):
    1. cfr.probe + its dependencies import cleanly.
    2. A tiny AdvNet pair can be wrapped in CFRAdvPolicy and queried.
    3. `run_default_probes` runs both AA and 72o scenarios to completion
       with the requested n_hands (no rejection-sampler timeout).
    4. The 2-D preflop_slot_counts array has the expected shape and at
       least the `open` row is populated (since every hand starts with
       SB's open decision — and seat alternation guarantees the net is
       SB for ~half of them).

Invoke:
    PYTHONPATH=/opt/pokertrainer/engine/build:/opt/pokertrainer/trainer \\
        python3 /opt/pokertrainer/trainer/tests/smoke_cfr_probe.py
"""
from __future__ import annotations

import sys

import numpy as np
import torch

import pokertrainer_engine as pte  # noqa: F401 — fail fast if engine missing

from cfr.config import CFRConfig
from cfr.models import AdvNet
from cfr.probe import (
    PRE_SITUATIONS,
    is_pocket_aces,
    is_seven_two_offsuit,
    run_default_probes,
)
from evaluate.match import NUM_ACTIONS


def main() -> None:
    # Tiny CPU nets — we only care about API shape, not learning quality.
    cfg = CFRConfig()
    cfg.model.hidden = 32
    cfg.model.n_layers = 2
    device = torch.device("cpu")
    adv_nets = [AdvNet(cfg.model).to(device), AdvNet(cfg.model).to(device)]

    # Sanity check predicates on a few hand-rolled card pairs.
    # AsAd = (rank 12, suit 3), (rank 12, suit 1) → AA, suits differ.
    AsAd = [pte.make_card(12, 3), pte.make_card(12, 1)]
    assert is_pocket_aces(AsAd), "is_pocket_aces failed on AsAd"
    # 7h2c = (rank 5, suit 2), (rank 0, suit 0) → 72o.
    SevenTwoOff = [pte.make_card(5, 2), pte.make_card(0, 0)]
    assert is_seven_two_offsuit(SevenTwoOff), "is_seven_two_offsuit failed on 7h2c"
    # 7c2c = same suit ⇒ 72s, NOT what the probe wants.
    SevenTwoSuited = [pte.make_card(5, 0), pte.make_card(0, 0)]
    assert not is_seven_two_offsuit(SevenTwoSuited), \
        "is_seven_two_offsuit must reject 72s"
    # AsKs = top pair, not AA — must reject.
    AsKs = [pte.make_card(12, 3), pte.make_card(11, 3)]
    assert not is_pocket_aces(AsKs), "is_pocket_aces must reject AsKs"

    # End-to-end probe — 5 hands per scenario is enough to exercise every
    # code path. With AA at ~1/221 deals, this takes ≪ 1s of seed trial.
    results = run_default_probes(adv_nets, device, n_hands=5)
    assert len(results) == 2, f"expected 2 probes (AA, 72o), got {len(results)}"

    for r in results:
        assert r.n_hands == 5, \
            f"{r.label}: only played {r.n_hands}/5 hands — rejection sampler stalled?"
        assert np.isfinite(r.mbb_per_hand), \
            f"{r.label}: mbb_per_hand is not finite ({r.mbb_per_hand})"
        assert r.preflop_slot_counts.shape == (len(PRE_SITUATIONS), NUM_ACTIONS), \
            (f"{r.label}: bad shape {r.preflop_slot_counts.shape}, "
             f"expected ({len(PRE_SITUATIONS)}, {NUM_ACTIONS})")
        # `open` is bucket 0 (PRE_SITUATIONS[0] == "open"). Seat alternates,
        # so ~half of 5 hands → ~2-3 SB hands → at least one `open` decision.
        # Lower-bounding at 1 keeps the test deterministic regardless of
        # which seat starts (we only need to verify the code path fires).
        assert r.preflop_slot_counts[0].sum() >= 1, \
            (f"{r.label}: no `open` decisions recorded — bucketing or "
             f"play loop is broken")

    # Confirm the formatter doesn't blow up either.
    from cfr.probe import format_probe_line
    for r in results:
        out = format_probe_line(r, iter_t=0)
        assert "n=5" in out, f"format_probe_line output missing hand count: {out!r}"

    print("[smoke_cfr_probe] OK — probe imports, predicates, end-to-end run, "
          "and formatting all green")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"[smoke_cfr_probe] FAIL: {e}", file=sys.stderr)
        sys.exit(1)
