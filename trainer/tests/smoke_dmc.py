"""End-to-end smoke test for the DMC loop (flat-sequence encoding, v0.2).

Runs a tiny rollout → buffer → gradient-step cycle and checks that:
  * transitions were produced and stored,
  * gradient steps ran without NaN,
  * model outputs stay finite,
  * loss decreases (or at least does not diverge) on an overfit-a-handful pass.

Invoke:
    PYTHONPATH=/home/elucterio/Poker/PokerTrainer/engine/build:/home/elucterio/Poker/PokerTrainer/trainer \\
        python /home/elucterio/Poker/PokerTrainer/trainer/tests/smoke_dmc.py
"""
from __future__ import annotations

import math

import numpy as np
import torch

import pokertrainer_engine as pte

from dmc.actor import rollout
from dmc.buffers import MCBuffer
from dmc.config import DMCConfig
from dmc.learner import build_optimizer, learn_step
from dmc.models import DMCNet, count_parameters


def main() -> None:
    cfg = DMCConfig()
    cfg.run.device = "cpu"
    cfg.learner.batch_size = 64

    # Cross-check shape constants between Python config and compiled engine.
    assert cfg.model.x_dim == pte.X_DIM
    assert cfg.model.a_dim == pte.A_DIM

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    device = torch.device("cpu")
    net = DMCNet(cfg.model).to(device)
    optim = build_optimizer(net, cfg.optim)
    env = pte.Env(cfg.actor.base_seed)
    buf = MCBuffer(
        capacity=4_000,
        x_dim=cfg.model.x_dim,
        a_dim=cfg.model.a_dim,
        rng=np.random.default_rng(1),
    )

    print(f"smoke_dmc: params={count_parameters(net):,}  "
          f"x_dim={pte.X_DIM} a_dim={pte.A_DIM} "
          f"hist={pte.HIST_MAX}x{pte.HIST_FEAT}")

    # --- phase 1: rollout some hands ---
    stats = rollout(env, net, buf,
                    n_hands=32,
                    epsilon=0.5,       # lots of exploration to populate buffer fast
                    device=device,
                    rng=rng)
    print(f"rollout: hands={stats['hands']} transitions={stats['transitions']} "
          f"sb_bb_sum={stats['sb_bb_sum']:+.2f}")
    assert stats["transitions"] > 0, "actor produced zero transitions"
    assert len(buf) == stats["transitions"], (len(buf), stats["transitions"])

    assert np.all(np.isfinite(buf.r[:len(buf)])), "non-finite reward in buffer"
    assert float(np.max(np.abs(buf.r[:len(buf)]))) <= 200.0, \
        f"reward magnitude too big: {np.max(np.abs(buf.r[:len(buf)]))}"

    # Sanity check (v0.3): is_real bit at offset 0 of each history row tells
    # us how many rows are populated. Each transition happens at least one
    # action into the hand for the non-first-actor, so most sampled x's have
    # at least one populated row; the bound is HIST_MAX.
    is_real_col_offsets = pte.X_OFF_HIST + np.arange(pte.HIST_MAX) * pte.HIST_FEAT
    any_valid = buf.x[:len(buf), is_real_col_offsets].sum(axis=1)
    assert (any_valid >= 0).all() and any_valid.max() <= pte.HIST_MAX

    # --- phase 2: a handful of gradient steps ---
    losses = []
    for i in range(30):
        batch = buf.sample(cfg.learner.batch_size)
        s = learn_step(net, optim, batch, device, cfg.optim.grad_clip)
        losses.append(s["loss"])
        assert math.isfinite(s["loss"]),     f"loss not finite at step {i}: {s}"
        assert math.isfinite(s["grad_norm"]), f"grad not finite at step {i}: {s}"
        if i % 10 == 0:
            print(f"step={i:3d}  loss={s['loss']:.4f}  "
                  f"grad={s['grad_norm']:.2f}  "
                  f"mean_pred={s['mean_pred']:+.3f}  "
                  f"mean_tgt={s['mean_target']:+.3f}")

    # It's a tiny random network on a tiny buffer — we don't claim convergence,
    # just that the learner doesn't diverge.
    first, last = losses[0], losses[-1]
    print(f"loss first={first:.4f} last={last:.4f}")
    assert last < 10 * max(first, 1.0), \
        f"loss diverged: {first:.4f} -> {last:.4f}"

    print("OK — DMC smoke loop runs end-to-end.")


if __name__ == "__main__":
    main()
