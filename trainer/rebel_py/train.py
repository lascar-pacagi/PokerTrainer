"""Single-process ReBeL training (Phase 2a).

The loop, per the paper:
  1. self-play: ``RlRunner`` plays games over depth-limited subgames (leaves
     valued by the *current* net), emitting ``(query, root-value)`` examples;
  2. learn: regress the net onto buffered targets (Huber loss);
  3. gate: every few epochs, measure the exploitability of the recursively-
     solved strategy (``recursive_strategy``) in the exact full tree. It must
     trend down toward the exact-solve level as the net learns.

This is the chance-free (river, artificial betting-depth limit) milestone — it
validates the value-net machinery before the turn→river chance node (Phase 2b).

Run:
    PYTHONPATH=engine/build:trainer python -m rebel_py.train
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .config import RebelConfig
from .public_tree import build_river_subgame
from .subgame_solver import CfrSolver
from .exploitability import exploitability_bb
from .hand_index import NUM_HANDS
from .pbs import query_dim
from .models import ValueNet, NetValueFn
from .replay import QueryValueBuffer
from .recursive import RlRunner, recursive_strategy, SubgameParams, uniform_beliefs


def build_trees(cfg: RebelConfig):
    return [build_river_subgame(seed=s, starting_stack_bb=cfg.stack_bb,
                                allowed_actions=cfg.actions) for s in cfg.seeds]


def evaluate_exploitability(trees, value_fn, depth_limit, iters) -> float:
    params = SubgameParams(depth_limit=depth_limit, num_iters=iters,
                           random_action_prob=0.0)
    es = []
    for full in trees:
        strat = recursive_strategy(full, value_fn, params)
        es.append(exploitability_bb(full, strat, uniform_beliefs(full.board)))
    return float(np.mean(es))


def full_solve_baseline(trees, iters=800) -> float:
    es = []
    for full in trees:
        b = uniform_beliefs(full.board)
        sv = CfrSolver(full, (b[0].copy(), b[1].copy()), num_iters=iters, linear=True)
        sv.multistep()
        es.append(exploitability_bb(full, sv.average_strategy(), b))
    return float(np.mean(es))


def train(cfg: RebelConfig):
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    # Self-play queries the net at batch≈1 (one subgame's net leaves at a time).
    # Torch's default intra-op thread pool makes such tiny ops *slower* by orders
    # of magnitude (thread-launch overhead per matmul), so pin to one thread on
    # CPU. The GPU learner (Phase 3) overrides this.
    if cfg.device == "cpu":
        torch.set_num_threads(1)
    device = torch.device(cfg.device)

    trees = build_trees(cfg)
    qdim = query_dim()
    net = ValueNet(qdim, n_hidden=cfg.n_hidden, n_layers=cfg.n_layers,
                   use_layer_norm=cfg.use_layer_norm).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    lossf = torch.nn.SmoothL1Loss()
    buf = QueryValueBuffer(cfg.buffer_cap, qdim, NUM_HANDS, seed=cfg.seed)
    vfn = NetValueFn(net, device)
    params = SubgameParams(depth_limit=cfg.depth_limit, num_iters=cfg.cfr_iters,
                           random_action_prob=cfg.random_action_prob)

    base = full_solve_baseline(trees)
    e0 = evaluate_exploitability(trees, vfn, cfg.depth_limit, cfg.eval_iters)
    print(f"# boards={len(trees)} stack={cfg.stack_bb}bb depth_limit={cfg.depth_limit} "
          f"query_dim={qdim}")
    print(f"# exact full-solve exploitability = {base:.5f} bb/hand (the target floor)")
    print(f"epoch  0  buf={len(buf):>7d}  exploit={e0:.5f} bb/hand  (untrained net)")

    last_loss = float("nan")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        net.eval()
        for _ in range(cfg.games_per_epoch):
            full = trees[int(rng.integers(len(trees)))]
            RlRunner(full, vfn, params, buf, rng).step()

        net.train()
        for _ in range(cfg.batches_per_epoch):
            q, v = buf.sample(cfg.batch_size)
            qt = torch.as_tensor(q, device=device)
            vt = torch.as_tensor(v, device=device)
            pred = net(qt)
            loss = lossf(pred, vt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            e = evaluate_exploitability(trees, vfn, cfg.depth_limit, cfg.eval_iters)
            dt = time.time() - t0
            print(f"epoch {epoch:>2d}  buf={len(buf):>7d}  loss={last_loss:.4f}  "
                  f"exploit={e:.5f} bb/hand  ({dt:.1f}s)")

    # Final high-resolution eval: the per-epoch gate runs few CFR iters (cheap),
    # which has its own convergence floor; resolve the final strategy harder to
    # show the deployable quality separate from that floor.
    hi = max(cfg.eval_iters * 4, 1024)
    e_hi = evaluate_exploitability(trees, vfn, cfg.depth_limit, hi)
    print(f"# final exploitability @ {hi} CFR iters = {e_hi:.5f} bb/hand "
          f"(floor {base:.5f})")

    if cfg.ckpt_path:
        save_checkpoint(net, cfg, cfg.ckpt_path)
        print(f"# checkpoint written to {cfg.ckpt_path}")
    return net


def save_checkpoint(net: ValueNet, cfg: RebelConfig, path: str) -> None:
    """state_dict + config + a TorchScript trace (for a future C++/FFI consumer,
    mirroring the CFR trainer's export discipline)."""
    import os
    from dataclasses import asdict
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "config": asdict(cfg),
                "query_dim": net.query_dim}, path)
    net.eval()
    example = torch.zeros(1, net.query_dim)
    ts = torch.jit.trace(net.cpu(), example)
    ts.save(path + ".ts")


if __name__ == "__main__":
    train(RebelConfig())
