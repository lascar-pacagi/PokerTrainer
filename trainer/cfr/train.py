"""Single-process Deep CFR trainer.

Outer loop:
    For t = 1..T:
        For each traverser player p ∈ {SB, BB}:
            Run K external-sampling traversals as p
              → adds to adv_buf[p] and policy_buf
        Refit adv_net[SB] from scratch on adv_buf[SB]
        Refit adv_net[BB] from scratch on adv_buf[BB]
    Train policy_net on policy_buf
    Save final checkpoint

Both `adv_net[p]` and `policy_net` use the same trunk shape (resmlp).
The networks are tiny by default (~370k params each) so re-init + refit
each iteration is cheap.

Usage:
    PYTHONPATH=engine/build:trainer python -m cfr.train --smoke
    PYTHONPATH=engine/build:trainer python -m cfr.train \
        --n-iterations 50 --n-traversals-per-iter 500 --device cpu
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .buffers import ReservoirBuffer
from .config import CFRConfig
from .models import AdvNet, PolicyNet, count_parameters
from .traversal import DEFAULT_MAX_DEPTH, TraversalNets, run_traversals


# ─── AdvNet refit ───────────────────────────────────────────────────────────

def refit_adv_net(net: AdvNet,
                  buf: ReservoirBuffer,
                  cfg: CFRConfig,
                  device: torch.device,
                  use_linear_cfr: bool) -> dict:
    """Train `net` from its current weights on `buf` for n_grad_steps.

    Loss is per-sample MSE between predicted regrets and stored regrets,
    optionally weighted by the stored iteration `t` (Linear CFR).

    Returns a stats dict with loss curve summary.
    """
    net.train(True)
    opt = torch.optim.Adam(net.parameters(),
                           lr=cfg.optim.lr,
                           weight_decay=cfg.optim.weight_decay)
    n_steps = cfg.optim.n_grad_steps_per_refit
    losses = []
    last_log_step = 0
    t_start = time.time()
    for step in range(1, n_steps + 1):
        batch = buf.sample(cfg.optim.batch_size)
        x      = torch.from_numpy(batch.x).to(device)
        target = torch.from_numpy(batch.target).to(device)
        weight = torch.from_numpy(batch.t).to(device) if use_linear_cfr else None

        pred = net(x)
        # Per-sample MSE (averaged across action dim).
        per_sample_mse = ((pred - target) ** 2).mean(dim=-1)   # (B,)
        if weight is not None:
            # Normalize weights so the magnitude doesn't drift with iteration.
            w = weight / weight.mean().clamp(min=1.0)
            loss = (per_sample_mse * w).mean()
        else:
            loss = per_sample_mse.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(net.parameters(), cfg.optim.grad_clip)
        opt.step()
        losses.append(float(loss.item()))
        if step - last_log_step >= max(1, n_steps // 4):
            last_log_step = step
            print(f"      refit step={step}/{n_steps} loss={loss.item():.3f} "
                  f"grad={float(gn):.2f}")
    return {
        "loss_first": losses[0] if losses else float("nan"),
        "loss_last":  losses[-1] if losses else float("nan"),
        "loss_mean":  float(np.mean(losses)) if losses else float("nan"),
        "wall_s":     time.time() - t_start,
    }


# ─── PolicyNet training (final pass) ────────────────────────────────────────

def train_policy_net(policy_net: PolicyNet,
                     buf: ReservoirBuffer,
                     cfg: CFRConfig,
                     device: torch.device,
                     use_linear_cfr: bool) -> dict:
    """Train PolicyNet on the policy buffer once at the end.

    Loss: per-sample cross-entropy with soft strategy targets, masked by
    the strategy itself (illegal slots have target probability 0, so they
    contribute 0 to the CE sum naturally).

    Cross-entropy on soft labels is equivalent (up to constants) to KL
    divergence and is the natural choice for a softmax head with a
    probability-distribution target. MSE on probabilities is also possible
    but gives shallower gradients near 0/1.
    """
    policy_net.train(True)
    opt = torch.optim.Adam(policy_net.parameters(),
                           lr=cfg.optim.lr,
                           weight_decay=cfg.optim.weight_decay)
    n_steps = cfg.train.policy_n_grad_steps
    losses = []
    t_start = time.time()
    last_log_step = 0
    for step in range(1, n_steps + 1):
        batch = buf.sample(cfg.train.policy_batch_size)
        x      = torch.from_numpy(batch.x).to(device)
        target = torch.from_numpy(batch.target).to(device)         # (B, A)
        # `target` already has 0 in illegal slots; build a legal mask from it
        # (any positive entry → legal, plus any zero-probability legal action
        # is rare in the average strategy).
        legal_mask = (target > 0).float()
        # Defensive fallback: if a row's mask is all zero (would imply no
        # legal actions, which is impossible mid-hand), use ones to avoid div-0.
        all_zero = (legal_mask.sum(dim=-1, keepdim=True) == 0)
        legal_mask = torch.where(all_zero, torch.ones_like(legal_mask), legal_mask)
        weight = torch.from_numpy(batch.t).to(device) if use_linear_cfr else None

        logits = policy_net.forward_logits(x)
        masked = logits.masked_fill(legal_mask < 0.5, -1e9)
        log_p  = nn.functional.log_softmax(masked, dim=-1)
        ce = -(target * log_p).sum(dim=-1)                          # (B,)
        if weight is not None:
            w = weight / weight.mean().clamp(min=1.0)
            loss = (ce * w).mean()
        else:
            loss = ce.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(policy_net.parameters(), cfg.optim.grad_clip)
        opt.step()
        losses.append(float(loss.item()))
        if step - last_log_step >= max(1, n_steps // 10):
            last_log_step = step
            print(f"  policy step={step}/{n_steps} loss={loss.item():.4f} "
                  f"grad={float(gn):.2f}")
    return {
        "loss_first": losses[0] if losses else float("nan"),
        "loss_last":  losses[-1] if losses else float("nan"),
        "wall_s":     time.time() - t_start,
    }


# ─── Outer trainer ──────────────────────────────────────────────────────────

def save_checkpoint(path: Path,
                    adv_nets: list[nn.Module],
                    policy_net: nn.Module,
                    iteration: int,
                    cfg: CFRConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "iteration":       iteration,
        "adv_net_sb":      {k: v.detach().cpu() for k, v in adv_nets[0].state_dict().items()},
        "adv_net_bb":      {k: v.detach().cpu() for k, v in adv_nets[1].state_dict().items()},
        "policy_net":      {k: v.detach().cpu() for k, v in policy_net.state_dict().items()},
        "cfg":             cfg,
    }, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt-dir", type=str, default="runs/cfr_latest")
    p.add_argument("--n-iterations", type=int, default=None,
                   help="overrides cfg.train.n_iterations")
    p.add_argument("--n-traversals-per-iter", type=int, default=None)
    p.add_argument("--adv-grad-steps", type=int, default=None,
                   help="overrides cfg.optim.n_grad_steps_per_refit")
    p.add_argument("--policy-grad-steps", type=int, default=None)
    p.add_argument("--no-linear-cfr", action="store_true")
    p.add_argument("--no-reset-adv", action="store_true",
                   help="warm-start AdvNet from previous iter (paper says reset)")
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--no-script", action="store_true",
                   help="disable TorchScript on traversal nets (debugging)")
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                   help="cap recursion at this engine history_size. Default "
                        "= HIST_MAX = 34. Lower it (e.g. 24) for cheaper "
                        "early-iteration traversals.")
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end run (T=2 K=20, ~1 min)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CFRConfig()
    cfg.run.device = args.device
    cfg.run.seed = args.seed
    cfg.run.ckpt_dir = args.ckpt_dir
    if args.n_iterations is not None:
        cfg.train.n_iterations = args.n_iterations
    if args.n_traversals_per_iter is not None:
        cfg.train.n_traversals_per_iter = args.n_traversals_per_iter
    if args.adv_grad_steps is not None:
        cfg.optim.n_grad_steps_per_refit = args.adv_grad_steps
    if args.policy_grad_steps is not None:
        cfg.train.policy_n_grad_steps = args.policy_grad_steps
    if args.hidden is not None:
        cfg.model.hidden = args.hidden
    if args.n_layers is not None:
        cfg.model.n_layers = args.n_layers
    if args.no_reset_adv:
        cfg.train.reset_adv_net_each_iter = False
    use_linear = not args.no_linear_cfr

    if args.smoke:
        cfg.train.n_iterations = 2
        cfg.train.n_traversals_per_iter = 20
        cfg.optim.n_grad_steps_per_refit = 200
        cfg.train.policy_n_grad_steps = 500
        cfg.buffer.adv_capacity = 10_000
        cfg.buffer.policy_capacity = 10_000

    torch.manual_seed(cfg.run.seed)
    device = torch.device(cfg.run.device)
    ckpt_dir = Path(cfg.run.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ─── Build nets and buffers ─────────────────────────────────────────────
    adv_nets = [AdvNet(cfg.model).to(device), AdvNet(cfg.model).to(device)]
    policy_net = PolicyNet(cfg.model).to(device)
    print(f"[cfr] AdvNet params (each):  {count_parameters(adv_nets[0]):,}")
    print(f"[cfr] PolicyNet params:       {count_parameters(policy_net):,}")
    print(f"[cfr] cfg.model={cfg.model}")
    print(f"[cfr] cfg.train={cfg.train}")
    print(f"[cfr] cfg.optim={cfg.optim}")
    print(f"[cfr] device={device}  ckpt_dir={ckpt_dir}  linear_cfr={use_linear}")

    adv_bufs = [
        ReservoirBuffer(cfg.buffer.adv_capacity, cfg.model.x_dim,
                        cfg.model.num_actions,
                        rng=np.random.default_rng(cfg.run.seed + 11)),
        ReservoirBuffer(cfg.buffer.adv_capacity, cfg.model.x_dim,
                        cfg.model.num_actions,
                        rng=np.random.default_rng(cfg.run.seed + 22)),
    ]
    pol_buf = ReservoirBuffer(cfg.buffer.policy_capacity, cfg.model.x_dim,
                              cfg.model.num_actions,
                              rng=np.random.default_rng(cfg.run.seed + 33))

    rng = np.random.default_rng(cfg.run.seed + 44)

    # ─── Outer CFR loop ─────────────────────────────────────────────────────
    t_total_start = time.time()
    for t in range(1, cfg.train.n_iterations + 1):
        print(f"\n[cfr] ════════════ iteration t={t}/{cfg.train.n_iterations} ════════════")

        # Build the inference-time TraversalNets snapshot for this iteration.
        nets = TraversalNets.build(
            [adv_nets[0], adv_nets[1]],
            device=torch.device("cpu"),   # traversal forward is single-state CPU
            script=not args.no_script,
        )

        for p in (0, 1):
            t_p_start = time.time()
            stats = run_traversals(
                env_seed=int(cfg.run.seed + t * 100 + p),
                n_traversals=cfg.train.n_traversals_per_iter,
                traverser=p,
                nets=nets,
                adv_buf_traverser=adv_bufs[p],
                policy_buf=pol_buf,
                t=t,
                rng=rng,
                max_depth=args.max_depth,
            )
            dt = time.time() - t_p_start
            print(f"  traversals(p={p}): K={stats['n_traversals']} wall={dt:.1f}s "
                  f"⇒ {dt/max(1,stats['n_traversals'])*1000:.0f} ms/traversal  "
                  f"adv_buf[{p}]={stats['adv_buf_n_filled']:,}/{cfg.buffer.adv_capacity:,} "
                  f"pol_buf={stats['pol_buf_n_filled']:,}/{cfg.buffer.policy_capacity:,}")

        # Refit AdvNets — paper-default is from scratch each iteration.
        for p in (0, 1):
            if cfg.train.reset_adv_net_each_iter:
                adv_nets[p] = AdvNet(cfg.model).to(device)
            else:
                # TraversalNets.build moved params to CPU in place; restore.
                adv_nets[p].to(device)
            print(f"  refit adv_net[p={p}] on {len(adv_bufs[p])} samples ...")
            r = refit_adv_net(adv_nets[p], adv_bufs[p], cfg, device,
                              use_linear_cfr=use_linear)
            print(f"    loss: first={r['loss_first']:.2f} last={r['loss_last']:.2f} "
                  f"mean={r['loss_mean']:.2f}  wall={r['wall_s']:.1f}s")

        # Periodic checkpoint (every iteration is cheap; we can throttle later).
        path = ckpt_dir / f"cfr_iter_{t:04d}.ckpt"
        save_checkpoint(path, adv_nets, policy_net, t, cfg)
        print(f"  saved {path}")

    # ─── Final policy-net training pass ─────────────────────────────────────
    print(f"\n[cfr] training PolicyNet on {len(pol_buf)} samples "
          f"({cfg.train.policy_n_grad_steps} grad steps)")
    pr = train_policy_net(policy_net, pol_buf, cfg, device,
                          use_linear_cfr=use_linear)
    print(f"  policy: loss first={pr['loss_first']:.4f} last={pr['loss_last']:.4f} "
          f"wall={pr['wall_s']:.1f}s")

    final = ckpt_dir / "cfr_final.ckpt"
    save_checkpoint(final, adv_nets, policy_net, cfg.train.n_iterations, cfg)
    print(f"\n[cfr] DONE.  total wall={time.time()-t_total_start:.1f}s  saved {final}")


if __name__ == "__main__":
    main()
