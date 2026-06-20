"""Refit/training helpers for the Deep CFR transformer trainers.

This module exposes three functions used by both `cfr_coro` (single-process,
coroutine-batched) and `cfr_mp_gpu` (multi-process actors + GPU learner):

  * `refit_adv_net(net, buf, cfg, device, use_linear_cfr) -> dict`
        Train an AdvNet on its reservoir buffer for N grad steps.
        Loss: per-sample MSE against stored regret targets, Linear CFR-weighted.

  * `train_policy_net(policy_net, buf, cfg, device, use_linear_cfr) -> dict`
        Final-pass training of the PolicyNet on the policy buffer.
        Loss: per-sample cross-entropy against stored strategy targets.

  * `save_checkpoint(path, adv_nets, policy_net, iteration, cfg)`
        Disk serialization.

The previous single-process trainer with synchronous traversal (used by
the now-removed `cfr.train` CLI) is gone. Use `cfr.cfr_coro` for laptop
runs and `cfr.cfr_mp_gpu` for cluster runs.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .buffers import ReservoirBuffer
from .config import CFRConfig
from .models import AdvNet, PolicyNet


# ─── AdvNet refit ───────────────────────────────────────────────────────────

def refit_adv_net(net: AdvNet,
                  buf: ReservoirBuffer,
                  cfg: CFRConfig,
                  device: torch.device,
                  use_linear_cfr: bool) -> dict:
    """Train `net` from its current weights on `buf` for n_grad_steps.

    Loss is per-sample MSE between predicted regrets and stored regrets,
    optionally weighted by the stored iteration `t` (Linear CFR).
    """
    net.train(True)
    opt = torch.optim.Adam(net.parameters(),
                           lr=cfg.optim.lr,
                           weight_decay=cfg.optim.weight_decay)
    # Cold-buffer warm-up: cap the refit at ~`refit_max_epochs` passes over the
    # CURRENTLY STORED samples, so a small reservoir (the first iterations of a
    # run, or the empty buffer right after a --resume) cannot overfit-and-destroy
    # the net with a full n_grad_steps grind. This is a strict no-op once the
    # buffer is large (min() keeps the full count): e.g. at 7M samples / batch
    # 1024, even 10 epochs is ~68k >> 15k, so steady-state is unchanged; at 76k
    # samples it caps ~740 steps (10 epochs) instead of 15k (≈200 epochs).
    refit_max_epochs = float(getattr(cfg.optim, "refit_max_epochs", 10.0))
    n_filled = len(buf)
    full_steps = cfg.optim.n_grad_steps_per_refit
    warm_cap = max(1, int(refit_max_epochs * max(1, n_filled) / cfg.optim.batch_size))
    n_steps = min(full_steps, warm_cap)
    if n_steps < full_steps:
        print(f"      [warm-up] buffer={n_filled:,} → refit {n_steps}/{full_steps} "
              f"steps (~{refit_max_epochs:g} epochs) to protect the net")
    losses: list[float] = []
    last_log_step = 0
    t_start = time.time()
    for step in range(1, n_steps + 1):
        batch = buf.sample(cfg.optim.batch_size)
        tokens   = torch.from_numpy(batch.tokens).to(device)
        pad_mask = torch.from_numpy(batch.pad_mask).to(device)
        dpos     = torch.from_numpy(batch.decision_pos).to(device)
        target   = torch.from_numpy(batch.target).to(device)
        weight   = torch.from_numpy(batch.t).to(device) if use_linear_cfr else None

        pred = net(tokens, pad_mask, dpos)                       # (B, NUM_ACTIONS)
        # NOTE (paper deviation, benign): mean over ALL NUM_ACTIONS slots —
        # illegal slots carry target 0, so the net also learns "0 on illegal".
        # This dilutes the printed loss (e.g. 3-legal/11 ⇒ ~8/11 of the terms
        # are near-zero) but regret matching re-masks at use time. Masking the
        # loss to legal slots would need the legal mask stored in the buffer.
        per_sample_mse = ((pred - target) ** 2).mean(dim=-1)     # (B,)
        if weight is not None:
            # Normalize weights so the magnitude doesn't drift with iter.
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
    """
    policy_net.train(True)
    opt = torch.optim.Adam(policy_net.parameters(),
                           lr=cfg.optim.lr,
                           weight_decay=cfg.optim.weight_decay)
    n_steps = cfg.train.policy_n_grad_steps
    losses: list[float] = []
    t_start = time.time()
    last_log_step = 0
    for step in range(1, n_steps + 1):
        batch = buf.sample(cfg.train.policy_batch_size)
        tokens   = torch.from_numpy(batch.tokens).to(device)
        pad_mask = torch.from_numpy(batch.pad_mask).to(device)
        dpos     = torch.from_numpy(batch.decision_pos).to(device)
        target   = torch.from_numpy(batch.target).to(device)
        # Stored target is a probability distribution; build a legal mask
        # from it (any positive entry → legal). Defensive: if all-zero
        # (impossible mid-hand), use ones to avoid div-0.
        legal_mask = (target > 0).float()
        all_zero = (legal_mask.sum(dim=-1, keepdim=True) == 0)
        legal_mask = torch.where(all_zero, torch.ones_like(legal_mask), legal_mask)
        weight = torch.from_numpy(batch.t).to(device) if use_linear_cfr else None

        logits = policy_net.forward_logits(tokens, pad_mask, dpos)
        masked = logits.masked_fill(legal_mask < 0.5, -1e9)
        log_p  = nn.functional.log_softmax(masked, dim=-1)
        ce = -(target * log_p).sum(dim=-1)                       # (B,)
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


# ─── Checkpointing ──────────────────────────────────────────────────────────


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


def find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Newest resumable checkpoint in `ckpt_dir`, or None.

    Prefers the highest-numbered ``cfr_iter_NNNN.ckpt`` (those carry the
    iteration count to continue from); falls back to ``cfr_final.ckpt``."""
    iters = sorted(ckpt_dir.glob("cfr_iter_*.ckpt"))
    if iters:
        return iters[-1]                      # zero-padded names sort by iteration
    final = ckpt_dir / "cfr_final.ckpt"
    return final if final.exists() else None


def load_checkpoint(path: Path,
                    adv_nets: list[nn.Module],
                    policy_net: nn.Module,
                    device: torch.device) -> int:
    """Restore both AdvNets + the PolicyNet in place; return the saved iteration.

    Weights only — the reservoir buffers are NOT checkpointed (they refill over
    the next few iterations). The caller protects the just-loaded nets from a
    cold-buffer refit via the warm-up scaling in ``refit_adv_net``. We trust our
    own checkpoint (weights_only=False to allow the embedded cfg object)."""
    blob = torch.load(path, map_location=device, weights_only=False)
    adv_nets[0].load_state_dict(blob["adv_net_sb"])
    adv_nets[1].load_state_dict(blob["adv_net_bb"])
    policy_net.load_state_dict(blob["policy_net"])
    return int(blob.get("iteration", 0))
