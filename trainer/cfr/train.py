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

def _amp_tools(amp: bool, device: torch.device):
    """(autocast-context factory, GradScaler, enabled) for mixed precision.

    AMP runs the matmul-heavy forward/backward in fp16 (Tensor Cores → ~1.5-2×
    on a transformer) while keeping fp32 master weights; the GradScaler scales
    the loss so fp16 gradients don't underflow, and skips the optimizer step if
    it ever sees inf/nan. fp16 is portable across this project's GPUs (Turing
    RTX 8000 + Ampere A40; bf16 would exclude Turing). Only meaningful on CUDA —
    on CPU (or --amp off) everything is enabled=False and the path is identical
    to plain fp32."""
    use = bool(amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use)

    def ctx():
        return torch.amp.autocast(device_type="cuda" if device.type == "cuda"
                                  else "cpu", dtype=torch.float16, enabled=use)
    return ctx, scaler, use


def refit_adv_net(net: AdvNet,
                  buf: ReservoirBuffer,
                  cfg: CFRConfig,
                  device: torch.device,
                  use_linear_cfr: bool,
                  *,
                  forward_module=None,
                  world_size: int = 1,
                  distributed: bool = False,
                  amp: bool = False,
                  verbose: bool = True) -> dict:
    """Train `net` from its current weights on `buf` for n_grad_steps.

    Loss is per-sample MSE between predicted regrets and stored regrets,
    optionally weighted by the stored iteration `t` (Linear CFR).

    DDP (multi-GPU): pass the DDP-wrapped module as `forward_module` (its forward
    all-reduces gradients onto `net`'s parameters, which the optimizer owns) and
    `world_size`/`distributed=True`. Each rank samples from its OWN reservoir
    shard `buf`; gradients are averaged across ranks, so the effective batch is
    `world_size * batch` over the union of shards. The warm-up/epoch cap uses
    that effective batch, and the step count is min-reduced across ranks (all
    ranks MUST run the same number of DDP steps or the all-reduce deadlocks)."""
    net.train(True)
    fwd = forward_module if forward_module is not None else net
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
    # Under DDP each step processes world_size*batch samples over the union of
    # shards, so the epoch math uses that effective batch (and naturally needs
    # fewer steps for the same data).
    refit_max_epochs = float(getattr(cfg.optim, "refit_max_epochs", 10.0))
    n_filled = len(buf)
    eff_batch = cfg.optim.batch_size * max(1, world_size)
    full_steps = cfg.optim.n_grad_steps_per_refit
    warm_cap = max(1, int(refit_max_epochs * max(1, n_filled) * max(1, world_size)
                          / eff_batch))
    n_steps = min(full_steps, warm_cap)
    if distributed:
        # Shards differ by a few samples → step counts could differ by 1, which
        # would deadlock the gradient all-reduce. Agree on the minimum.
        from . import distributed as ddist
        n_steps = ddist.all_reduce_min_int(
            n_steps, ddist.DistInfo(True, 0, world_size, 0, device))
    if verbose and n_steps < full_steps:
        print(f"      [warm-up] buffer={n_filled:,} (×{world_size} ranks) → refit "
              f"{n_steps}/{full_steps} steps (~{refit_max_epochs:g} epochs) "
              f"to protect the net")
    amp_ctx, scaler, _ = _amp_tools(amp, device)
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

        with amp_ctx():
            pred = fwd(tokens, pad_mask, dpos)                   # (B, NUM_ACTIONS)
            # NOTE (paper deviation, benign): mean over ALL NUM_ACTIONS slots —
            # illegal slots carry target 0, so the net also learns "0 on illegal".
            # This dilutes the printed loss (e.g. 3-legal/11 ⇒ ~8/11 of the terms
            # are near-zero) but regret matching re-masks at use time. Masking the
            # loss to legal slots would need the legal mask stored in the buffer.
            per_sample_mse = ((pred - target) ** 2).mean(dim=-1)  # (B,)
            if weight is not None:
                # Normalize weights so the magnitude doesn't drift with iter.
                w = weight / weight.mean().clamp(min=1.0)
                loss = (per_sample_mse * w).mean()
            else:
                loss = per_sample_mse.mean()

        opt.zero_grad(set_to_none=True)
        # AMP-safe step: scale → backward → unscale (so the clip sees true-scale
        # grads) → clip → step (skipped by the scaler if it saw inf/nan) → update.
        # All no-ops when AMP is disabled, so the fp32 path is unchanged.
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = nn.utils.clip_grad_norm_(net.parameters(), cfg.optim.grad_clip)
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.item()))
        if verbose and step - last_log_step >= max(1, n_steps // 4):
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
                     use_linear_cfr: bool,
                     *,
                     amp: bool = False) -> dict:
    """Train PolicyNet on the policy buffer once at the end.

    Loss: per-sample cross-entropy with soft strategy targets, masked by
    the strategy itself (illegal slots have target probability 0, so they
    contribute 0 to the CE sum naturally). `amp` → fp16 autocast (see _amp_tools).
    """
    policy_net.train(True)
    opt = torch.optim.Adam(policy_net.parameters(),
                           lr=cfg.optim.lr,
                           weight_decay=cfg.optim.weight_decay)
    n_steps = cfg.train.policy_n_grad_steps
    amp_ctx, scaler, _ = _amp_tools(amp, device)
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

        with amp_ctx():
            logits = policy_net.forward_logits(tokens, pad_mask, dpos)
            # Mask illegal slots, then log_softmax in fp32 (numerically safer).
            # Fill with the dtype's most-negative finite value: under --amp
            # `logits` is fp16 and a literal -1e9 overflows Half (max |x| ≈
            # 65504), raising inside masked_fill before the .float() upcast.
            # finfo(dtype).min is representable by construction and still drives
            # illegal-slot probabilities to 0 after softmax.
            masked = logits.masked_fill(legal_mask < 0.5,
                                        torch.finfo(logits.dtype).min)
            log_p  = nn.functional.log_softmax(masked.float(), dim=-1)
            ce = -(target * log_p).sum(dim=-1)                   # (B,)
            if weight is not None:
                w = weight / weight.mean().clamp(min=1.0)
                loss = (ce * w).mean()
            else:
                loss = ce.mean()

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = nn.utils.clip_grad_norm_(policy_net.parameters(), cfg.optim.grad_clip)
        scaler.step(opt)
        scaler.update()
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
    # Atomic write: serialize to a temp file, then rename into place. A cluster
    # wall-time kill (SIGTERM→SIGKILL) mid-write would otherwise leave a
    # truncated cfr_iter_*.ckpt that find_latest_checkpoint picks on restart and
    # torch.load then chokes on. os.replace (Path.replace) is atomic on the same
    # filesystem, so the final path is always either the old ckpt or a complete
    # new one — never a partial.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "iteration":       iteration,
        "adv_net_sb":      {k: v.detach().cpu() for k, v in adv_nets[0].state_dict().items()},
        "adv_net_bb":      {k: v.detach().cpu() for k, v in adv_nets[1].state_dict().items()},
        "policy_net":      {k: v.detach().cpu() for k, v in policy_net.state_dict().items()},
        "cfg":             cfg,
    }, tmp)
    tmp.replace(path)


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
