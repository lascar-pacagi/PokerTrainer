"""Offline DDP average-policy harvest for Deep CFR (cfr_coro).

The average strategy (PolicyNet) is a *separate, heavier* learning problem than
the AdvNet refit: it fits the cumulative strategy reservoir (100M+ snapshots
spanning every iteration, dominated by early near-uniform strategies), so it
needs many epochs over a large batch to escape the marginal-predictor plateau
and actually condition on the hole cards. Doing that inline in cfr_coro meant
rank 0 trained solo for hours while the other ranks idled — wasteful and
undertrained at any affordable step count. So it lives here instead:

  * loads the strategy reservoir SHARDS saved by cfr_coro
    (cfr_buffers_rank*.npz — each rank persisted its own shard);
  * trains ONE PolicyNet across ALL GPUs (DDP: each rank on its own shard,
    gradients all-reduced → the net sees the union of shards);
  * recovers the model architecture + AdvNets from the run's checkpoint (its
    embedded cfg), so nothing about the net shape is re-specified here;
  * saves a complete, deployable checkpoint (AdvNets passed through unchanged +
    the freshly-harvested PolicyNet).

Decoupled from the CFR loop → train it to convergence (big --steps/--batch)
without touching the collection/refit run.

Launch (multi-GPU — use the SAME nproc as the training run so the per-rank
shard files line up):
    torchrun --standalone --nnodes=1 --nproc_per_node=N \
        -m cfr.harvest_policy --ckpt-dir /path/to/run \
        --steps 200000 --batch 8192 --amp
Single GPU / CPU:
    python -m cfr.harvest_policy --ckpt-dir /path/to/run --steps 200000 ...
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .buffers import ReservoirBuffer, load_buffers
from .models import AdvNet, PolicyNet, count_parameters
from .train import (train_policy_net, save_checkpoint, load_checkpoint,
                    find_latest_checkpoint)
from .probe import run_policy_street_probes, format_street_probe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", required=True,
                   help="run directory holding cfr_iter_*.ckpt + "
                        "cfr_buffers_rank*.npz (the harvest reads both).")
    p.add_argument("--net-from", default=None,
                   help="specific checkpoint for cfg + AdvNets (default: latest "
                        "cfr_iter_*.ckpt / cfr_final.ckpt in --ckpt-dir).")
    p.add_argument("--out", default="cfr_policy_harvested.ckpt",
                   help="output checkpoint name (written into --ckpt-dir).")
    p.add_argument("--steps", type=int, default=200_000,
                   help="policy gradient steps (each rank runs this many).")
    p.add_argument("--batch", type=int, default=8192,
                   help="per-rank policy batch size (effective = batch×nproc).")
    p.add_argument("--lr", type=float, default=None,
                   help="Adam lr override (default: the run's cfg.optim.lr).")
    p.add_argument("--amp", action="store_true",
                   help="fp16 mixed precision (Tensor Cores; ~1.5-2×).")
    p.add_argument("--no-linear-cfr", action="store_true",
                   help="disable the Linear-CFR t-weighting of samples.")
    p.add_argument("--warm-start", action="store_true",
                   help="init the PolicyNet from the checkpoint's policy_net "
                        "instead of fresh (default: fresh random init).")
    p.add_argument("--policy-capacity", type=int, default=None,
                   help="override reservoir capacity if it can't be read from "
                        "the checkpoint cfg (must match the saved buffers).")
    p.add_argument("--probe-hands", type=int, default=400,
                   help="AA/72o hands for the final per-street strategy read.")
    p.add_argument("--device", default="cuda:0",
                   help="single-process device (ignored under torchrun; each "
                        "rank pins cuda:LOCAL_RANK).")
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    from . import distributed as ddist
    from torch.nn.parallel import DistributedDataParallel as DDP
    dist = ddist.setup(default_device=args.device)
    device = dist.device

    def log(*a, **k):
        if dist.is_main:
            print(*a, **k)

    ckpt_dir = Path(args.ckpt_dir)

    # ── Source checkpoint → cfg (model shape) + AdvNets (pass-through) ───────
    src = Path(args.net_from) if args.net_from else find_latest_checkpoint(ckpt_dir)
    if src is None or not src.exists():
        raise SystemExit(f"[harvest] no checkpoint in {ckpt_dir} "
                         f"(looked for cfr_iter_*.ckpt / cfr_final.ckpt).")
    # weights_only=False to unpickle the embedded CFRConfig object — same trust
    # assumption as train.load_checkpoint (our own checkpoint, not user input).
    blob = torch.load(src, map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    log(f"[harvest] source checkpoint: {src} (iteration {blob.get('iteration', 0)})")
    log(f"[harvest] cfg.model={cfg.model}")

    # Harvest hyperparameters override the run's policy-training config.
    cfg.train.policy_n_grad_steps = args.steps
    cfg.train.policy_batch_size = args.batch
    if args.lr is not None:
        cfg.optim.lr = args.lr
    if args.policy_capacity is not None:
        cfg.buffer.policy_capacity = args.policy_capacity
    use_linear = not args.no_linear_cfr

    # AdvNets: load once (pass through into the output ckpt unchanged). PolicyNet:
    # fresh random init by default — the average net is trained from scratch on
    # the reservoir; --warm-start keeps the checkpoint's (undertrained) weights.
    adv_nets = [AdvNet(cfg.model).to(device), AdvNet(cfg.model).to(device)]
    policy_net = PolicyNet(cfg.model).to(device)
    iteration = load_checkpoint(src, adv_nets, policy_net, device)
    if not args.warm_start:
        policy_net = PolicyNet(cfg.model).to(device)   # discard loaded weights
    log(f"[harvest] PolicyNet params: {count_parameters(policy_net):,}  "
        f"init={'warm (from ckpt)' if args.warm_start else 'fresh'}")

    # ── Load THIS rank's strategy shard (only the 'pol' buffer) ─────────────
    buffers_path = (ckpt_dir / (f"cfr_buffers_rank{dist.rank}.npz" if dist.enabled
                                else "cfr_buffers.npz"))
    if not buffers_path.exists():
        raise SystemExit(
            f"[harvest] missing buffer shard {buffers_path}. Launch with the "
            f"SAME nproc as the training run (rank {dist.rank} of "
            f"{dist.world_size}) so every shard file is present.")
    pol_buf = ReservoirBuffer(cfg.buffer.policy_capacity, cfg.model.num_actions,
                              rng=np.random.default_rng(args.seed + dist.rank))
    bt0 = time.time()
    buf_iter = load_buffers(str(buffers_path), {"pol": pol_buf})
    log(f"[harvest] loaded pol shards: rank{dist.rank}={len(pol_buf):,} "
        f"(saved at iter {buf_iter}, {time.time()-bt0:.1f}s)")
    if len(pol_buf) == 0:
        raise SystemExit(f"[harvest] rank {dist.rank}: empty pol buffer.")

    # ── DDP-wrap + train ────────────────────────────────────────────────────
    if dist.enabled:
        log(f"[harvest] DDP world_size={dist.world_size}; effective batch = "
            f"{dist.world_size}×{args.batch} over the union of shards.")
    fwd = (DDP(policy_net, device_ids=([dist.local_rank] if device.type == "cuda"
                                       else None))
           if dist.enabled else None)
    log(f"\n[harvest] training PolicyNet: {args.steps} steps × batch {args.batch}"
        f"{' (amp)' if args.amp else ''} ...")
    pr = train_policy_net(policy_net, pol_buf, cfg, device,
                          use_linear_cfr=use_linear,
                          forward_module=fwd, world_size=dist.world_size,
                          distributed=dist.enabled, amp=args.amp,
                          verbose=dist.is_main)
    log(f"[harvest] loss first={pr['loss_first']:.4f} last={pr['loss_last']:.4f} "
        f"wall={pr['wall_s']:.1f}s")
    del fwd

    # ── Final per-street strategy read + save (rank 0) ──────────────────────
    if dist.is_main:
        policy_net.eval()   # no-op at dropout=0, but the harvest is a deliverable
        for r in run_policy_street_probes(policy_net, device,
                                          n_hands=args.probe_hands,
                                          base_seed=args.seed):
            print(format_street_probe(r, iter_t=iteration))
        out = ckpt_dir / args.out
        save_checkpoint(out, adv_nets, policy_net, iteration, cfg)
        print(f"[harvest] saved {out}")

    ddist.barrier(dist)
    ddist.cleanup(dist)


if __name__ == "__main__":
    main()
