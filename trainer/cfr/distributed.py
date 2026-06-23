"""Minimal torch.distributed helpers for multi-GPU Deep CFR (cfr_coro).

DDP here parallelizes the dominant cost — the AdvNet *refit* (~95% of each
iteration's wall time once the net is large) — and shards *collection* across
ranks. Each rank owns one GPU, runs its slice of the traversals into its own
reservoir shard, and during the refit every rank samples from its shard while
DDP all-reduces the gradients: the effective batch becomes
``world_size * per_gpu_batch`` over the union of shards.

Launched with ``torchrun --nproc_per_node=N`` (sets RANK/WORLD_SIZE/LOCAL_RANK).
With no such env (plain ``python -m``), ``setup`` returns a disabled DistInfo and
the trainer runs exactly as the single-process path it always had.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup(default_device: str = "cuda:0") -> DistInfo:
    """Init the process group if launched under torchrun; else single-process.

    `default_device` is used only in the single-process case (honours the old
    --device flag). Under torchrun each rank pins cuda:LOCAL_RANK."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistInfo(False, 0, 1, 0, torch.device(default_device))

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # GPU → NCCL (the real cluster path, one rank per cuda:LOCAL_RANK); CPU → Gloo
    # (lets the DDP logic be exercised/tested without GPUs). An explicit
    # `--device cpu` forces the CPU path even where torch reports a (phantom)
    # CUDA. Rendezvous env (MASTER_ADDR/PORT) is set by torchrun.
    use_cuda = torch.cuda.is_available() and not str(default_device).startswith("cpu")
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl")
    else:
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo")
    return DistInfo(True, rank, world_size, local_rank, device)


def barrier(info: DistInfo) -> None:
    if info.enabled:
        dist.barrier()


def all_reduce_min_int(value: int, info: DistInfo) -> int:
    """Min of `value` across ranks (used to agree on a refit step count — every
    rank MUST run the same number of DDP steps or the all-reduce deadlocks)."""
    if not info.enabled:
        return value
    t = torch.tensor([int(value)], device=info.device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item())


def cleanup(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.destroy_process_group()
