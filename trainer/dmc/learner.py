"""Gradient step on sampled transitions.

DouZero's learner (dmc/dmc.py::learn) runs RMSProp(lr=1e-4, momentum=0, eps=1e-5)
with an L2 grad clip at 40. We keep the same recipe: it's the empirical setting
that made DMC converge on Doudizhu, and we're deliberately staying close to it
until we have a reason to diverge.

Loss is plain MSE(net(x, a, z), r) — pure Monte-Carlo target, no bootstrapping,
no target network. Simpler than DQN because the return is already a sample of
the true action-value at hand end.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import OptimConfig
from .models import DMCNet


def build_optimizer(net: DMCNet, cfg: OptimConfig) -> torch.optim.Optimizer:
    """RMSProp — DouZero's choice, kept identical here."""
    return torch.optim.RMSprop(
        net.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        eps=cfg.eps,
    )


def learn_step(net: DMCNet,
               optim: torch.optim.Optimizer,
               batch: dict,
               device: torch.device,
               grad_clip: float) -> dict:
    """One gradient step on a sampled batch.

    `batch` is the dict returned by MCBuffer.sample — numpy arrays for x, a, z, r.
    Returns a small stats dict suitable for logging.
    """
    net.train(True)

    x = torch.from_numpy(batch["x"]).to(device)
    a = torch.from_numpy(batch["a"]).to(device)
    r = torch.from_numpy(batch["r"]).to(device)

    v = net.forward(x, a)                     # (B,)
    loss = nn.functional.mse_loss(v, r)

    optim.zero_grad(set_to_none=True)
    loss.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
    optim.step()

    with torch.no_grad():
        mean_v = float(v.mean().item())
        mean_r = float(r.mean().item())

    return {
        "loss":       float(loss.item()),
        "grad_norm":  float(total_norm.item()),
        "mean_pred":  mean_v,
        "mean_target": mean_r,
    }
