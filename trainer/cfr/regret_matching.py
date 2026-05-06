"""Regret matching: turn cumulative-regret predictions into a strategy.

Given:
    pred_regrets : shape (NUM_ACTIONS,) — AdvNet output (can be negative).
    legal_mask   : shape (NUM_ACTIONS,) — 1.0 for legal actions, 0.0 otherwise.

Compute the current strategy:
    1. Mask out illegal actions (set their regret to 0 via legal_mask).
    2. Take positive part:  R+(a) = max(0, regret(a)) * legal_mask(a).
    3. Normalize: σ(a) = R+(a) / Σ_a R+(a).
    4. If Σ_a R+(a) == 0 (all legal regrets non-positive), fall back to
       uniform over legal actions.

This is the "regret matching" step of vanilla CFR; in Deep CFR the
cumulative regrets come from the network instead of being tabular.

Both numpy and torch variants are provided. Numpy is used in the
traversal loop (CPU-only, single state at a time); torch is used in
the training loss for the PolicyNet (batched).
"""
from __future__ import annotations

import numpy as np
import torch


def regret_matching_np(pred_regrets: np.ndarray,
                       legal_mask: np.ndarray,
                       eps: float = 0.0) -> np.ndarray:
    """Single-state regret matching. Returns a strategy (sums to 1 over legal).

    Output has 0.0 in every illegal slot.
    `eps` (optional): mix in `eps` × uniform-legal for floor-exploration.
    """
    assert pred_regrets.shape == legal_mask.shape, (
        pred_regrets.shape, legal_mask.shape)
    pos = np.maximum(pred_regrets, 0.0) * legal_mask
    z = pos.sum()
    if z > 0.0:
        sigma = pos / z
    else:
        n_legal = max(1.0, legal_mask.sum())
        sigma = legal_mask / n_legal
    if eps > 0.0:
        n_legal = max(1.0, legal_mask.sum())
        unif = legal_mask / n_legal
        sigma = (1.0 - eps) * sigma + eps * unif
    return sigma.astype(np.float32, copy=False)


def regret_matching_batch(pred_regrets: torch.Tensor,
                          legal_mask: torch.Tensor) -> torch.Tensor:
    """Batched regret matching. Inputs (B, NUM_ACTIONS), output same shape.

    Used inside training loops where we want gradient-free strategy targets.
    """
    pos = torch.clamp(pred_regrets, min=0.0) * legal_mask
    z = pos.sum(dim=-1, keepdim=True)                          # (B, 1)
    safe_z = torch.where(z > 0, z, torch.ones_like(z))          # avoid div-0
    sigma = pos / safe_z
    # For rows with all-non-positive regrets, fall back to uniform-legal.
    legal_count = legal_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    uniform = legal_mask / legal_count
    return torch.where(z > 0, sigma, uniform)
