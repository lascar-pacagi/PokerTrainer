"""Deep CFR networks.

Two networks share an identical resmlp trunk and differ only at the head:

    AdvNet:    Linear(in=hidden, out=NUM_ACTIONS)             — raw regret per action.
    PolicyNet: Linear(in=hidden, out=NUM_ACTIONS) → softmax   — average strategy.

Crucially, both networks take only `x` as input — *not* `(x, a)` like the DMC
DMCNet does. The DMC net evaluates one action at a time (`v(s, a)`), which
forced an N-fan-out forward at each decision (N = number of legal actions).
The Deep CFR setup wants all action values (regrets / probabilities) at once,
so the head outputs a `NUM_ACTIONS`-dim vector and we mask illegal positions
afterwards. Single forward per state — meaningfully cheaper, especially in
the recursive traversal loop.

The trunk is the same `resmlp_v1` block stack from `dmc/models.py`; we
duplicate it here intentionally because Deep CFR is a separate trainer and
we don't want a shared-trunk import dependency that couples the two
trainers' release cadences.
"""
from __future__ import annotations

from typing import Final

import torch
from torch import nn
from torch.nn import functional as F

from .config import CFRModelConfig


class _ResBlock(nn.Module):
    """Pre-LN transformer-style FFN block:  x + W2(GELU(W1(LN(x))))."""

    def __init__(self, hidden: int, expansion: int):
        super().__init__()
        ffn = hidden * expansion
        self.ln  = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, ffn)
        self.fc2 = nn.Linear(ffn, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.gelu(self.fc1(self.ln(x))))


class _Trunk(nn.Module):
    """Shared trunk: stem → N residual blocks → LayerNorm. Returns hidden features.

    Lives as its own module so the AdvNet/PolicyNet heads can be attached
    cleanly and the parameter counts read sensibly.
    """

    __constants__ = ["arch"]
    arch: Final[str]

    def __init__(self, cfg: CFRModelConfig):
        super().__init__()
        self.arch = cfg.arch
        if self.arch == "resmlp_v1":
            self.stem    = nn.Linear(cfg.x_dim, cfg.hidden)
            self.blocks  = nn.ModuleList([
                _ResBlock(cfg.hidden, cfg.mlp_expansion)
                for _ in range(cfg.n_layers)
            ])
            self.ln_out  = nn.LayerNorm(cfg.hidden)
            self.mlp     = nn.Identity()                # placeholder for ckpt-compat
        elif self.arch == "mlp_v1":
            layers = []
            prev = cfg.x_dim
            for _ in range(cfg.n_layers):
                layers += [nn.Linear(prev, cfg.hidden), nn.ReLU()]
                prev = cfg.hidden
            self.mlp = nn.Sequential(*layers)
            self.stem    = nn.Identity()
            self.blocks  = nn.ModuleList([])
            self.ln_out  = nn.Identity()
        else:
            raise ValueError(f"unknown arch: {self.arch!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch == "mlp_v1":
            return self.mlp(x)
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        return self.ln_out(h)


class AdvNet(nn.Module):
    """Predicts cumulative regret per action from `x`.

    Output: (B, NUM_ACTIONS) raw regrets, no activation. Negative values
    are valid (CFR cumulative regrets can be negative); regret-matching
    strips them via clamp(min=0) at policy-derivation time.
    """

    def __init__(self, cfg: CFRModelConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = _Trunk(cfg)
        self.head  = nn.Linear(cfg.hidden, cfg.num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


class PolicyNet(nn.Module):
    """Predicts the average strategy as a probability distribution.

    Output: (B, NUM_ACTIONS) probabilities. Caller must pass `legal_mask`
    so illegal positions can be set to -∞ before softmax. Forward returns
    *probabilities*, not logits — convenient for inference. For training,
    use `forward_logits()` and apply your own loss.
    """

    def __init__(self, cfg: CFRModelConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = _Trunk(cfg)
        self.head  = nn.Linear(cfg.hidden, cfg.num_actions)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))

    def forward(self,
                x: torch.Tensor,
                legal_mask: torch.Tensor) -> torch.Tensor:
        """Return masked-softmax strategy. Illegal slots are exactly 0.

        legal_mask: (B, NUM_ACTIONS) with 1.0 for legal, 0.0 otherwise.
        """
        logits = self.forward_logits(x)
        # Set illegal slots to a large negative number so softmax masks them.
        # We can't use -inf because it breaks the gradient if every slot is
        # illegal (impossible in practice, but defensive).
        masked = logits.masked_fill(legal_mask < 0.5, -1e9)
        return F.softmax(masked, dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
