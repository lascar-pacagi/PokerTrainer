"""DMC value network.

Architecture (flat-sequence encoding, v0.2):

    x (X_DIM,) ──┐
                 ├──▶ concat ──▶ trunk ──▶ v (scalar)
    a (A_DIM,) ──┘

`x` already contains the full flat action history and valid-mask (see
docs/STATE_ENCODING.md), so we don't need an LSTM or any sequence processing
here — the trunk's per-slot weights specialize to positional roles.

Two trunk variants, gated by `cfg.arch`:

    "mlp_v1"     — `Linear → ReLU` stacked `mlp_layers` times, then a scalar
                   head. Original architecture; all checkpoints from
                   gpu_100k_clip10, exploit_*, and the 1M run use this.
    "resmlp_v1"  — Linear stem → `mlp_layers` pre-LN residual blocks
                   (`x = x + W2(GELU(W1(LN(x))))`, FFN width `4×hidden`)
                   → final LN → scalar head. Standard transformer-MLP residual
                   block, which lets us stack deeper without gradient pathology.

At inference we batch the network over all legal actions for one state
(same `x` broadcast across a `(n_legal, A_DIM)` action tensor). During
training we keep only the taken action per transition and target the
terminal Monte-Carlo return (chip delta in BB).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


# ─── Residual MLP block (resmlp_v1) ─────────────────────────────────────────
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


# ─── DMCNet ─────────────────────────────────────────────────────────────────
class DMCNet(nn.Module):
    """Submodule layout depends on `cfg.arch`:
        mlp_v1     — `self.mlp = nn.Sequential(...)`     (legacy ckpt-compat)
        resmlp_v1  — `self.stem`, `self.blocks`, `self.ln_out`, `self.head`
    Forward dispatches on the same flag.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.x_dim + cfg.a_dim
        self.arch = getattr(cfg, "arch", "mlp_v1")

        if self.arch == "mlp_v1":
            layers: list[nn.Module] = []
            prev = in_dim
            for _ in range(cfg.mlp_layers):
                layers += [nn.Linear(prev, cfg.mlp_hidden), nn.ReLU()]
                prev = cfg.mlp_hidden
            layers += [nn.Linear(prev, 1)]
            self.mlp = nn.Sequential(*layers)
        elif self.arch == "resmlp_v1":
            expansion = getattr(cfg, "mlp_expansion", 4)
            self.stem    = nn.Linear(in_dim, cfg.mlp_hidden)
            self.blocks  = nn.ModuleList([
                _ResBlock(cfg.mlp_hidden, expansion) for _ in range(cfg.mlp_layers)
            ])
            self.ln_out  = nn.LayerNorm(cfg.mlp_hidden)
            self.head    = nn.Linear(cfg.mlp_hidden, 1)
        else:
            raise ValueError(f"unknown ModelConfig.arch: {self.arch!r}")

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Value of taking action `a` in state `x`.

        Shapes:
            x: (B, X_DIM)
            a: (B, A_DIM)
        Returns:
            v: (B,) scalar per transition.
        """
        feats = torch.cat([x, a], dim=-1)
        if self.arch == "mlp_v1":
            return self.mlp(feats).squeeze(-1)
        # resmlp_v1
        h = self.stem(feats)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_out(h)
        return self.head(h).squeeze(-1)

    @torch.no_grad()
    def score_legal(self, x: torch.Tensor, a_legal: torch.Tensor) -> torch.Tensor:
        """Score all legal actions for a single state. `a_legal` is (n_legal, A_DIM)."""
        n = a_legal.shape[0]
        x_rep = x.unsqueeze(0).expand(n, -1)
        return self.forward(x_rep, a_legal)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
