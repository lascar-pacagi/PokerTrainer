"""Transformer-based Deep CFR networks.

═══════════════════════════════════════════════════════════════════════════════
WHY A TRANSFORMER (and not the previous resmlp_v1 on a flat 816-dim x)
═══════════════════════════════════════════════════════════════════════════════

The diagnostic at iter 80 showed that a trained resmlp AdvNet's sigma varied
by only 0.017 across 38 different hole cards. Hole cards were 52 out of 816
input dims; the Linear(816, 512) stem learned to focus on the larger action-
history region where the regret-target variance was higher, and the hole
card features got drowned out.

Token-based encoding (see cfr/tokenize.py) puts each card in its own token
slot with a dedicated learned embedding. A transformer over the resulting
sequence cannot ignore card tokens — every position is treated symmetrically
by attention, with attention heads free to learn arbitrary card↔action
interactions.

Architecture (default):
  vocab_size  = 106  (tokens)
  d_model     = 128
  n_layers    = 4    (pre-LN transformer blocks)
  n_heads     = 4    (so head_dim = 32)
  d_ff        = 512  (4× d_model FFN inner)
  ~700k params total — substantially smaller than the 17M-param resmlp,
  and decoupled from input-vector dimension.

═══════════════════════════════════════════════════════════════════════════════
INPUT/OUTPUT CONTRACT
═══════════════════════════════════════════════════════════════════════════════

Forward signature for both AdvNet and PolicyNet:

    tokens       : LongTensor (B, L)     — padded token ids
    pad_mask     : BoolTensor (B, L)     — True at PAD positions
    decision_pos : LongTensor (B,)       — index of DECISION in each row

AdvNet returns:
    raw regrets : FloatTensor (B, NUM_ACTIONS)

PolicyNet returns (with legal_mask kwarg):
    probabilities : FloatTensor (B, NUM_ACTIONS)
    (illegal slots are exactly 0; masked-softmax handles the mask.)
"""
from __future__ import annotations

import math
from typing import Final

import torch
from torch import nn
from torch.nn import functional as F

from .config import CFRModelConfig


# ═══════════════════════════════════════════════════════════════════════════
# PRE-LN TRANSFORMER BLOCK
# ═══════════════════════════════════════════════════════════════════════════
#
# Pre-LN (LayerNorm before residual sub-layers) is more stable than post-LN
# for small models and shallow stacks; it lets us drop the warmup-LR
# scheduling that post-LN often needs. We use PyTorch's built-in
# `nn.TransformerEncoderLayer` with `norm_first=True` for the pre-LN variant.

class _Trunk(nn.Module):
    """Token embedding + learned positional encoding + N transformer layers.

    The transformer outputs a per-position hidden state of dim `d_model`.
    Downstream heads slice out the [DECISION]-position embedding for
    final prediction.
    """
    def __init__(self, cfg: CFRModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model,
                                      padding_idx=0)
        self.pos_emb   = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation=F.gelu,        # GELU matches resmlp_v1
            batch_first=True,         # (B, L, d_model) tensor ordering
            norm_first=True,          # pre-LN
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.ln_out = nn.LayerNorm(cfg.d_model)

    def forward(self,
                tokens: torch.Tensor,             # (B, L) int64
                pad_mask: torch.Tensor,           # (B, L) bool, True at PAD
                ) -> torch.Tensor:                # (B, L, d_model)
        B, L = tokens.shape
        # Token + position embeddings.
        pos_ids = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        h = self.token_emb(tokens) + self.pos_emb(pos_ids)
        # PyTorch's TransformerEncoder: src_key_padding_mask=True means
        # "ignore this position" → matches our pad_mask convention.
        h = self.transformer(h, src_key_padding_mask=pad_mask)
        return self.ln_out(h)


def _gather_decision_state(h: torch.Tensor,             # (B, L, D)
                           decision_pos: torch.Tensor   # (B,) int64
                           ) -> torch.Tensor:           # (B, D)
    """Pick out the hidden state at each row's [DECISION] index."""
    B, L, D = h.shape
    idx = decision_pos.view(B, 1, 1).expand(B, 1, D)
    return h.gather(dim=1, index=idx).squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════
# AdvNet
# ═══════════════════════════════════════════════════════════════════════════


class AdvNet(nn.Module):
    """Predicts cumulative regret per action from the token sequence.

    Head is a Linear(d_model → NUM_ACTIONS). Zero-initialized so a freshly-
    constructed AdvNet returns all-zero regrets — `regret_matching_np`'s
    fallback (highest-regret legal action, exact ties split equally) then
    degrades to uniform-over-legal sigma, mirroring the Brown 2019 Deep
    CFR paper's "iter 1 uses uniform sigma" prescription without needing a
    t==1 special case in the traversal. See [[cfr-iter1-degenerate-sigma-trap]]
    for the prior incident that motivated this design.
    """

    def __init__(self, cfg: CFRModelConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = _Trunk(cfg)
        self.head  = nn.Linear(cfg.d_model, cfg.num_actions)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self,
                tokens: torch.Tensor,           # (B, L)
                pad_mask: torch.Tensor,         # (B, L)
                decision_pos: torch.Tensor,     # (B,)
                ) -> torch.Tensor:              # (B, NUM_ACTIONS)
        h = self.trunk(tokens, pad_mask)
        dh = _gather_decision_state(h, decision_pos)
        return self.head(dh)


# ═══════════════════════════════════════════════════════════════════════════
# PolicyNet
# ═══════════════════════════════════════════════════════════════════════════


class PolicyNet(nn.Module):
    """Predicts the average strategy as a probability distribution.

    Output: (B, NUM_ACTIONS) probabilities. Caller must pass `legal_mask`
    so illegal positions can be set to -∞ before softmax.
    """

    def __init__(self, cfg: CFRModelConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = _Trunk(cfg)
        self.head  = nn.Linear(cfg.d_model, cfg.num_actions)

    def forward_logits(self,
                       tokens: torch.Tensor,
                       pad_mask: torch.Tensor,
                       decision_pos: torch.Tensor) -> torch.Tensor:
        h = self.trunk(tokens, pad_mask)
        dh = _gather_decision_state(h, decision_pos)
        return self.head(dh)

    def forward(self,
                tokens: torch.Tensor,
                pad_mask: torch.Tensor,
                decision_pos: torch.Tensor,
                legal_mask: torch.Tensor,     # (B, NUM_ACTIONS) 1=legal
                ) -> torch.Tensor:
        """Return masked-softmax strategy. Illegal slots are exactly 0."""
        logits = self.forward_logits(tokens, pad_mask, decision_pos)
        # Set illegal slots to the dtype's most-negative finite value so softmax
        # masks them. A literal -1e9 overflows fp16 (Half, max |x| ≈ 65504) if
        # this ever runs under autocast; finfo(dtype).min is representable in any
        # float dtype by construction.
        masked = logits.masked_fill(legal_mask < 0.5,
                                    torch.finfo(logits.dtype).min)
        return F.softmax(masked, dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
