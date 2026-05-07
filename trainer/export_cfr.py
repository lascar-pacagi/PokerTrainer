"""Export a CFR PolicyNet `.ckpt` to TorchScript `.pt` for the Flutter inspector.

The Flutter UI / C++ inference path expects a TorchScript module with signature

    forward(x: Tensor[B, X_DIM], a: Tensor[B, A_DIM]) -> Tensor[B]

where the C++ side broadcasts a single state's `x` across `(n_legal, X_DIM)`
rows and feeds one row of `a` per legal action (one-hot ActionType). The
output `v[i]` is interpreted as "value of legal action i".

CFR PolicyNet has a *different* native signature:

    forward(x: Tensor[B, X_DIM], legal_mask: Tensor[B, NUM_ACTIONS])
        -> probs: Tensor[B, NUM_ACTIONS]

To deploy the CFR policy through the existing inspector we wrap it in a
shim that mimics the DMC signature. The shim:

    1. Reads `x[0:1]`  (all rows are identical because the C++ side broadcasts
       a single state across legal actions).
    2. Derives `legal_mask = a.sum(dim=0, keepdim=True)` — `a` rows are mutually
       exclusive one-hots over ActionType, so column-wise sum is exactly the
       NUM_ACTIONS legal-action mask.
    3. Calls `policy_net(x_one, mask)` to get a (1, NUM_ACTIONS) probability
       distribution (illegal slots = 0, legal slots sum to 1).
    4. Returns per-row probabilities by gathering at each row's `argmax(a)`
       column — i.e. the probability of the action that row represents.

The Flutter inspector renders these per-legal-action values as bars; it
doesn't need to know they're probabilities vs Q-values. (The UI does
auto-detect probability mode by checking that the values sum to ~1 and live
in [0, 1], so it can format the labels as percentages and use absolute
scaling — but that's purely cosmetic.)

Usage:
    PYTHONPATH=trainer python -m export_cfr \\
        --ckpt runs/cfr_laptop_1h/cfr_final.ckpt \\
        --out  runs/cfr_laptop_1h/cfr_final.policy.pt

The `.policy.pt` extension is a convention — the file is structurally a
normal TorchScript .pt and the inspector loads it via the same dialog.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.serialization as _ts
from torch import nn

from cfr.config import (
    CFRBufferConfig,
    CFRConfig,
    CFRModelConfig,
    CFROptimConfig,
    CFRRunConfig,
    CFRTrainConfig,
)
from cfr.models import PolicyNet


_ts.add_safe_globals([
    CFRConfig, CFRModelConfig, CFROptimConfig,
    CFRBufferConfig, CFRTrainConfig, CFRRunConfig,
])


class _CFRPolicyShim(nn.Module):
    """TorchScript-friendly DMC-signature wrapper around a PolicyNet.

    Inputs match the DMC inference path; output is per-legal-action probability.
    """

    def __init__(self, policy_net: PolicyNet, num_actions: int):
        super().__init__()
        self.policy_net = policy_net
        self.num_actions = num_actions

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # x: (n_legal, X_DIM) — all rows identical (broadcast).
        # a: (n_legal, NUM_ACTIONS) — one-hot rows for each legal ActionType.
        x_one = x[0:1]                                     # (1, X_DIM)
        legal_mask = a.sum(dim=0, keepdim=True)            # (1, NUM_ACTIONS)
        # Defensive clamp: legal_mask should already be 0/1 floats; clamp keeps
        # PolicyNet's masked-softmax happy if any drift somehow snuck in.
        legal_mask = legal_mask.clamp(min=0.0, max=1.0)
        probs = self.policy_net(x_one, legal_mask)         # (1, NUM_ACTIONS)
        # For each row, gather the probability at that row's ActionType.
        action_idx = a.argmax(dim=1)                       # (n_legal,)
        # probs[0, action_idx] selects (n_legal,) values.
        return probs[0].index_select(0, action_idx)


def _load_cfg(blob: dict) -> CFRModelConfig:
    cfg = blob.get("cfg")
    if cfg is not None and hasattr(cfg, "model"):
        mcfg = cfg.model
        if isinstance(mcfg, CFRModelConfig):
            return mcfg
        # Reconstruct from a duck-typed object (defensive against ckpt drift).
        try:
            return CFRModelConfig(
                x_dim=int(mcfg.x_dim),
                num_actions=int(mcfg.num_actions),
                hidden=int(mcfg.hidden),
                n_layers=int(mcfg.n_layers),
                arch=str(getattr(mcfg, "arch", "resmlp_v1")),
                mlp_expansion=int(getattr(mcfg, "mlp_expansion", 4)),
            )
        except Exception:
            pass
    return CFRConfig().model


def export(ckpt_path: Path, out_path: Path, device: str = "cpu",
           parity_atol: float = 1e-6) -> None:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = _load_cfg(blob)
    print(f"[export_cfr] model cfg: x_dim={mcfg.x_dim} num_actions={mcfg.num_actions} "
          f"hidden={mcfg.hidden} layers={mcfg.n_layers} arch={mcfg.arch}")

    pol = PolicyNet(mcfg).to(device)
    pol.load_state_dict(blob["policy_net"])
    pol.train(False)

    shim = _CFRPolicyShim(pol, num_actions=mcfg.num_actions).to(device)
    shim.train(False)

    scripted = torch.jit.script(shim)
    scripted.train(False)

    # Parity check: synthesize a (n_legal, X_DIM) broadcast x and a one-hot
    # legal mask matching some N legal slots. Compare eager vs scripted.
    torch.manual_seed(0)
    n_legal = 5
    legal_idx = torch.tensor([0, 1, 5, 9, 10])           # FOLD, CHECK_CALL, R75, R300, AI
    x = torch.randn(1, mcfg.x_dim, device=device).expand(n_legal, mcfg.x_dim).contiguous()
    a = torch.zeros(n_legal, mcfg.num_actions, device=device)
    for row, k in enumerate(legal_idx.tolist()):
        a[row, k] = 1.0
    with torch.no_grad():
        v_eager  = shim(x, a)
        v_script = scripted(x, a)
    max_abs = (v_eager - v_script).abs().max().item()
    print(f"[export_cfr] parity max|Δ|={max_abs:.3e} over n_legal={n_legal}")
    if max_abs > parity_atol:
        raise RuntimeError(
            f"parity check failed: max|Δ|={max_abs} > atol={parity_atol}"
        )

    # Sanity: probabilities sum to ~1 across legal rows (this is what the UI
    # auto-detect uses to switch to π-mode).
    psum = float(v_eager.sum().item())
    print(f"[export_cfr] sample probs: sum={psum:.4f}  values={v_eager.tolist()}")
    if not (0.95 <= psum <= 1.05):
        raise RuntimeError(
            f"probabilities don't sum to ~1 (sum={psum}). The shim is wrong, "
            f"or the PolicyNet weights aren't the trained ones."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))
    iter_n = blob.get("iteration", "?")
    size_mb = out_path.stat().st_size / 1e6
    print(f"[export_cfr] wrote {out_path}  ({size_mb:.2f} MB)  iteration={iter_n}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out",  type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    export(Path(args.ckpt), Path(args.out), device=args.device)


if __name__ == "__main__":
    main()
