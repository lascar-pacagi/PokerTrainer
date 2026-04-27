"""Export a DMCNet `.ckpt` to TorchScript `.pt` for C++ LibTorch inference.

The scripted module has a single `forward(x, a) -> v` method. The caller
(Python harness OR the C++ inference wrapper) is responsible for broadcasting
a single encoded state's `x` across the `(n_legal, A_DIM)` action rows, and
for argmax-over-legal. That keeps the TorchScript surface minimal and lets the
same `.pt` file serve both paths.

Usage:
    PYTHONPATH=trainer python -m export \\
        --ckpt runs/cpu_long_50k/weights_final_00050000.ckpt \\
        --out  runs/cpu_long_50k/model.pt

The script re-runs the eager and scripted networks on a random batch and
asserts allclose — if this fails, the produced `.pt` is rejected, because the
C++ side would silently pick different argmax actions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.serialization as _ts

from dmc.config import (
    ActorConfig,
    DMCConfig,
    LearnerConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
)
from dmc.models import DMCNet


_ts.add_safe_globals([
    DMCConfig, ModelConfig, OptimConfig,
    ActorConfig, LearnerConfig, RunConfig,
])


def _load_cfg(blob: dict) -> ModelConfig:
    """Pull ModelConfig from a ckpt blob, falling back to defaults."""
    cfg = blob.get("cfg")
    if cfg is not None and hasattr(cfg, "model"):
        mcfg = cfg.model
        if isinstance(mcfg, ModelConfig):
            return mcfg
        try:
            return ModelConfig(
                x_dim=int(mcfg.x_dim),
                a_dim=int(mcfg.a_dim),
                mlp_hidden=int(mcfg.mlp_hidden),
                mlp_layers=int(mcfg.mlp_layers),
            )
        except Exception:
            pass
    return DMCConfig().model


def export(ckpt_path: Path, out_path: Path, device: str = "cpu",
           parity_batch: int = 7, parity_atol: float = 1e-6) -> None:
    blob = torch.load(ckpt_path, map_location=device)
    mcfg = _load_cfg(blob)
    print(f"[export] model cfg: x_dim={mcfg.x_dim} a_dim={mcfg.a_dim} "
          f"hidden={mcfg.mlp_hidden} layers={mcfg.mlp_layers}")

    net = DMCNet(mcfg).to(device)
    net.load_state_dict(blob["model"])
    net.train(False)

    scripted = torch.jit.script(net)
    scripted.train(False)

    torch.manual_seed(0)
    x = torch.randn(parity_batch, mcfg.x_dim, device=device)
    a = torch.randn(parity_batch, mcfg.a_dim, device=device)
    with torch.no_grad():
        v_eager = net(x, a)
        v_script = scripted(x, a)
    max_abs = (v_eager - v_script).abs().max().item()
    print(f"[export] parity max|Δ|={max_abs:.3e} over {parity_batch} rows")
    if max_abs > parity_atol:
        raise RuntimeError(
            f"parity check failed: max|Δ|={max_abs} > atol={parity_atol}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))
    step = blob.get("step", "?")
    size_mb = out_path.stat().st_size / 1e6
    print(f"[export] wrote {out_path}  ({size_mb:.2f} MB)  step={step}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out",  type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    export(Path(args.ckpt), Path(args.out), device=args.device)


if __name__ == "__main__":
    main()
