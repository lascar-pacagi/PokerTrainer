"""Hyperparameters for Deep CFR training.

Defaults are tuned for a laptop smoke run (small T, small buffer, small net).
The cluster-scale defaults live in scripts/run_cfr_cluster.sh.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CFRModelConfig:
    """Both AdvNet and PolicyNet share this trunk shape — only the head differs.

    AdvNet head:    Linear(hidden → NUM_ACTIONS) — predicts cumulative regret per action.
    PolicyNet head: Linear(hidden → NUM_ACTIONS) → softmax over legal mask — average strategy.
    """
    x_dim: int = 816                # must match engine X_DIM (encoding v0.5)
    num_actions: int = 11           # = engine NUM_ACTIONS
    # Laptop defaults are intentionally small: traversal does ~5k single-state
    # forwards per hand; the larger the net, the longer each forward takes.
    # For cluster runs, override via CLI: --hidden 512 --n-layers 8.
    hidden: int = 128
    n_layers: int = 2
    arch: str = "resmlp_v1"         # "mlp_v1" or "resmlp_v1"
    mlp_expansion: int = 4          # FFN width in resmlp blocks


@dataclass
class CFROptimConfig:
    lr: float = 1e-3                # Adam tends to converge faster than RMSProp on regret-MSE
    weight_decay: float = 1e-3      # mild L2 helps with refit-from-scratch stability
    grad_clip: float = 10.0
    # When refitting AdvNet at each iteration, do `epochs_per_refit` passes
    # over the buffer with `batch_size` samples per gradient step.
    batch_size: int = 1024
    n_grad_steps_per_refit: int = 4_000


@dataclass
class CFRBufferConfig:
    # Reservoir capacities. The paper uses ~40M for advantage and ~40M for
    # strategy on no-limit poker; we use 1M as a laptop default. Cluster
    # config bumps both.
    adv_capacity: int = 1_000_000
    policy_capacity: int = 1_000_000


@dataclass
class CFRTrainConfig:
    # T = number of CFR iterations. Each iteration:
    #   1. K traversals with player p=SB as traverser → add to adv_buf[SB] + policy_buf
    #   2. K traversals with player p=BB as traverser → add to adv_buf[BB] + policy_buf
    #   3. Refit adv_net[SB] from scratch on adv_buf[SB]; same for BB.
    # After T iterations, train policy_net on policy_buf.
    n_iterations: int = 100
    n_traversals_per_iter: int = 1_000
    # Also include the linear weighting (each entry's loss weight = its iteration t).
    use_linear_cfr: bool = True
    # If True, refit AdvNet from scratch at each iteration (paper default).
    # If False, warm-start from previous AdvNet weights (faster but biased).
    reset_adv_net_each_iter: bool = True
    # Final policy net training pass (after all CFR iterations).
    policy_n_grad_steps: int = 50_000
    policy_batch_size: int = 1024


@dataclass
class CFRRunConfig:
    ckpt_dir: str = "runs/cfr_latest"
    device: str = "cpu"             # "cuda:0" for GPU
    seed: int = 42


@dataclass
class CFRConfig:
    model:  CFRModelConfig  = field(default_factory=CFRModelConfig)
    optim:  CFROptimConfig  = field(default_factory=CFROptimConfig)
    buffer: CFRBufferConfig = field(default_factory=CFRBufferConfig)
    train:  CFRTrainConfig  = field(default_factory=CFRTrainConfig)
    run:    CFRRunConfig    = field(default_factory=CFRRunConfig)
