"""Training hyperparameters.

All fields are intentionally overridable via CLI flags in dmc.py so experiments
don't require touching this file. Defaults are a reasonable "works on a laptop"
setting; actual training runs will use the profiles in docs/TRAINING_RECIPE.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Flat-sequence encoding (v0.3 — see docs/STATE_ENCODING.md).
    # x carries chip-state scalars + an 11-bit legal-actions mask + the flat
    # action history (each row's bit 0 = is_real). a-rows are pure 11-dim
    # one-hots — no scalar features.
    x_dim: int = 732         # must match engine X_DIM
    a_dim: int = 11          # must match engine A_DIM
    mlp_hidden: int = 512
    mlp_layers: int = 5
    # arch:
    #   "mlp_v1"     — Linear+ReLU stack, no normalization (legacy default).
    #   "resmlp_v1"  — pre-LN residual blocks (LN→Lin→GELU→Lin + skip),
    #                  GELU activation, deeper, more stable.
    arch: str = "mlp_v1"
    # FFN expansion ratio inside resmlp_v1 blocks (transformer convention is 4).
    # Ignored for mlp_v1.
    mlp_expansion: int = 4


@dataclass
class OptimConfig:
    lr: float = 1e-4
    momentum: float = 0.0
    eps: float = 1e-5        # DouZero RMSProp eps
    grad_clip: float = 40.0  # global L2 clip (DouZero default)


@dataclass
class ActorConfig:
    n_actors: int = 1                 # start single-process; ramp up in Phase 1.2
    rollout_batch: int = 32           # Envs per actor (single-process: total Envs)
    # Linear epsilon schedule for rollout exploration. DouZero uses a fixed
    # 0.01 on card games where MC-return exploration collapse is rare, but HU
    # NLHE with pure-MC returns exposes a pathological attractor where the net
    # commits to shoving preflop after a few lucky wins (the 600-step smoke run
    # on 2026-04-24 converged to 100% AI preflop; see eval telemetry). We warm
    # up at `epsilon_start`, decay linearly to `epsilon_end` over the first
    # `epsilon_decay_steps` learner grad-steps.
    epsilon_start:       float = 0.15
    epsilon_end:         float = 0.02
    epsilon_decay_steps: int   = 20_000
    # Random seed stream — each Env derives a child seed from this + actor/slot.
    base_seed: int = 12345


@dataclass
class LearnerConfig:
    batch_size: int = 256
    unroll_length: int = 8            # unused for pure MC returns; reserved for future
    max_grad_steps: int = 10_000
    checkpoint_every_steps: int = 500
    log_every_steps: int = 50


@dataclass
class RunConfig:
    ckpt_dir: str = "runs/latest"
    device: str = "cpu"               # "cuda:0" for GPU
    seed: int = 42


@dataclass
class DMCConfig:
    model:   ModelConfig   = field(default_factory=ModelConfig)
    optim:   OptimConfig   = field(default_factory=OptimConfig)
    actor:   ActorConfig   = field(default_factory=ActorConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    run:     RunConfig     = field(default_factory=RunConfig)
