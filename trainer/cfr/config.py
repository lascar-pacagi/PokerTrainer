"""Hyperparameters for Deep CFR training (transformer AdvNet/PolicyNet).

Defaults are tuned for a laptop smoke run (small T, small buffer, small net).
The cluster-scale defaults live in scripts/run_cfr_cluster.sh.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .tokenize import VOCAB_SIZE, MAX_SEQ_LEN


@dataclass
class CFRModelConfig:
    """Transformer architecture for AdvNet and PolicyNet.

    Both nets share a token embedding + transformer trunk; only the head
    differs (AdvNet outputs raw regrets, PolicyNet outputs masked-softmax
    probabilities).

    Replaces the previous resmlp_v1 / mlp_v1 flat-vector architecture. See
    project_token_encoding for the design discussion. Checkpoint format
    changes (no backward compat with the 816-dim x version).
    """
    # Vocabulary and sequence shape come from the tokenizer to keep them
    # in lockstep. Bumping vocab/seq requires changes in cfr/tokenize.py.
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = MAX_SEQ_LEN

    num_actions: int = 11           # = engine NUM_ACTIONS

    # Transformer hyperparameters.
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4                # d_model must be divisible by n_heads
    d_ff: int = 512                 # FFN inner dim; typically 4 * d_model
    dropout: float = 0.0            # Deep CFR uses MSE on noisy regret
                                    # targets; dropout adds yet more noise.
                                    # Leave 0 unless evidence suggests need.


@dataclass
class CFROptimConfig:
    lr: float = 1e-3                # Adam tends to converge faster than RMSProp on regret-MSE
    weight_decay: float = 1e-3      # mild L2 helps with refit-from-scratch stability
    grad_clip: float = 10.0
    # When refitting AdvNet at each iteration, do `n_grad_steps_per_refit` grad
    # steps with `batch_size` samples per step.
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


# Engine chip conventions (must match engine/src/game_hu.h):
#   1 bb = 100 chips; default starting stack = 100 bb = 10_000 chips.
BIG_BLIND_CHIPS = 100


@dataclass
class CFRStageConfig:
    """Curriculum stage: a restriction of the full game the network solves.

    The network architecture is identical across stages — a "stage" only
    changes (a) how deep the starting stacks are and (b) which engine action
    types the traversal is allowed to branch on. Because nothing about the
    weight shapes depends on these, a converged checkpoint from an easier
    stage is carried forward to the next stage with a plain `load_state_dict`
    (no net2net widening, no distillation). See project_cfr_token_architecture
    for why the token-transformer makes this free.

    Defaults reproduce the FULL game (100 bb, every action legal), so runs
    that don't opt into a stage are unchanged.

    `allowed_actions` is a tuple of engine ActionType integers (see
    engine/src/action.h: FOLD=0, CHECK_CALL=1, RAISE_25=2 .. RAISE_300=9,
    ALL_IN=10). `None` means "no restriction — use whatever the engine
    reports legal". Stage 1 (push/fold) sets this to (0, 1, 10).

    Stored as a tuple (not a set) so it survives pickling into the checkpoint
    with a stable, log-friendly repr.
    """
    starting_stack_chips: int = BIG_BLIND_CHIPS * 100   # 100 bb (full game)
    allowed_actions: tuple[int, ...] | None = None

    @property
    def starting_stack_bb(self) -> float:
        return self.starting_stack_chips / BIG_BLIND_CHIPS

    @property
    def restricts_actions(self) -> bool:
        return self.allowed_actions is not None


@dataclass
class CFRConfig:
    model:  CFRModelConfig  = field(default_factory=CFRModelConfig)
    optim:  CFROptimConfig  = field(default_factory=CFROptimConfig)
    buffer: CFRBufferConfig = field(default_factory=CFRBufferConfig)
    train:  CFRTrainConfig  = field(default_factory=CFRTrainConfig)
    run:    CFRRunConfig    = field(default_factory=CFRRunConfig)
    stage:  CFRStageConfig  = field(default_factory=CFRStageConfig)
