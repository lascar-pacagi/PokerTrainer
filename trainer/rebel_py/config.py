"""ReBeL training configuration (Phase 2a: river, betting-depth limit)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RebelConfig:
    # ─── game / abstraction ──────────────────────────────────────────────
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)  # one river board per seed
    stack_bb: int = 12
    actions: tuple[int, ...] = (0, 1, 10)              # FOLD, CHECK_CALL, ALL_IN
    depth_limit: int = 2                               # betting actions → net leaf

    # ─── subgame solver ──────────────────────────────────────────────────
    # CFR iterations per subgame solve. This also sets the *accuracy of the
    # value-net targets*: the target is the time-averaged root value, whose
    # O(1/T) bias (the opponent not yet sharpened in the average) is learned by
    # the net and shows up as residual exploitability. Too few iters here caps
    # how low the trained net can drive exploitability, independent of net size
    # or epochs — keep it generous.
    cfr_iters: int = 256
    random_action_prob: float = 0.25

    # ─── value net ───────────────────────────────────────────────────────
    n_hidden: int = 512
    n_layers: int = 3
    use_layer_norm: bool = True
    lr: float = 3e-4
    device: str = "cpu"

    # ─── training loop ───────────────────────────────────────────────────
    buffer_cap: int = 200_000
    epochs: int = 60
    games_per_epoch: int = 48
    batches_per_epoch: int = 64
    batch_size: int = 512
    eval_every: int = 5
    eval_iters: int = 256          # CFR iters for the exploitability gate
    seed: int = 0

    # ─── checkpointing ───────────────────────────────────────────────────
    ckpt_path: str = ""            # "" → no checkpoint written

    def __post_init__(self):
        if isinstance(self.seeds, list):
            self.seeds = tuple(self.seeds)
        if isinstance(self.actions, list):
            self.actions = tuple(self.actions)
