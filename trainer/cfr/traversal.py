"""Shared constants for Deep CFR traversal.

This module used to contain the synchronous obs.x-based traversal
(`traverse`, `run_traversals`, `TraversalNets`). Those were removed when
the trainer migrated to token-based encoding — `cfr.cfr_coro.traverse_coro`
(single-process, coroutine-batched GPU inference) and
`cfr.cfr_mp_gpu._traverse_collect` (multi-process actors) are now the
two live traversal implementations.

What remains here is just the two constants both trainers share.
"""
from __future__ import annotations


# Default depth cap. Matches the engine's HIST_MAX (encoder truncates beyond
# this anyway). Override per-run via the trainer CLI knob `--max-depth`.
DEFAULT_MAX_DEPTH = 34


# Regret-target scaling. Terminal HU NLHE payoffs are in BB units (±100 BB on
# shove paths), so raw regret targets r(s,a) = v(s,a) - V(s) reach magnitudes
# of order 100. With those raw targets the AdvNet's MSE loss sits at ~1000
# and gradient norms regularly hit 1500-5000 — the grad clip at 10 fires
# every step, biasing each Adam update's direction.
#
# We divide stored regrets by REGRET_SCALE so the network learns 1/100-scale
# values; loss drops by 10^4, grads drop by 10^4, and the clip stops firing.
# Regret matching (`regret_matching_np`) is invariant under positive rescaling
# of its input, so the σ derived from the AdvNet is unchanged. Only the
# depth-cap value bootstrap needs to multiply by REGRET_SCALE to reconvert
# the scaled prediction to chip units before mixing with terminal payoffs.
REGRET_SCALE = 100.0
