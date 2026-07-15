"""Single-process Deep CFR trainer with coroutine-based batched GPU inference.

═══════════════════════════════════════════════════════════════════════════════
WHY THIS FILE EXISTS
═══════════════════════════════════════════════════════════════════════════════

`cfr_mp_gpu.py` (the multi-process trainer) uses CPU inference inside actor
processes. For a small AdvNet (~370k params), CPU forward beats GPU forward
because GPU kernel-launch overhead (~250µs) dominates the actual compute (~5µs).

But at cluster scale (HIDDEN=512, LAYERS=8 → 17M params), CPU forward grows
to ~5ms per call. With ~5000 inference calls per traversal, that's 25 seconds
per traversal — too slow even with many actors.

GPU forward stays fast at ~17M params, BUT only if you batch. A single B=1 GPU
forward is still launch-bound. The trick: run many concurrent traversals in
ONE process and batch their inference calls.

This file implements that pattern using Python generators-as-coroutines.

═══════════════════════════════════════════════════════════════════════════════
COROUTINE PRIMER (for the unfamiliar)
═══════════════════════════════════════════════════════════════════════════════

A Python "generator" is a function that contains `yield`. Calling it returns
a generator OBJECT — the function body doesn't run yet:

    def counter():
        x = 0
        while True:
            received = yield x   # YIELDS x, RECEIVES the .send() value
            x += received

    g = counter()           # generator object, not started
    print(g.send(None))     # → 0   (must prime with None on first send)
    print(g.send(10))       # → 10  (received=10, x became 10)
    print(g.send(5))        # → 15

Each `.send(v)` resumes the generator at its last `yield`, makes the yield
expression evaluate to `v`, runs until the NEXT yield, and returns whatever
that yield expression provides. This bidirectional channel — a generator
yields values OUT, and receives values IN via .send() — is the coroutine
pattern.

When the generator's function body returns (or raises StopIteration), .send()
raises `StopIteration` whose `.value` attribute holds the return value.

`yield from` delegates: when `parent` says `yield from child`, the parent's
generator becomes a passthrough for the child. Values yielded by child go
out to the caller; values .send()-ed by the caller go into child. When child
returns, the return value becomes the value of the `yield from` expression
in parent. This is exactly what we want for recursive traversal — the
recursion is implemented with `yield from` and the driver only sees the
leaf-level yields.

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE (single process)
═══════════════════════════════════════════════════════════════════════════════

   ┌── Main process ────────────────────────────────────────────────────────┐
   │                                                                        │
   │   N "virtual actors" = N generator objects (coroutines)                │
   │      each holds its own engine state and yields (actor, x) when        │
   │      it needs an inference                                             │
   │                                                                        │
   │           ↓ collect pending yields                                     │
   │                                                                        │
   │   Driver loop:                                                         │
   │     1. Group pending requests by actor (SB vs BB)                      │
   │     2. Stack each group into a batch tensor on GPU                     │
   │     3. Run ONE forward per group (B = number of pending requests)      │
   │     4. Split results, .send() each back to its coroutine               │
   │     5. When a coroutine returns, drain its writes to buffers,          │
   │        start a new traversal in that slot                              │
   │                                                                        │
   │           ↓ writes accumulate in adv_buf[0/1], pol_buf                 │
   │                                                                        │
   │   Learner refit + final policy training in same process                │
   │                                                                        │
   └────────────────────────────────────────────────────────────────────────┘

KEY PROPERTIES:
  + No IPC, no serialization. All in one address space.
  + Inference batched naturally — N coroutines = batch size N.
  + GPU-resident nets the whole time. No CPU sync needed.
  + Easy to debug: drop into pdb in the recursion, examine state.
  - One Python interpreter. No CPU parallelism beyond what NumPy/PyTorch
    release the GIL for. Fine here because the bottleneck is GPU forward
    + recursion bookkeeping (both fast).

LATENCY MATH (for HIDDEN=512, LAYERS=8 → 17M params, single A100/H100):
  * Per-inference: B=32 batched forward ≈ 200µs (vs 250µs for B=1).
    Per coroutine, an inference takes the same wall time regardless of B.
    But B coroutines all advance one step in the same 200µs.
  * Per traversal: ~5000 yields × 200µs / N parallel coros = 1000ms / N.
    With N=32 coros, ~31ms per traversal of throughput.
  * K=5000 traversals → 5000 × 31ms = 156 seconds per iteration. ~3 min.

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import torch

import pokertrainer_engine as pte

from .buffers import ReservoirBuffer, save_buffers, load_buffers
from .config import CFRConfig, BIG_BLIND_CHIPS
from .models import AdvNet, PolicyNet, count_parameters
from .regret_matching import regret_matching_np
from .tokenize import TokenizedState, tokenize_state, pad_batch
from .traversal import DEFAULT_MAX_DEPTH, REGRET_SCALE
from .train import (refit_adv_net, save_checkpoint,
                    find_latest_checkpoint, load_checkpoint)
from .probe import run_default_probes, format_probe_line


NUM_ACTIONS = 11


# ═══════════════════════════════════════════════════════════════════════════
# THE COROUTINE: traversal as a generator
# ═══════════════════════════════════════════════════════════════════════════
#
# Type signature (informal):
#   traverse_coro(...) -> Generator[
#       (actor: int, x: np.ndarray),    # YIELDED: inference request
#       np.ndarray,                     # SENT IN: regret prediction
#       float,                          # RETURNED: traverser's expected utility
#   ]
#
# Reading the body:
#   `pred_r = (yield (actor, x))`
#       Pause here, give (actor, x) to the driver, resume when driver calls
#       .send(prediction). `pred_r` is bound to the sent value.
#
#   `v_a = yield from traverse_coro(child, ...)`
#       Delegate to a child generator. Yields and sends pass through
#       transparently. When child returns, `v_a` gets its return value.
# ═══════════════════════════════════════════════════════════════════════════


def _legal_mask_from_obs(obs) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for at in obs.legal:
        mask[int(at)] = 1.0
    return mask


def traverse_coro(env,
                  traverser: int,
                  rng: np.random.Generator,
                  adv_writes: list,
                  pol_writes: list,
                  max_depth: int = DEFAULT_MAX_DEPTH,
                  allowed: Optional[frozenset] = None,
                  regret_scale: float = REGRET_SCALE,
                  max_raises_per_street: Optional[int] = None):
    """Generator-coroutine implementing one external-sampling MCCFR traversal.

    YIELDS:
        (actor: int, tok_state: TokenizedState) — inference request. The
        TokenizedState carries the token sequence and the index of the
        [DECISION] token; the driver batches these together with padding.
    EXPECTS via .send():
        np.ndarray of shape (NUM_ACTIONS,) — predicted regrets for the
        AdvNet of player `actor`.
    RETURNS:
        float — the traverser's expected utility at the root (chip units).

    Token-based migration (2026-05-20): yields TokenizedState instead of
    a flat x ndarray. The downstream batcher pads sequences to max-in-batch
    and runs the transformer over the result. Adv/pol writes store the
    TokenizedState along with the target so the buffer can replay
    variable-length sequences at refit time.
    """
    if env.is_terminal():
        return float(env.payoffs_bb()[traverser])

    obs   = env.observation()
    actor = int(env.to_act())
    legal = list(obs.legal)
    if allowed is not None:
        # Curriculum stage restriction: drop any action the stage forbids.
        # CHECK_CALL (and ALL_IN) are always legal in HU NLHE, so the filtered
        # set is never empty. Suppressed slots simply never get branched or
        # sampled, and never receive a regret target → their AdvNet head row
        # stays at its zero-init baseline (neutral when a later stage unlocks
        # the action). See CFRStageConfig.
        legal = [at for at in legal if int(at) in allowed]
    # Re-raise cap (action ABSTRACTION, training-only): once `max_raises_per_street`
    # voluntary aggressive actions (RAISE_* / ALL_IN, type >= 2) have happened on
    # the CURRENT street, drop all raises so only FOLD / CHECK_CALL remain. This
    # bounds 100bb re-raise wars — the dominant source of full-depth tree blowup
    # once the depth cap is removed — at modest abstraction cost (most solvers and
    # Pluribus cap raises too). It is NOT an engine rule: play/eval/the Slumbot
    # mirror still allow arbitrary raises. None = uncapped (legacy behavior).
    if max_raises_per_street is not None:
        st = env.state()
        cur_street = int(st.street)
        raises = sum(1 for h in st.history
                     if int(h.street) == cur_street and int(h.type) >= 2)
        if raises >= max_raises_per_street:
            legal = [at for at in legal if int(at) <= 1]   # keep FOLD, CHECK_CALL
    n_legal = len(legal)
    # Build the mask from the (possibly filtered) legal set so regret matching
    # can put no mass on a forbidden slot.
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for at in legal:
        mask[int(at)] = 1.0
    # Tokenize from the to-act player's perspective. The tokenizer pulls
    # state.history, state.hole, state.board, state.starting_stacks, so the
    # cost is dominated by the action-history walk (typically <30 entries).
    tok_state = tokenize_state(env.state(), hero_seat=actor)

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  THE YIELD POINT — request an inference, suspend until satisfied.   ║
    # ║                                                                      ║
    # ║  Driver collects this yielded (actor, tok_state), batches with other║
    # ║  coros' yields (padding to max-in-batch), runs one GPU forward, and ║
    # ║  .send()s the per-row result here.                                   ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    pred_r = (yield (actor, tok_state))    # type: np.ndarray
    sigma = regret_matching_np(pred_r, mask)

    # Strategy memory: σ is stored ONLY at opponent nodes (Brown 2019, Alg. 1).
    # The opponent samples its own actions from σ, so its nodes are visited
    # proportionally to its own reach probability — the weighting the average
    # strategy needs. Traverser nodes are reached by branching every action
    # (reach-unweighted); storing σ there biases the PolicyNet targets.
    if actor != traverser:
        pol_writes.append((tok_state, sigma))

    # Depth cap: substitute AdvNet-σ-weighted bootstrap for further recursion.
    # KNOWN DEVIATION from Brown 2019 (paper traverses to terminal): σ-weighted
    # predicted REGRETS are not a state value (regrets are relative to v(σ); a
    # converged net gives ≈0 here), so this bootstrap ≈ "deep tails are 0 EV".
    # Rarely hit at max_depth=34; never at short stacks. A proper fix would be
    # a value head or removing the cap.
    # if env.state().history_size >= max_depth:
    #    return float((sigma * pred_r * mask).sum()) * regret_scale
    if actor == traverser:
        # ── TRAVERSER NODE: branch every legal action ───────────────────────
        action_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for at in legal:
            child = env.clone()
            child.step_action(at)
            v_a = yield from traverse_coro(child, traverser, rng,
                                           adv_writes, pol_writes,
                                           max_depth=max_depth, allowed=allowed,
                                           regret_scale=regret_scale,
                                           max_raises_per_street=max_raises_per_street)
            action_values[int(at)] = v_a
        v_state = float((sigma * action_values).sum())
        regrets = (action_values - v_state) * mask / regret_scale
        adv_writes.append((tok_state, regrets.astype(np.float32, copy=False)))
        return v_state

    # ── OPPONENT NODE: external sampling ────────────────────────────────────
    legal_int = np.array([int(at) for at in legal], dtype=np.int64)
    legal_probs = sigma[legal_int]
    z = legal_probs.sum()
    if z > 0.0:
        legal_probs = legal_probs / z
    else:
        legal_probs = np.full(n_legal, 1.0 / n_legal, dtype=np.float32)
    chosen_local_idx = int(rng.choice(n_legal, p=legal_probs))
    # Apply by ActionType, NOT by env.step(index): env.step() interprets the
    # index against the ENGINE's unfiltered legal list, so with a curriculum
    # filter the same index means a different action (e.g. filtered [F,C,AI]
    # idx 2 → engine legal[2] = RAISE_25, a silent wrong move).
    env.step_action(legal[chosen_local_idx])
    return (yield from traverse_coro(env, traverser, rng,
                                     adv_writes, pol_writes,
                                     max_depth=max_depth, allowed=allowed,
                                     regret_scale=regret_scale,
                                     max_raises_per_street=max_raises_per_street))


# ═══════════════════════════════════════════════════════════════════════════
# DRIVER: schedule N coroutines, batch their yields, run on GPU
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _Slot:
    """Per-virtual-actor mutable state.

    A `_Slot` is recycled across many traversals: when a coroutine completes,
    we drain its writes and start a fresh coroutine in the same slot. This
    keeps the GPU batch full at constant size N rather than tapering as
    coroutines finish.
    """
    env: pte.Env
    coro: Optional[Generator] = None
    # The next pending request from this coroutine: (actor_player_id,
    # TokenizedState). None means: this slot has just finished a traversal
    # and hasn't been restarted yet (transient state during driver iter).
    pending: Optional[tuple[int, "TokenizedState"]] = None
    traverser: int = 0
    adv_writes: list = field(default_factory=list)
    pol_writes: list = field(default_factory=list)
    # Curriculum action restriction (frozenset of engine ActionType ints, or
    # None for the full game). Persisted on the slot so restarts reuse it.
    allowed: Optional[frozenset] = None
    # Regret-target divisor; should track the effective stack in bb so targets
    # are O(1) regardless of stack depth (= REGRET_SCALE for the 100bb game).
    regret_scale: float = REGRET_SCALE
    # Action-abstraction re-raise cap per street (None = uncapped). Persisted on
    # the slot so restarts reuse it.
    max_raises_per_street: Optional[int] = None


def _start_traversal(slot: _Slot,
                     rng: np.random.Generator) -> None:
    """Begin a fresh traversal in this slot.

    Picks a random traverser (50/50 SB/BB), resets the engine, creates the
    coroutine, and PRIMES it — i.e., calls .send(None) to advance to the
    first yield. Without priming, the coroutine wouldn't actually have
    started executing, and there'd be no pending request to batch.

    Why .send(None) and not .next()? Both work in Python 3 (next(g) is
    equivalent to g.send(None)). Using .send(None) makes the symmetry with
    later .send(prediction) calls more obvious in the code.
    """
    slot.traverser = int(rng.integers(0, 2))
    slot.env.reset()
    slot.adv_writes = []
    slot.pol_writes = []
    slot.coro = traverse_coro(slot.env, slot.traverser, rng,
                              slot.adv_writes, slot.pol_writes,
                              allowed=slot.allowed,
                              regret_scale=slot.regret_scale,
                              max_raises_per_street=slot.max_raises_per_street)
    # Prime: run until first yield. If the traversal is trivial enough to
    # finish without yielding (impossible in practice — every non-terminal
    # state requires inference), we'd catch StopIteration here.
    try:
        slot.pending = slot.coro.send(None)
    except StopIteration:
        # Edge case: the coroutine returned without yielding (e.g., if reset()
        # somehow produced an immediately-terminal state, which shouldn't
        # happen but we're defensive).
        slot.coro = None
        slot.pending = None


def _batch_forward(net,
                   tok_states: list[TokenizedState],
                   device: torch.device) -> list[np.ndarray]:
    """Run ONE GPU forward over a list of TokenizedStates; return per-row regrets.

    Pads variable-length token sequences to the max in-batch length via
    `pad_batch`, then runs the transformer once. The model returns raw
    regrets at the [DECISION] position of each row.
    """
    if not tok_states:
        return []
    tokens_np, pad_mask_np, dpos_np = pad_batch(tok_states)
    tokens   = torch.from_numpy(tokens_np).to(device, non_blocking=True)
    pad_mask = torch.from_numpy(pad_mask_np).to(device, non_blocking=True)
    dpos     = torch.from_numpy(dpos_np).to(device, non_blocking=True)
    with torch.no_grad():
        out = net(tokens, pad_mask, dpos)         # (B, NUM_ACTIONS)
    out_np = out.detach().cpu().numpy().astype(np.float32, copy=False)
    return [out_np[k] for k in range(len(tok_states))]


def _drive_round(slots: list[_Slot],
                 adv_nets_gpu: list[torch.nn.Module],
                 device: torch.device) -> list[int]:
    """Advance every active coroutine by ONE yield-resume cycle.

    Returns the list of slot indices whose coroutines completed in this round
    (so the caller can drain their writes and restart them).

    The "lockstep" property: each round runs one inference per active slot
    (split across two batches: one per AdvNet). This is exactly what makes
    batching efficient — we always have ~N pending requests at any time.

    BATCHING BY ACTOR (not by traverser):
        We split into SB-net batch and BB-net batch based on `actor`, not
        `traverser`. Why? `actor` is who's currently to act in that slot's
        engine state — that determines which AdvNet must be queried. The
        same coroutine may bounce between SB-actor and BB-actor states
        within a single traversal as the betting alternates.
    """
    # Group pending requests by which AdvNet they need.
    sb_idx: list[int] = []
    sb_states: list[TokenizedState] = []
    bb_idx: list[int] = []
    bb_states: list[TokenizedState] = []
    for i, s in enumerate(slots):
        if s.pending is None:
            continue
        actor, tok_state = s.pending
        if actor == 0:
            sb_idx.append(i); sb_states.append(tok_state)
        else:
            bb_idx.append(i); bb_states.append(tok_state)

    # ONE forward per AdvNet. AdvNets don't share weights so we can't fuse.
    sb_preds = _batch_forward(adv_nets_gpu[0], sb_states, device)
    bb_preds = _batch_forward(adv_nets_gpu[1], bb_states, device)

    completed: list[int] = []

    # Dispatch SB results.
    for k, slot_i in enumerate(sb_idx):
        s = slots[slot_i]
        try:
            # .send() resumes the coroutine, replacing the value of its
            # `(yield (actor, x))` expression with the prediction.
            s.pending = s.coro.send(sb_preds[k])
        except StopIteration as e:
            # Coroutine returned. e.value is the traverser's expected utility
            # at the root (we don't actually use it here — we want the
            # accumulated writes — but it's available if you wanted to
            # log per-traversal mean utility).
            s.pending = None
            s.coro = None
            completed.append(slot_i)

    # Dispatch BB results — same pattern.
    for k, slot_i in enumerate(bb_idx):
        s = slots[slot_i]
        try:
            s.pending = s.coro.send(bb_preds[k])
        except StopIteration as e:
            s.pending = None
            s.coro = None
            completed.append(slot_i)

    return completed


def run_K_traversals(K: int,
                     adv_nets_gpu: list[torch.nn.Module],
                     adv_bufs: list[ReservoirBuffer],
                     pol_buf: ReservoirBuffer,
                     n_virtual: int,
                     t: int,
                     rng: np.random.Generator,
                     base_seed: int,
                     device: torch.device,
                     max_depth: int = DEFAULT_MAX_DEPTH,
                     starting_stack_chips: Optional[int] = None,
                     allowed: Optional[frozenset] = None,
                     regret_scale: float = REGRET_SCALE,
                     max_raises_per_street: Optional[int] = None) -> dict:
    """Run K traversals using N_VIRTUAL coroutines in lockstep.

    Each completed traversal contributes its accumulated `adv_writes` to
    the appropriate adv_buf[traverser] and `pol_writes` to pol_buf, then
    we start a new traversal in the freed slot (until n_completed == K).

    Returns a stats dict. Maintains the invariant: at the start of each
    round, every active slot has a pending request OR is being restarted.

    PUTTING NETS INTO EVAL MODE:
        We call `.train(False)` once at entry — this disables Dropout and
        BatchNorm running-mean updates (we don't have either in resmlp_v1
        but it's defensive). We restore train(True) at exit so the caller
        can refit immediately without surprises.
    """
    for net in adv_nets_gpu:
        net.train(False)

    # Per-slot persistent state. Engine instances are created once and
    # reset() between traversals — avoids the cost of re-allocating the
    # engine's internal RNG state every traversal.
    def _make_env(seed: int) -> "pte.Env":
        # Short-stack stages pass an explicit starting stack; the full game
        # leaves it None and lets the engine use its 100 bb default.
        if starting_stack_chips is None:
            return pte.Env(seed)
        return pte.Env(seed, starting_stack_chips)

    slots = [
        _Slot(env=_make_env((base_seed + i * 1009 + t * 1234567) & 0xFFFFFFFFFFFFFFFF),
              allowed=allowed, regret_scale=regret_scale,
              max_raises_per_street=max_raises_per_street)
        for i in range(n_virtual)
    ]
    for s in slots:
        _start_traversal(s, rng)

    n_completed = 0
    n_yields = 0
    t_start = time.time()

    # Outer driver loop: one round = one batched forward per AdvNet, then
    # dispatch results and restart any completed slots.
    while n_completed < K:
        # Advance every active coroutine by one step.
        completed = _drive_round(slots, adv_nets_gpu, device)
        n_yields += sum(1 for s in slots if s.coro is not None or s.pending is not None)

        # Drain completed slots into the buffers, then restart them.
        for slot_i in completed:
            s = slots[slot_i]
            tr = s.traverser
            for tok_state, regrets in s.adv_writes:
                adv_bufs[tr].add(tok_state.tokens, tok_state.decision_pos,
                                 regrets, t)
            for tok_state, sigma in s.pol_writes:
                pol_buf.add(tok_state.tokens, tok_state.decision_pos,
                            sigma, t)
            n_completed += 1
            if n_completed < K:
                _start_traversal(s, rng)
            # else: leave the slot inactive; the driver loop will exit once
            # all K are collected.

        # Defensive: if for some reason every slot becomes inactive but we
        # haven't hit K (shouldn't happen — completed slots get restarted),
        # break to avoid infinite loop.
        if all(s.coro is None for s in slots):
            break

    # Restore train mode for the upcoming refit.
    for net in adv_nets_gpu:
        net.train(True)

    return {
        "n_traversals": n_completed,
        "wall_s": time.time() - t_start,
        "n_yields": n_yields,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI / MAIN
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-virtual", type=int, default=32,
                   help="number of concurrent coroutines = GPU batch size for "
                        "inference. Higher = better GPU utilization but more "
                        "Python overhead per round. Sweet spot is usually 16-64.")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="device for AdvNets, PolicyNet, and inference batches")
    p.add_argument("--n-iterations", type=int, default=100)
    p.add_argument("--n-traversals-per-iter", type=int, default=5000)
    p.add_argument("--adv-grad-steps", type=int, default=4000)
    p.add_argument("--policy-grad-steps", type=int, default=50000,
                   help="DEPRECATED / ignored — average-policy training moved to "
                        "cfr.harvest_policy (--steps there). Accepted so existing "
                        "launchers don't break.")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="Adam L2 on AdvNet/PolicyNet refits. NOTE: weight_decay "
                        "shrinks all weights every step and overwhelms the weak "
                        "hole-card-discrimination signal — a controlled "
                        "diagnostic showed wd=1e-3 freezes the net to a constant "
                        "action while wd=0 learns the correct hand-dependent "
                        "range. Use 0 unless you have a specific reason.")
    p.add_argument("--adv-capacity", type=int, default=2_000_000)
    p.add_argument("--policy-capacity", type=int, default=4_000_000)
    # Transformer hyperparameters (see cfr/config.py:CFRModelConfig).
    p.add_argument("--d-model", type=int, default=128,
                   help="transformer hidden dim. Must be divisible by --n-heads.")
    p.add_argument("--n-layers", type=int, default=4,
                   help="number of transformer encoder layers")
    p.add_argument("--n-heads", type=int, default=4,
                   help="number of attention heads")
    p.add_argument("--d-ff", type=int, default=512,
                   help="FFN inner dim (typically 4 * d_model)")
    p.add_argument("--no-linear-cfr", action="store_true")
    p.add_argument("--ckpt-dir", type=str, default="runs/cfr_coro_latest")
    p.add_argument("--checkpoint-every-iter", type=int, default=5)
    p.add_argument("--no-resume", action="store_true",
                   help="start fresh even if a checkpoint exists in --ckpt-dir "
                        "(default: auto-resume from the latest cfr_iter_*.ckpt).")
    p.add_argument("--resume-from", type=str, default="",
                   help="resume from a specific checkpoint file (overrides the "
                        "auto-scan of --ckpt-dir).")
    p.add_argument("--no-save-buffers", action="store_true",
                   help="do NOT persist the reservoir buffers with checkpoints. "
                        "By default the buffers (the actual training data) are "
                        "written to <ckpt-dir>/cfr_buffers.npz at each checkpoint "
                        "and restored on resume → exact, zero-regression resume. "
                        "They are multi-GB; --checkpoint-every-iter controls the "
                        "write cadence (raise it if buffer I/O dominates).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    p.add_argument("--amp", action="store_true",
                   help="mixed precision (fp16 autocast + GradScaler) for the "
                        "AdvNet refit and PolicyNet training. ~1.5-2× faster on "
                        "the matmul-heavy transformer, stacks with DDP. CUDA only "
                        "(no-op on CPU); fp16 is portable across Turing+Ampere. "
                        "Default off.")
    p.add_argument("--max-raises-per-street", type=int, default=0,
                   help="action-abstraction re-raise cap per street (0 = uncapped). "
                        "Once this many voluntary raises/all-ins have happened on a "
                        "street, only fold/call remain — bounds 100bb re-raise wars "
                        "that blow up full-depth traversal when --max-depth is large. "
                        "Training-only; play/eval still allow arbitrary raises. "
                        "Typical: 3.")
    # ── Curriculum stage knobs (see cfr/config.py:CFRStageConfig) ───────────
    p.add_argument("--starting-stack-bb", type=float, default=100.0,
                   help="Effective starting stack in big blinds. Stage 1 "
                        "(push/fold) uses 10. Default 100 = full game.")
    p.add_argument("--push-fold", action="store_true",
                   help="Stage-1 action restriction: only FOLD / CHECK_CALL / "
                        "ALL_IN are legal in traversal. Combine with "
                        "--starting-stack-bb 10. When set, the training "
                        "validation compares the net's jam/call ranges against "
                        "the computed HU Nash push/fold oracle each probe "
                        "interval (replaces the AA/72o vs-random probe).")
    p.add_argument("--oracle-deals", type=int, default=12_000_000,
                   help="Monte-Carlo deals for the push/fold equity matrix the "
                        "Nash oracle is solved from. Built+cached once; ~80s at "
                        "the default. Only used with --push-fold.")
    p.add_argument("--probe-every-iter", type=int, default=1,
                   help="Run AA / 72o learning probes every N iterations. "
                        "0 disables. Each probe plays --probe-hands against "
                        "uniform-random and reports mbb/hand + preflop action "
                        "mix — a cheap 'is the net learning poker?' signal.")
    p.add_argument("--probe-hands", type=int, default=400,
                   help="Hands per probe scenario (AA, then 72o). 400 gives "
                        "SE ≈ 300 mbb/hand on the AA scenario — fine for "
                        "tracking a several-bb signal across iterations.")
    p.add_argument("--policy-every-iter", type=int, default=0,
                   help="DEPRECATED / ignored — the average-policy harvest is now "
                        "offline (python -m cfr.harvest_policy). Accepted so "
                        "existing launchers don't break.")
    p.add_argument("--policy-probe-hands", type=int, default=400,
                   help="DEPRECATED / ignored — see cfr.harvest_policy.")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CFRConfig()
    cfg.run.device = args.device
    cfg.run.seed = args.seed
    cfg.run.ckpt_dir = args.ckpt_dir
    cfg.train.n_iterations = args.n_iterations
    cfg.train.n_traversals_per_iter = args.n_traversals_per_iter
    cfg.optim.n_grad_steps_per_refit = args.adv_grad_steps
    cfg.train.policy_n_grad_steps = args.policy_grad_steps
    cfg.optim.weight_decay = args.weight_decay
    cfg.buffer.adv_capacity = args.adv_capacity
    cfg.buffer.policy_capacity = args.policy_capacity
    cfg.model.d_model  = args.d_model
    cfg.model.n_layers = args.n_layers
    cfg.model.n_heads  = args.n_heads
    cfg.model.d_ff     = args.d_ff
    use_linear = not args.no_linear_cfr

    # ── Curriculum stage (see cfr/config.py:CFRStageConfig) ─────────────────
    cfg.stage.starting_stack_chips = int(round(args.starting_stack_bb * BIG_BLIND_CHIPS))
    if args.push_fold:
        cfg.stage.allowed_actions = (int(pte.ActionType.FOLD),
                                     int(pte.ActionType.CHECK_CALL),
                                     int(pte.ActionType.ALL_IN))

    if args.smoke:
        args.n_virtual = 4
        cfg.train.n_iterations = 2
        cfg.train.n_traversals_per_iter = 20
        cfg.optim.n_grad_steps_per_refit = 200
        cfg.train.policy_n_grad_steps = 500
        cfg.buffer.adv_capacity = 10_000
        cfg.buffer.policy_capacity = 10_000
        cfg.model.d_model = 64
        cfg.model.n_layers = 2
        cfg.model.n_heads = 4
        cfg.model.d_ff = 256
        args.checkpoint_every_iter = 1
        # Shrink probes so smoke still exercises the path but stays quick.
        args.probe_hands = 40
        # Tiny oracle so --push-fold smoke doesn't pay the full equity build.
        args.oracle_deals = 200_000

    torch.manual_seed(cfg.run.seed)
    # Distributed (multi-GPU DDP): active only under torchrun (WORLD_SIZE>1);
    # otherwise a disabled DistInfo → the exact single-process path as before.
    from . import distributed as ddist
    from torch.nn.parallel import DistributedDataParallel as DDP
    dist = ddist.setup(default_device=args.device)
    device = dist.device
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"--device={args.device} but CUDA unavailable")

    # Only rank 0 logs / probes / checkpoints; other ranks stay quiet.
    def log(*a, **k):
        if dist.is_main:
            print(*a, **k)
    if dist.enabled:
        log(f"[cfr_coro] DDP: world_size={dist.world_size} "
            f"(rank {dist.rank} on {device}); refit + collection sharded "
            f"across ranks, effective batch = {dist.world_size}×"
            f"{cfg.optim.batch_size}.")

    ckpt_dir = Path(cfg.run.ckpt_dir)
    if dist.is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    ddist.barrier(dist)

    # ── Build nets directly on GPU (single process, no fork concerns) ──────
    # No share_memory / forkserver dance. We're one process, we have one
    # CUDA context, all the nets live on GPU the whole time.
    adv_nets = [
        AdvNet(cfg.model).to(device),
        AdvNet(cfg.model).to(device),
    ]
    policy_net = PolicyNet(cfg.model).to(device)

    print(f"[cfr_coro] device={device}  n_virtual={args.n_virtual}")
    print(f"[cfr_coro] AdvNet params (each): {count_parameters(adv_nets[0]):,}")
    print(f"[cfr_coro] cfg.model={cfg.model}")
    print(f"[cfr_coro] cfg.train={cfg.train}")
    print(f"[cfr_coro] cfg.stage={cfg.stage}")

    # ── Resume from the latest checkpoint in --ckpt-dir (weights + iteration) ──
    # Nets + iteration here; the reservoir buffers (the training data) are
    # restored separately once they are constructed, below — giving an exact,
    # zero-regression resume when --no-save-buffers is not set. If buffers are
    # NOT available, refit_adv_net's warm-up scaling protects the loaded net from
    # a cold-buffer overfit. --no-resume forces fresh; --resume-from picks a file.
    # Buffers are sharded per rank under DDP (each rank owns a reservoir shard),
    # so each saves/restores its own file; single-process keeps the old name.
    buffers_path = (ckpt_dir / (f"cfr_buffers_rank{dist.rank}.npz" if dist.enabled
                                else "cfr_buffers.npz"))
    start_iter = 0
    resumed = False
    if not args.no_resume:
        resume_path = (Path(args.resume_from) if args.resume_from
                       else find_latest_checkpoint(ckpt_dir))
        if resume_path and resume_path.exists():
            start_iter = load_checkpoint(resume_path, adv_nets, policy_net, device)
            resumed = True
            print(f"[cfr_coro] RESUMED nets from {resume_path} at iteration "
                  f"{start_iter} → continuing at t={start_iter + 1}.")
        else:
            print("[cfr_coro] no checkpoint to resume; starting fresh.")

    # Re-raise cap (0 → None = uncapped). Bounds full-depth re-raise wars.
    max_raises = args.max_raises_per_street if args.max_raises_per_street > 0 else None
    print(f"[cfr_coro] max_raises_per_street={max_raises}  max_depth={args.max_depth}")

    # Curriculum action restriction as a frozenset (fast membership in the
    # traversal hot path); None for the full game.
    allowed = (frozenset(cfg.stage.allowed_actions)
               if cfg.stage.allowed_actions is not None else None)

    # Regret-target scale = effective stack in bb, so loss/grad magnitudes stay
    # interpretable across stack depths (at 10bb the old /100 made the loss print
    # as 0.000). This is a numerics convenience only — Adam + regret matching are
    # scale-invariant, so it does NOT change learning (the BB freeze was
    # weight_decay, not scale; see --weight-decay and diag_regret_scale_freeze).
    regret_scale = float(cfg.stage.starting_stack_bb)
    print(f"[cfr_coro] regret_scale={regret_scale:g} (=effective stack in bb); "
          f"weight_decay={cfg.optim.weight_decay:g}")

    # Stage-1 push/fold: solve the Nash oracle once (cached) so the per-iter
    # validation has a ground-truth target to score the net's ranges against.
    pushfold_oracle = None
    if args.push_fold:
        from .pushfold_validation import run_pushfold_validation
        from evaluate.pushfold_solver import build_oracle, format_grid
        from evaluate.pushfold_reference import format_comparison
        pushfold_oracle = build_oracle(stack_bb=cfg.stage.starting_stack_bb,
                                       n_deals=args.oracle_deals)
        print(format_grid(pushfold_oracle.jam_grid(),
                          f"[cfr_coro] ORACLE SB open-jam % "
                          f"({cfg.stage.starting_stack_bb:g}bb Nash):"))
        # Cross-check the solved oracle against published Nash ranges (a few
        # boundary hands differ — see pushfold_reference for why that's fine).
        if abs(cfg.stage.starting_stack_bb - 10.0) < 1e-6:
            print(format_comparison(pushfold_oracle))

    # Buffers live in main process (only process). Direct attribute access,
    # no IPC payloads — coroutines write into per-traversal lists, the
    # driver drains them inline.
    adv_bufs = [
        ReservoirBuffer(cfg.buffer.adv_capacity, cfg.model.num_actions,
                        rng=np.random.default_rng(cfg.run.seed + 11)),
        ReservoirBuffer(cfg.buffer.adv_capacity, cfg.model.num_actions,
                        rng=np.random.default_rng(cfg.run.seed + 22)),
    ]
    pol_buf = ReservoirBuffer(cfg.buffer.policy_capacity, cfg.model.num_actions,
                              rng=np.random.default_rng(cfg.run.seed + 33))
    named_bufs = {"adv0": adv_bufs[0], "adv1": adv_bufs[1], "pol": pol_buf}

    # Restore the reservoir buffers for an exact resume (the nets were loaded
    # above). If absent (e.g. --no-save-buffers on the prior run), continue with
    # empty buffers — they refill, and the refit warm-up protects the loaded net.
    if resumed and not args.no_save_buffers and buffers_path.exists():
        bt0 = time.time()
        buf_iter = load_buffers(str(buffers_path), named_bufs)
        print(f"[cfr_coro] RESTORED buffers from {buffers_path} "
              f"(saved at iteration {buf_iter}) in {time.time()-bt0:.1f}s → "
              f"adv=[{len(adv_bufs[0]):,},{len(adv_bufs[1]):,}] pol={len(pol_buf):,}")
        if buf_iter != start_iter:
            print(f"  note: buffer iteration {buf_iter} != net iteration "
                  f"{start_iter} (crash between writes); harmless — buffer is "
                  f"training data, nets refit from it.")
    elif resumed:
        print("[cfr_coro] no buffer file to restore; buffers start empty "
              "(refit warm-up will protect the loaded net as they refill).")

    # Rank-distinct RNG + seed offset so each rank's sharded collection explores
    # DIFFERENT hands (otherwise all ranks would traverse identical trees and DDP
    # would just average identical gradients — no data-parallel benefit).
    rng = np.random.default_rng(cfg.run.seed + 44 + dist.rank * 7_000_003)
    # Shard the traversal budget across ranks (rank 0 absorbs the remainder).
    K_total = cfg.train.n_traversals_per_iter
    K_local = K_total // dist.world_size + (
        K_total % dist.world_size if dist.is_main else 0)

    # ── Outer CFR loop ─────────────────────────────────────────────────────
    t_total_start = time.time()
    for t in range(start_iter + 1, cfg.train.n_iterations + 1):
        log(f"\n[cfr_coro] ════════════ iteration t={t}/{cfg.train.n_iterations} ════════════")

        t_collect_start = time.time()
        stats = run_K_traversals(
            K=K_local,
            adv_nets_gpu=adv_nets,
            adv_bufs=adv_bufs,
            pol_buf=pol_buf,
            n_virtual=args.n_virtual,
            t=t,
            rng=rng,
            base_seed=cfg.run.seed + t * 100 + dist.rank * 1_000_003,
            device=device,
            max_depth=args.max_depth,
            starting_stack_chips=cfg.stage.starting_stack_chips,
            allowed=allowed,
            regret_scale=regret_scale,
            max_raises_per_street=max_raises,
        )
        wall = time.time() - t_collect_start
        log(f"  collected K={stats['n_traversals']}×{dist.world_size}ranks in "
            f"{wall:.1f}s ({wall*1000/max(1,stats['n_traversals']):.0f} ms/trav)  "
            f"adv_buf(rank0)=[{len(adv_bufs[0]):,},{len(adv_bufs[1]):,}]  "
            f"pol_buf={len(pol_buf):,}")

        # Refit AdvNets. Under DDP each net is wrapped so its forward all-reduces
        # gradients across ranks; DDP construction also broadcasts rank-0's
        # (freshly re-initialized) weights, so every rank starts the refit from
        # identical parameters even though reset_adv_net_each_iter re-inits with a
        # rank-distinct RNG. After the refit the weights are identical on all
        # ranks (gradients were averaged), so collection next iter is consistent.
        for p in (0, 1):
            if cfg.train.reset_adv_net_each_iter:
                adv_nets[p] = AdvNet(cfg.model).to(device)
            log(f"  refit adv_net[p={p}] on {len(adv_bufs[p])} samples/rank ...")
            fwd = (DDP(adv_nets[p],
                       device_ids=([dist.local_rank] if device.type == "cuda"
                                   else None))
                   if dist.enabled else None)
            r = refit_adv_net(adv_nets[p], adv_bufs[p], cfg, device,
                              use_linear_cfr=use_linear,
                              forward_module=fwd, world_size=dist.world_size,
                              distributed=dist.enabled, amp=args.amp,
                              verbose=dist.is_main)
            del fwd   # drop the DDP wrapper; keep the trained adv_nets[p]
            log(f"    loss: first={r['loss_first']:.2f} last={r['loss_last']:.2f} "
                f"wall={r['wall_s']:.1f}s")

        # ── Learning probes ────────────────────────────────────────────────
        # Cheap "is the net learning poker?" signal: play the freshly-refit
        # AdvNets against a uniform-random opponent in two scenarios — the
        # net dealt AA, and the net dealt 72o. The bb gap between the two
        # is what tells you the policy is hand-strength-aware.
        # CFRAdvPolicy(__init__) puts nets in eval mode; we restore train
        # mode afterwards so the next iteration's refit isn't surprised.
        if dist.is_main and args.probe_every_iter > 0 and t % args.probe_every_iter == 0:
            if pushfold_oracle is not None:
                # Stage-1: score the net's whole jam/call range vs Nash.
                rep = run_pushfold_validation(
                    adv_nets, device, pushfold_oracle,
                    starting_stack_chips=cfg.stage.starting_stack_chips,
                    allowed=allowed)
                print(rep.grids())
                for line in rep.summary_lines(iter_t=t):
                    print(line)
            else:
                results = run_default_probes(adv_nets, device,
                                             n_hands=args.probe_hands,
                                             base_seed=cfg.run.seed + t * 7919)
                for r in results:
                    print(format_probe_line(r, iter_t=t))
            for net in adv_nets:
                net.train(True)

        if (args.checkpoint_every_iter > 0
                and t % args.checkpoint_every_iter == 0):
            # Net checkpoint: rank 0 only (all ranks hold identical synced nets).
            if dist.is_main:
                path = ckpt_dir / f"cfr_iter_{t:04d}.ckpt"
                save_checkpoint(path, adv_nets, policy_net, t, cfg)
                print(f"  saved {path}")
            # Buffers: EACH rank persists its OWN shard (the data lives sharded).
            if not args.no_save_buffers:
                bt0 = time.time()
                save_buffers(str(buffers_path), named_bufs, t)
                log(f"  saved buffers → {buffers_path.name} per rank "
                    f"(rank0 adv=[{len(adv_bufs[0]):,},{len(adv_bufs[1]):,}] "
                    f"pol={len(pol_buf):,}, {time.time()-bt0:.1f}s)")

        # ── Average-strategy (PolicyNet) harvest is now OFFLINE ──────────────
        # The average net is a separate, heavier training problem (many epochs
        # over the full strategy reservoir) and was the rank-0-solo section that
        # idled the other ranks for hours. It has moved out of the loop entirely:
        # cfr_coro only COLLECTS strategy samples (pol_buf) and persists them via
        # the buffer checkpoints above. Train the deployable policy separately
        # with `python -m cfr.harvest_policy --ckpt-dir <this run>` (DDP across
        # all GPUs). policy_net here stays at init and is saved only to keep the
        # checkpoint format stable for resume/load.

        # Resync all ranks before the next iteration's DDP refit (rank 0 may have
        # spent time in the probe/checkpoint blocks above).
        ddist.barrier(dist)

    # ── Final checkpoint (rank 0) ────────────────────────────────────────────
    # No inline PolicyNet training — the average strategy is harvested offline
    # from the saved buffers (see cfr.harvest_policy). policy_net is saved at
    # init only to keep the checkpoint format stable.
    if dist.is_main:
        final = ckpt_dir / "cfr_final.ckpt"
        save_checkpoint(final, adv_nets, policy_net,
                        cfg.train.n_iterations, cfg)
        print(f"\n[cfr_coro] DONE total wall={time.time()-t_total_start:.1f}s "
              f"saved {final}")
        print(f"[cfr_coro] harvest the average policy with:\n"
              f"    torchrun --standalone --nproc_per_node={dist.world_size} "
              f"-m cfr.harvest_policy --ckpt-dir {ckpt_dir} --amp")
    ddist.barrier(dist)
    ddist.cleanup(dist)


if __name__ == "__main__":
    main()
