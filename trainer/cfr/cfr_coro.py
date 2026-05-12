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

from .buffers import ReservoirBuffer
from .config import CFRConfig
from .models import AdvNet, PolicyNet, count_parameters
from .regret_matching import regret_matching_np
from .traversal import DEFAULT_MAX_DEPTH, REGRET_SCALE
from .train import refit_adv_net, train_policy_net, save_checkpoint
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
                  max_depth: int = DEFAULT_MAX_DEPTH):
    """Generator-coroutine implementing one external-sampling MCCFR traversal.

    YIELDS:
        (actor: int, x: np.ndarray) tuples, one per inference request.
    EXPECTS via .send():
        np.ndarray of shape (NUM_ACTIONS,) — predicted regrets for the
        AdvNet of player `actor`.
    RETURNS:
        float — the traverser's expected utility at the root (chip units).

    SEMANTIC EQUIVALENCE TO `traversal.traverse`:
        Identical algorithm; only the inference call site is different.
        Where `traversal.traverse` calls `_predict_regrets(net, x, dev)`
        directly, this function YIELDS the request and resumes when given
        the prediction. Buffer writes also accumulate to local lists
        (so the driver can attribute them to a traversal/traverser at
        completion time) instead of pushing directly to a ReservoirBuffer.
    """
    # ── Terminal: no inference needed, return payoff directly. ────────────
    # `return X` in a generator raises StopIteration(value=X) at the caller's
    # `yield from` site (or at .send() in the driver if this is a top-level
    # coroutine, see _drive_round below).
    if env.is_terminal():
        return float(env.payoffs_bb()[traverser])

    obs   = env.observation()
    actor = int(env.to_act())
    # Defensive copy: the obs.x ndarray is freshly built by pybind11 (see
    # py_env.cpp:153), so we don't strictly need .copy(). But we're going
    # to store this in pol_writes/adv_writes, and we want the stored array
    # to be independent of any future re-evaluation of obs.x. Cheap insurance.
    x     = obs.x.astype(np.float32, copy=False).copy()
    legal = list(obs.legal)
    n_legal = len(legal)
    mask  = _legal_mask_from_obs(obs)

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  THE YIELD POINT — request an inference, suspend until satisfied.   ║
    # ║                                                                      ║
    # ║  Driver collects this yielded (actor, x), batches with other        ║
    # ║  coros' yields, runs one GPU forward, and .send()s the result here. ║
    # ║  This is THE crux of why this file exists.                          ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    pred_r = (yield (actor, x))    # type: np.ndarray
    sigma = regret_matching_np(pred_r, mask)

    # Every visited state contributes to the policy buffer (see Deep CFR
    # algorithm note in traversal.py).
    pol_writes.append((x, sigma))

    # Depth cap: substitute AdvNet-σ-weighted bootstrap for further recursion.
    # Multiply by REGRET_SCALE to convert from scaled-regret units back to
    # chip units (so it integrates correctly with terminal payoffs at higher
    # recursion levels).
    if env.state().history_size >= max_depth:
        return float((sigma * pred_r * mask).sum()) * REGRET_SCALE

    if actor == traverser:
        # ── TRAVERSER NODE: branch every legal action ───────────────────────
        # Each branch is a separate sub-coroutine via `yield from`. The
        # driver sees a stream of yields from this generator AS IF they
        # came from this function body — `yield from` is transparent.
        # When child returns its value, `v_a` binds to it.
        action_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for at in legal:
            child = env.clone()
            child.step_action(at)
            v_a = yield from traverse_coro(child, traverser, rng,
                                           adv_writes, pol_writes,
                                           max_depth=max_depth)
            action_values[int(at)] = v_a
        v_state = float((sigma * action_values).sum())
        regrets = (action_values - v_state) * mask / REGRET_SCALE
        adv_writes.append((x, regrets.astype(np.float32, copy=False)))
        return v_state

    # ── OPPONENT NODE: external sampling ────────────────────────────────────
    # Sample one action, recurse on env in place (no clone needed for a
    # single-path branch). Use `yield from` to delegate; the recursive
    # call's return value is THIS function's return value.
    legal_int = np.array([int(at) for at in legal], dtype=np.int64)
    legal_probs = sigma[legal_int]
    z = legal_probs.sum()
    if z > 0.0:
        legal_probs = legal_probs / z
    else:
        legal_probs = np.full(n_legal, 1.0 / n_legal, dtype=np.float32)
    chosen_local_idx = int(rng.choice(n_legal, p=legal_probs))
    env.step(chosen_local_idx)
    # Note the parentheses: `return (yield from ...)` — the `yield from`
    # expression evaluates to the inner traversal's return value, and we
    # return that. Without parens, Python parses this as `return yield ...`
    # which is a SyntaxError. Trailing `yield from` returns are common in
    # coroutine code; the parens are a habit worth forming.
    return (yield from traverse_coro(env, traverser, rng,
                                     adv_writes, pol_writes,
                                     max_depth=max_depth))


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
    # The next pending request from this coroutine: (actor_player_id, x).
    # None means: this slot has just finished a traversal and hasn't been
    # restarted yet (transient state during driver's iteration).
    pending: Optional[tuple[int, np.ndarray]] = None
    traverser: int = 0
    adv_writes: list = field(default_factory=list)
    pol_writes: list = field(default_factory=list)


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
                              slot.adv_writes, slot.pol_writes)
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
                   xs: list[np.ndarray],
                   device: torch.device) -> list[np.ndarray]:
    """Run ONE GPU forward over a list of x-vectors and return per-row outputs.

    The whole point of this trainer: amortize kernel launch overhead by
    batching all coroutines' pending requests into one forward.

    np.stack copies into a contiguous (B, X_DIM) ndarray. torch.from_numpy
    is zero-copy. .to(device, non_blocking=True) starts the H2D transfer
    while we keep going (next instruction is the forward, which queues
    behind the transfer on the same CUDA stream).
    """
    if not xs:
        return []
    # Stack into (B, x_dim). dtype=float32 already, but explicit for safety.
    batch_np = np.stack(xs, axis=0).astype(np.float32, copy=False)
    batch = torch.from_numpy(batch_np).to(device, non_blocking=True)
    with torch.no_grad():
        out = net(batch)
    # .cpu().numpy() forces sync (GPU work to complete before numpy reads it).
    # This is implicit barrier between batch run and result dispatch.
    out_np = out.detach().cpu().numpy().astype(np.float32, copy=False)
    return [out_np[k] for k in range(len(xs))]


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
    sb_idx: list[int] = []   # slot indices needing AdvNet[0]
    sb_xs:  list[np.ndarray] = []
    bb_idx: list[int] = []
    bb_xs:  list[np.ndarray] = []
    for i, s in enumerate(slots):
        if s.pending is None:
            continue
        actor, x = s.pending
        if actor == 0:
            sb_idx.append(i); sb_xs.append(x)
        else:
            bb_idx.append(i); bb_xs.append(x)

    # ONE forward per AdvNet. We could fuse them (concat both, run one
    # forward, split) only if the AdvNets shared weights — they don't,
    # by design (per-player regret functions). So two batches: one each.
    sb_preds = _batch_forward(adv_nets_gpu[0], sb_xs, device)
    bb_preds = _batch_forward(adv_nets_gpu[1], bb_xs, device)

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
                     max_depth: int = DEFAULT_MAX_DEPTH) -> dict:
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
    slots = [
        _Slot(env=pte.Env((base_seed + i * 1009 + t * 1234567) & 0xFFFFFFFFFFFFFFFF))
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
            for x, regrets in s.adv_writes:
                adv_bufs[tr].add(x, regrets, t)
            for x, sigma in s.pol_writes:
                pol_buf.add(x, sigma, t)
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
    p.add_argument("--policy-grad-steps", type=int, default=50000)
    p.add_argument("--adv-capacity", type=int, default=2_000_000)
    p.add_argument("--policy-capacity", type=int, default=4_000_000)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--no-linear-cfr", action="store_true")
    p.add_argument("--ckpt-dir", type=str, default="runs/cfr_coro_latest")
    p.add_argument("--checkpoint-every-iter", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    p.add_argument("--probe-every-iter", type=int, default=1,
                   help="Run AA / 72o learning probes every N iterations. "
                        "0 disables. Each probe plays --probe-hands against "
                        "uniform-random and reports mbb/hand + preflop action "
                        "mix — a cheap 'is the net learning poker?' signal.")
    p.add_argument("--probe-hands", type=int, default=400,
                   help="Hands per probe scenario (AA, then 72o). 400 gives "
                        "SE ≈ 300 mbb/hand on the AA scenario — fine for "
                        "tracking a several-bb signal across iterations.")
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
    cfg.buffer.adv_capacity = args.adv_capacity
    cfg.buffer.policy_capacity = args.policy_capacity
    cfg.model.hidden = args.hidden
    cfg.model.n_layers = args.n_layers
    use_linear = not args.no_linear_cfr

    if args.smoke:
        args.n_virtual = 4
        cfg.train.n_iterations = 2
        cfg.train.n_traversals_per_iter = 20
        cfg.optim.n_grad_steps_per_refit = 200
        cfg.train.policy_n_grad_steps = 500
        cfg.buffer.adv_capacity = 10_000
        cfg.buffer.policy_capacity = 10_000
        cfg.model.hidden = 128
        cfg.model.n_layers = 2
        args.checkpoint_every_iter = 1
        # Shrink probes so smoke still exercises the path but stays quick.
        args.probe_hands = 40

    torch.manual_seed(cfg.run.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"--device={args.device} but CUDA unavailable")

    ckpt_dir = Path(cfg.run.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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

    # Buffers live in main process (only process). Direct attribute access,
    # no IPC payloads — coroutines write into per-traversal lists, the
    # driver drains them inline.
    adv_bufs = [
        ReservoirBuffer(cfg.buffer.adv_capacity, cfg.model.x_dim,
                        cfg.model.num_actions,
                        rng=np.random.default_rng(cfg.run.seed + 11)),
        ReservoirBuffer(cfg.buffer.adv_capacity, cfg.model.x_dim,
                        cfg.model.num_actions,
                        rng=np.random.default_rng(cfg.run.seed + 22)),
    ]
    pol_buf = ReservoirBuffer(cfg.buffer.policy_capacity, cfg.model.x_dim,
                              cfg.model.num_actions,
                              rng=np.random.default_rng(cfg.run.seed + 33))

    rng = np.random.default_rng(cfg.run.seed + 44)

    # ── Outer CFR loop ─────────────────────────────────────────────────────
    t_total_start = time.time()
    for t in range(1, cfg.train.n_iterations + 1):
        print(f"\n[cfr_coro] ════════════ iteration t={t}/{cfg.train.n_iterations} ════════════")

        t_collect_start = time.time()
        stats = run_K_traversals(
            K=cfg.train.n_traversals_per_iter,
            adv_nets_gpu=adv_nets,
            adv_bufs=adv_bufs,
            pol_buf=pol_buf,
            n_virtual=args.n_virtual,
            t=t,
            rng=rng,
            base_seed=cfg.run.seed + t * 100,
            device=device,
            max_depth=args.max_depth,
        )
        wall = time.time() - t_collect_start
        print(f"  collected K={stats['n_traversals']} in {wall:.1f}s "
              f"({wall*1000/max(1,stats['n_traversals']):.0f} ms/traversal)  "
              f"adv_buf=[{len(adv_bufs[0]):,},{len(adv_bufs[1]):,}]  "
              f"pol_buf={len(pol_buf):,}")

        # Refit AdvNets. No CPU sync needed — actors and learner share the
        # same nets in this architecture (they ARE the same Python objects).
        for p in (0, 1):
            if cfg.train.reset_adv_net_each_iter:
                adv_nets[p] = AdvNet(cfg.model).to(device)
            print(f"  refit adv_net[p={p}] on {len(adv_bufs[p])} samples ...")
            r = refit_adv_net(adv_nets[p], adv_bufs[p], cfg, device,
                              use_linear_cfr=use_linear)
            print(f"    loss: first={r['loss_first']:.2f} last={r['loss_last']:.2f} "
                  f"wall={r['wall_s']:.1f}s")

        # ── Learning probes ────────────────────────────────────────────────
        # Cheap "is the net learning poker?" signal: play the freshly-refit
        # AdvNets against a uniform-random opponent in two scenarios — the
        # net dealt AA, and the net dealt 72o. The bb gap between the two
        # is what tells you the policy is hand-strength-aware.
        # CFRAdvPolicy(__init__) puts nets in eval mode; we restore train
        # mode afterwards so the next iteration's refit isn't surprised.
        if args.probe_every_iter > 0 and t % args.probe_every_iter == 0:
            results = run_default_probes(adv_nets, device,
                                         n_hands=args.probe_hands,
                                         base_seed=cfg.run.seed + t * 7919)
            for r in results:
                print(format_probe_line(r, iter_t=t))
            for net in adv_nets:
                net.train(True)

        if (args.checkpoint_every_iter > 0
                and t % args.checkpoint_every_iter == 0):
            path = ckpt_dir / f"cfr_iter_{t:04d}.ckpt"
            save_checkpoint(path, adv_nets, policy_net, t, cfg)
            print(f"  saved {path}")

    # ── Final policy training ──────────────────────────────────────────────
    print(f"\n[cfr_coro] training PolicyNet on {len(pol_buf):,} samples")
    pr = train_policy_net(policy_net, pol_buf, cfg, device,
                          use_linear_cfr=use_linear)
    print(f"  loss first={pr['loss_first']:.4f} last={pr['loss_last']:.4f} "
          f"wall={pr['wall_s']:.1f}s")

    final = ckpt_dir / "cfr_final.ckpt"
    save_checkpoint(final, adv_nets, policy_net,
                    cfg.train.n_iterations, cfg)
    print(f"\n[cfr_coro] DONE total wall={time.time()-t_total_start:.1f}s "
          f"saved {final}")


if __name__ == "__main__":
    main()
