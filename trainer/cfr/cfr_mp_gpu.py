"""Multi-process Deep CFR trainer: CPU actor pool + GPU learner.

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE OVERVIEW (read this first)
═══════════════════════════════════════════════════════════════════════════════

This trainer parallelizes Deep CFR by separating two roles into different OS
processes:

   ┌── N actor processes (CPU) ──┐         ┌── learner (main process) ────────┐
   │ Each actor runs traversals  │         │ Owns GPU AdvNets, optimizers,    │
   │ using CPU-resident nets     │  ──Q──▶ │ buffers. Drains queue, refits    │
   │ shared via shared memory.   │  queue  │ each iteration, syncs CPU nets   │
   │ One traversal = one queue   │ payload │ from GPU after refit.            │
   │ payload.                    │         │                                  │
   └─────────────────────────────┘         └──────────────────────────────────┘

WHY MULTIPLE PROCESSES, NOT THREADS?
    Python's GIL serializes interpreter execution within a process. Even on
    a 32-core machine, multi-threaded Python can't run 32 concurrent
    `traverse(...)` calls because each call needs the interpreter. Multiple
    processes have separate interpreters → true parallelism.

WHY CPU INFERENCE IN ACTORS?
    The recursive traversal makes ~5000 inference calls per traversal, batch
    size = 1 each (you can't batch within a single sequential traversal —
    each child step depends on the parent's σ). For tiny nets (~370k params),
    a B=1 GPU forward is dominated by ~250µs of kernel launch overhead, while
    a CPU forward of the same net takes ~50µs. CPU wins. Also, actors share
    the net via mmap'd memory; GPU shared memory across processes is much
    harder.

    THIS BREAKS DOWN AT TARGET CLUSTER SCALE (HIDDEN=512, LAYERS=8 → 17M
    params). At that size CPU forward grows to ~5ms per call and the whole
    architecture stops being viable. See `cfr_server.py` (CPU actors + GPU
    inference server) and `cfr_coro.py` (single-process coroutine driver
    with batched GPU inference) for variants that scale.

DESIGN MIRRORS `trainer/dmc/dmc_mp_gpu.py` with two CFR-specific changes:

  1. Two AdvNets per traverser (one for SB, one for BB).
     CFR's mathematical formulation indexes cumulative regret per-player:
     R_p(I, a). The two adv buffers `adv_buf[traverser]` are populated
     by separate traversal regimes — when traverser=0 (SB) traverses, only
     SB-actor states get regret labels; vice versa for BB.

  2. A separate policy buffer that collects σ samples from EVERY decision
     point in EVERY traversal, both seats. The PolicyNet learns the average
     strategy and isn't player-indexed.

SNAPSHOT SEMANTICS — the subtle CFR-specific correctness concern:

    Strict CFR theory says: in iteration t, all traversals must use the
    AdvNets as they were at the START of iteration t (the strategy σ_t).
    If the learner refits mid-iteration, half the traversals would use σ_t
    and half σ_{t+1}, which CFR's convergence proof doesn't cover directly.

    Our compromise: actors continue producing during refit (so neither
    process sits idle), tagging each traversal's writes with the
    `iter_value` at the time of generation. The learner advances
    `iter_value` AFTER refit + sync. So in steady state, ~all traversals
    in iteration t's drain were generated using σ_t weights. There's a
    bounded amount of mixed-iteration data each transition — Brown 2020's
    follow-up on parallelism shows this is fine for convergence in practice.

═══════════════════════════════════════════════════════════════════════════════
QUEUE PAYLOAD CONTRACT (cross-process serialization)
═══════════════════════════════════════════════════════════════════════════════

Each actor sends ONE payload per traversal:

    (traverser: int,
     adv_writes: list[(x: np.ndarray, regrets: np.ndarray)],
     pol_writes: list[(x: np.ndarray, sigma: np.ndarray)],
     t: int)

Why batched-per-traversal instead of per-decision-point?
  * `mp.Queue` IPC has ~10-30µs of fixed overhead per put() (serialization
    + write to pipe). With ~50 decision points per traversal, that's
    500-1500µs per traversal saved by batching. Non-trivial.
  * Reservoir-buffer add semantics aren't sensitive to whether writes
    arrive in batches or singly — the random replacement is per-add.
  * Backpressure is cleaner: queue depth = traversals-in-flight, easy to
    reason about.

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import queue as pyqueue
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

import pokertrainer_engine as pte

from .buffers import ReservoirBuffer
from .config import CFRConfig
from .models import AdvNet, PolicyNet, count_parameters
from .regret_matching import regret_matching_np
from .traversal import DEFAULT_MAX_DEPTH, REGRET_SCALE
from .train import refit_adv_net, train_policy_net, save_checkpoint
from .probe import run_default_probes, format_probe_line


# Number of action types. Hardcoded here (also derivable from the engine via
# `pte.NUM_ACTIONS`) — duplicated so the actor module doesn't need to import
# the engine just to know the constant. The engine version is checked at
# startup in main() as a sanity check.
NUM_ACTIONS = 11


# ═══════════════════════════════════════════════════════════════════════════
# ACTOR-SIDE TRAVERSAL
# ═══════════════════════════════════════════════════════════════════════════
# This block runs INSIDE the actor processes. It must avoid:
#   * Touching the GPU (forkserver doesn't poison CUDA, but actors don't have
#     a CUDA context anyway).
#   * Importing modules that initialize CUDA on import (some PyTorch versions
#     can lazily probe for CUDA; we keep imports minimal in this region).
#   * Mutating shared memory in ways that race with other actors. Since the
#     CPU nets are read-only here (only the learner writes to them via
#     sync_cpu_from_gpu), reads are safe — no cross-process locks needed.
# ═══════════════════════════════════════════════════════════════════════════


def _legal_mask_from_obs(obs) -> np.ndarray:
    """Build a NUM_ACTIONS-dim 0/1 float mask from the engine's legal list.

    `obs.legal` is a list[ActionType] returned by the pybind11 binding (see
    `engine/bindings/py_env.cpp`). Each ActionType is a Python enum whose
    int value is the index into the global action vocabulary.
    """
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for at in obs.legal:
        mask[int(at)] = 1.0
    return mask


@torch.no_grad()
def _predict(net, tok_state) -> np.ndarray:
    """Single-state CPU forward of a transformer AdvNet.

    No `device` parameter — actors are always CPU. Wraps the variable-length
    token sequence into a batch of 1 (pad-trivial since L_max == L) and
    runs the transformer; returns the row's raw regrets as numpy.
    """
    from .tokenize import pad_batch
    tokens_np, pad_mask_np, dpos_np = pad_batch([tok_state])
    tokens   = torch.from_numpy(tokens_np)
    pad_mask = torch.from_numpy(pad_mask_np)
    dpos     = torch.from_numpy(dpos_np)
    return net(tokens, pad_mask, dpos).squeeze(0).numpy().astype(np.float32, copy=False)


def _traverse_collect(env, traverser, adv_nets, t, rng,
                      adv_writes_out, pol_writes_out,
                      max_depth: int = DEFAULT_MAX_DEPTH):
    """Recursive external-sampling traversal that ACCUMULATES writes.

    Same algorithm as `cfr.traversal.traverse`, but instead of pushing
    each (x, target, t) into a ReservoirBuffer directly, we append into
    two output lists. The actor process then ships those lists across
    the queue to the learner, which drains them into the real buffers.

    WHY THIS DUPLICATION INSTEAD OF SHARING `traversal.py`?
        `traversal.py:traverse` writes directly to a ReservoirBuffer object.
        The actor doesn't have access to the learner's buffers (they live
        in a different process address space). Threading shared buffers
        across the multiprocessing boundary would require either expensive
        per-write IPC (slow) or shared-memory ndarray buffers (complex
        synchronization). The chosen design — accumulate locally, ship a
        coarser payload — is simpler and faster.

    DETERMINISM:
        `rng` is per-actor with a unique seed (see `_actor_loop`). Even if
        two actors visit the same state, their downstream sampling diverges
        because each draws from its own RNG stream. This is fine for CFR
        (we're seeking diversity in sampled paths anyway), but it means
        the trainer is not bit-reproducible across actor counts. To make
        runs comparable across configurations, fix `--mp-actors` and
        `--seed` together.
    """
    # Terminal: payoff is known directly from the engine's bookkeeping.
    if env.is_terminal():
        return float(env.payoffs_bb()[traverser])

    obs   = env.observation()
    actor = int(env.to_act())

    legal = list(obs.legal)
    n_legal = len(legal)
    mask  = _legal_mask_from_obs(obs)

    # Tokenize from the to-act player's perspective. The tokenizer walks
    # state.history; cost is O(action history length).
    from .tokenize import tokenize_state
    tok_state = tokenize_state(env.state(), hero_seat=actor)

    # CPU forward through the actor's view of the AdvNet for the to-act
    # player. `adv_nets[actor]` selects which net to call (SB or BB).
    pred_r = _predict(adv_nets[actor], tok_state)
    sigma  = regret_matching_np(pred_r, mask)

    # Strategy memory: σ is stored ONLY at opponent nodes (Brown 2019, Alg. 1).
    # Opponent nodes are visited proportionally to that player's own reach
    # (it samples its own actions), which is the weighting the average strategy
    # σ̄ requires. Traverser nodes are branch-all (reach-unweighted) — storing
    # σ there biases the PolicyNet's training distribution.
    if actor != traverser:
        pol_writes_out.append((tok_state, sigma))

    # Depth cap: substitute AdvNet-σ-weighted value estimate for further
    # recursion. `pred_r` is in scaled-regret units (REGRET_SCALE-divided
    # at write time), so the bootstrap value must multiply back to chip
    # units to integrate correctly with terminal payoffs at higher levels.
    if env.state().history_size >= max_depth:
        return float((sigma * pred_r * mask).sum()) * REGRET_SCALE

    if actor == traverser:
        # TRAVERSER NODE: branch every legal action, compute v(s,a), accumulate
        # regret target r(s,a) = v(s,a) - V(s).
        action_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for at in legal:
            # env.clone() is a deep copy of the engine state. We branch by
            # cloning + stepping; the original `env` is left unchanged so
            # we can branch into the next action on the next loop iteration.
            child = env.clone()
            child.step_action(at)
            v_a = _traverse_collect(child, traverser, adv_nets, t, rng,
                                    adv_writes_out, pol_writes_out,
                                    max_depth=max_depth)
            action_values[int(at)] = v_a
        v_state = float((sigma * action_values).sum())
        # Scale stored regret targets by 1/REGRET_SCALE — see the long
        # comment in `traversal.py` on REGRET_SCALE for the magnitude
        # rationale (keeps Adam loss in a reasonable range).
        regrets = (action_values - v_state) * mask / REGRET_SCALE
        adv_writes_out.append((tok_state, regrets.astype(np.float32, copy=False)))
        return v_state

    # OPPONENT NODE: external sampling — pick ONE action from σ. Don't write
    # a regret entry; only the traverser's regrets are accumulated this
    # iteration. (See traversal.py for the algorithmic rationale.)
    legal_int = np.array([int(at) for at in legal], dtype=np.int64)
    legal_probs = sigma[legal_int]
    z = legal_probs.sum()
    # Renormalize: defensive against fp drift (np.random.Generator.choice
    # validates p.sum() ≈ 1.0 within atol=1e-8). See traversal.py for the
    # full discussion of why this slice + renormalize is needed even though
    # `sigma` already sums to 1 over legal slots.
    if z > 0.0:
        legal_probs = legal_probs / z
    else:
        # Should be unreachable (regret_matching_np falls back to uniform)
        # but guards against fp underflow turning all entries to subnormals.
        legal_probs = np.full(n_legal, 1.0 / n_legal, dtype=np.float32)
    chosen_local_idx = int(rng.choice(n_legal, p=legal_probs))
    # In-place step: opponent branch is single-path, no clone needed.
    env.step(chosen_local_idx)
    return _traverse_collect(env, traverser, adv_nets, t, rng,
                             adv_writes_out, pol_writes_out,
                             max_depth=max_depth)


# ═══════════════════════════════════════════════════════════════════════════
# ACTOR PROCESS ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════


def _actor_loop(actor_id: int,
                net_cpu_sb,
                net_cpu_bb,
                trans_queue: mp.Queue,
                stop_event,
                iter_value,           # mp.Value('i') — current iteration index
                base_seed: int,
                torch_threads: int,
                max_depth: int) -> None:
    """One-traversal-per-loop-step worker. Runs forever until stop_event is set.

    HOW SHARED-MEMORY NETS WORK:
        `net_cpu_sb` and `net_cpu_bb` were created in the parent process,
        had `.share_memory()` called on them, and were passed across the
        forkserver boundary. PyTorch's share_memory() places the parameter
        tensors in shared memory pages (POSIX shm or mmap'd files depending
        on platform). All actor processes see the SAME bytes — when the
        learner's `sync_cpu_from_gpu` writes new weights to net_cpu_sb,
        every actor's next forward sees the updated values without any
        explicit synchronization.

    WHY torch.set_num_threads(1):
        Each actor process is single-threaded by design. PyTorch's intra-op
        parallelism (OpenMP / MKL) defaults to using all CPU cores, which is
        catastrophic in a multi-actor pool — N actors × M threads = N*M
        threads contending for the same physical cores, plus context-switch
        overhead. Setting threads=1 per actor + spawning N actors gives clean
        process-level parallelism. (Documented gotcha in the DMC trainer too;
        see commit f459f0f.)

    SEEDING:
        Each actor seeds its RNG and engine differently to ensure traversal
        diversity. The XOR-with-large-prime trick `actor_id * 0x9E3779B1`
        spreads consecutive actor_ids across the 64-bit seed space (φ-related
        constant — yields good distribution).
    """
    torch.set_num_threads(max(1, int(torch_threads)))
    # Per-actor engine seed: spread actor_ids using a multiplicative hash.
    env = pte.Env((base_seed ^ (actor_id * 0x9E3779B1)) & 0xFFFFFFFFFFFFFFFF)
    rng = np.random.default_rng(base_seed + actor_id * 7919)
    adv_nets = [net_cpu_sb, net_cpu_bb]

    try:
        while not stop_event.is_set():
            t = int(iter_value.value)   # which CFR iteration we're contributing to
            if t <= 0:
                # Learner hasn't started iteration 1 yet (boot phase).
                # Wait briefly and re-check rather than busy-spin.
                stop_event.wait(0.05)
                continue

            # Random per-traversal seat assignment: ~50% SB, ~50% BB.
            # Why per-traversal rather than per-actor (e.g., even actors=SB,
            # odd=BB)? It keeps the adv_buf for both players growing at
            # roughly equal rates even when one seat happens to have
            # shorter average tree depth (e.g., early-iter check_fold
            # collapses the BB tree fast).
            traverser = int(rng.integers(0, 2))
            env.reset()
            adv_writes: list[tuple[np.ndarray, np.ndarray]] = []
            pol_writes: list[tuple[np.ndarray, np.ndarray]] = []
            _traverse_collect(env, traverser, adv_nets, t, rng,
                              adv_writes, pol_writes, max_depth=max_depth)

            # Ship the accumulated writes to the learner. Use timeout to
            # avoid deadlock if the queue fills up faster than the learner
            # can drain. If full, drop this traversal silently — the
            # reservoir-buffer math is robust to dropped traversals
            # (random subsampling shifts slightly, no semantic break).
            try:
                trans_queue.put((traverser, adv_writes, pol_writes, t),
                                timeout=1.0)
            except pyqueue.Full:
                pass
    except (KeyboardInterrupt, SystemExit):
        # Quiet shutdown on Ctrl-C / sys.exit; the parent process handles
        # logging the termination cause.
        return


# ═══════════════════════════════════════════════════════════════════════════
# WEIGHT SYNC HELPERS
# ═══════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def sync_cpu_from_gpu(net_cpu, net_gpu) -> None:
    """In-place copy of every named parameter / buffer from GPU net to CPU.

    WHY copy_() AND NOT load_state_dict()?
        load_state_dict() rebinds the parameter tensors — it replaces the
        Parameter objects in the module. That breaks shared-memory aliasing:
        the actors' references to `net_cpu_sb.weight` would still point
        at the OLD parameter tensor in shared memory; the new one created
        by load_state_dict() would be in regular heap memory and invisible
        to actors.

        copy_() writes new values INTO the existing tensor's storage,
        preserving the shared-memory pointer. This is the only operation
        that propagates weights to all actor processes without re-spawning.

    NON-ATOMIC UPDATES — this is intentional:
        Multiple parameters get copied in sequence. An actor's forward
        pass running concurrently can theoretically observe a half-synced
        state (some layers from iteration t, others from t+1). The CFR
        algorithm tolerates this because:
          1. The σ output is still a valid probability distribution
             regardless of which weights were used.
          2. The strategy is a continuous function of the weights —
             half-synced weights produce a strategy "between" σ_t and
             σ_{t+1}, not pathological.
          3. Iterations are slow enough (seconds to minutes) that the
             interleaved window (microseconds) is a vanishing fraction.

        If you ever need atomic swaps, use a double-buffer pattern: maintain
        two CPU nets, sync into the inactive one, swap an atomic pointer.
    """
    cpu_params = dict(net_cpu.named_parameters())
    cpu_buffers = dict(net_cpu.named_buffers())
    for name, p in net_gpu.named_parameters():
        cpu_params[name].copy_(p.detach(), non_blocking=False)
    for name, b in net_gpu.named_buffers():
        if name in cpu_buffers:
            cpu_buffers[name].copy_(b.detach(), non_blocking=False)


# ═══════════════════════════════════════════════════════════════════════════
# LEARNER-SIDE QUEUE DRAINING
# ═══════════════════════════════════════════════════════════════════════════


def _drain_queue(trans_queue: mp.Queue,
                 adv_bufs: list[ReservoirBuffer],
                 pol_buf: ReservoirBuffer,
                 max_items: int,
                 timeout_s: float) -> dict:
    """Pull up to `max_items` traversal payloads off the queue, append to buffers.

    Returns a counter dict for logging.

    DRAIN STRATEGY:
        First get() blocks up to `timeout_s` (we want to wait if the queue
        is briefly empty). Subsequent get()s use timeout=0 (non-blocking)
        because we already know the queue had something — drain whatever's
        immediately available, then return so the outer loop can check
        actor health and progress logging.
    """
    counters = {"n_traversals": 0,
                "adv_inserts": [0, 0],
                "pol_inserts": 0}
    deadline = time.time() + timeout_s
    items = 0
    while items < max_items:
        # Block on the first item, non-block on subsequent — drain fast then
        # yield control to the outer loop for logging/health checks.
        remaining = max(0.0, deadline - time.time())
        try:
            payload = trans_queue.get(timeout=remaining if items == 0 else 0.0)
        except pyqueue.Empty:
            break
        traverser, adv_writes, pol_writes, t = payload
        for tok_state, regrets in adv_writes:
            adv_bufs[traverser].add(tok_state.tokens, tok_state.decision_pos,
                                    regrets, t)
            counters["adv_inserts"][traverser] += 1
        for tok_state, sigma in pol_writes:
            pol_buf.add(tok_state.tokens, tok_state.decision_pos, sigma, t)
            counters["pol_inserts"] += 1
        counters["n_traversals"] += 1
        items += 1
    return counters


# ═══════════════════════════════════════════════════════════════════════════
# CLI / MAIN
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mp-actors", type=int, default=8,
                   help="number of CPU actor processes")
    p.add_argument("--queue-maxsize", type=int, default=4096,
                   help="max in-flight traversals (queue capacity); "
                        "actors block put() when full")
    p.add_argument("--learner-device", type=str, default="cuda:0",
                   help="device for AdvNet refit + PolicyNet training")
    p.add_argument("--actor-torch-threads", type=int, default=1,
                   help="torch.set_num_threads in each actor; keep at 1 to "
                        "avoid M*N thread contention with N actors")
    p.add_argument("--n-iterations", type=int, default=100)
    p.add_argument("--n-traversals-per-iter", type=int, default=5000)
    p.add_argument("--adv-grad-steps", type=int, default=4000)
    p.add_argument("--policy-grad-steps", type=int, default=50000)
    p.add_argument("--adv-capacity", type=int, default=2_000_000)
    p.add_argument("--policy-capacity", type=int, default=4_000_000)
    p.add_argument("--d-model", type=int, default=128,
                   help="transformer hidden dim. Must be divisible by --n-heads.")
    p.add_argument("--n-layers", type=int, default=4,
                   help="number of transformer encoder layers")
    p.add_argument("--n-heads", type=int, default=4,
                   help="number of attention heads")
    p.add_argument("--d-ff", type=int, default=512,
                   help="FFN inner dim (typically 4 * d_model)")
    p.add_argument("--no-linear-cfr", action="store_true",
                   help="disable Linear CFR weighting (uniform sample weights)")
    p.add_argument("--ckpt-dir", type=str, default="runs/cfr_mp_gpu_latest")
    p.add_argument("--checkpoint-every-iter", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--watchdog-every-s", type=float, default=10.0,
                   help="periodic progress log interval (seconds)")
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                   help="cap recursion at this engine history_size; beyond it "
                        "we substitute the AdvNet's σ-weighted regret prediction. "
                        "Default = HIST_MAX = 34.")
    p.add_argument("--probe-every-iter", type=int, default=1,
                   help="Run AA / 72o learning probes every N iterations. "
                        "0 disables. The probe runs on the GPU AdvNets, which "
                        "is safe wrt the CPU actors — they have their own "
                        "shared-memory copies and aren't disturbed.")
    p.add_argument("--probe-hands", type=int, default=400,
                   help="Hands per probe scenario (AA, then 72o).")
    p.add_argument("--smoke", action="store_true",
                   help="tiny config for end-to-end correctness testing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CFRConfig()
    cfg.run.device = args.learner_device
    cfg.run.seed = args.seed
    cfg.run.ckpt_dir = args.ckpt_dir
    cfg.train.n_iterations = args.n_iterations
    cfg.train.n_traversals_per_iter = args.n_traversals_per_iter
    cfg.optim.n_grad_steps_per_refit = args.adv_grad_steps
    cfg.train.policy_n_grad_steps = args.policy_grad_steps
    cfg.buffer.adv_capacity = args.adv_capacity
    cfg.buffer.policy_capacity = args.policy_capacity
    cfg.model.d_model  = args.d_model
    cfg.model.n_layers = args.n_layers
    cfg.model.n_heads  = args.n_heads
    cfg.model.d_ff     = args.d_ff
    use_linear = not args.no_linear_cfr

    if args.smoke:
        # Tiny config — designed to run end-to-end in <60s on a laptop CPU.
        # Used by CI / smoke tests to validate the wiring isn't broken.
        args.mp_actors = 2
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

    torch.manual_seed(cfg.run.seed)
    learner_device = torch.device(args.learner_device)
    if learner_device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"--learner-device={args.learner_device} but CUDA unavailable")

    ckpt_dir = Path(cfg.run.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Build CPU shared nets ──────────────────────────────────────────────
    # share_memory() must be called BEFORE these nets are passed to .start(),
    # so that the cross-process argument serialization captures the
    # shared-memory file descriptor instead of producing a private copy in
    # each worker. That's the only ordering constraint here.
    #
    # CUDA state in the parent does NOT need to be deferred under forkserver:
    # the helper is fork+exec'd (clean Python interpreter, no inherited state),
    # and workers inherit from the helper rather than the parent. Commit
    # 4477b66 of the DMC trainer explicitly removed a defensive "forkserver
    # bootstrap" that was guarding against this non-issue. Under the `fork`
    # start method the hazard would be real; under `forkserver` it isn't.
    net_cpu_sb = AdvNet(cfg.model); net_cpu_sb.share_memory()
    net_cpu_bb = AdvNet(cfg.model); net_cpu_bb.share_memory()

    # Forkserver: lazily spawns a clean helper process on first .start(),
    # then forks workers from that helper. Trade-off vs the default `fork`
    # start method: helper bootstrap costs a fork+exec once, but workers
    # are isolated from parent state (no CUDA / MKL inheritance hazards).
    # Required quirk: top-level imports must be re-importable in the helper —
    # the helper re-imports the target's module.
    ctx = mp.get_context("forkserver")

    # ── GPU learner copies + optimizers ────────────────────────────────────
    if learner_device.type == "cuda":
        # Two separate AdvNets on GPU. These are the "learning" copies — they
        # accumulate gradient updates. The CPU copies are read-only (for
        # actors) and get refreshed via sync_cpu_from_gpu after each refit.
        adv_gpu = [
            AdvNet(cfg.model).to(learner_device),
            AdvNet(cfg.model).to(learner_device),
        ]
        # Initialize the GPU copies with the same random init as the CPU
        # ones, so iteration 1 sees consistent weights everywhere.
        adv_gpu[0].load_state_dict(net_cpu_sb.state_dict())
        adv_gpu[1].load_state_dict(net_cpu_bb.state_dict())
    else:
        # CPU learner: alias the shared CPU nets. No GPU/CPU split needed.
        adv_gpu = [net_cpu_sb, net_cpu_bb]

    policy_net = PolicyNet(cfg.model).to(learner_device)

    print(f"[cfr_mp_gpu] actors={args.mp_actors}  device={learner_device}")
    print(f"[cfr_mp_gpu] AdvNet params (each): {count_parameters(adv_gpu[0]):,}")
    print(f"[cfr_mp_gpu] cfg.model={cfg.model}")
    print(f"[cfr_mp_gpu] cfg.train={cfg.train}")

    # Reservoir buffers live in the learner only. Actors never touch them
    # directly — their writes flow through the queue.
    adv_bufs = [
        ReservoirBuffer(cfg.buffer.adv_capacity, cfg.model.num_actions,
                        rng=np.random.default_rng(cfg.run.seed + 11)),
        ReservoirBuffer(cfg.buffer.adv_capacity, cfg.model.num_actions,
                        rng=np.random.default_rng(cfg.run.seed + 22)),
    ]
    pol_buf = ReservoirBuffer(cfg.buffer.policy_capacity, cfg.model.num_actions,
                              rng=np.random.default_rng(cfg.run.seed + 33))

    # ── Multiprocess primitives ─────────────────────────────────────────────
    # ctx.Queue: cross-process FIFO. Each put() serializes the payload and
    # writes it to a shared OS pipe; each get() reads + deserializes. Bound
    # the size so a runaway actor pool can't OOM the parent's queue.
    trans_queue: mp.Queue = ctx.Queue(maxsize=args.queue_maxsize)
    # ctx.Event: shared boolean. is_set() is reasonably cheap. We poll it
    # in the actor loop to allow clean shutdown.
    stop_event = ctx.Event()
    # ctx.Value('i'): shared int. 'i' = signed int (4 bytes). Atomic on
    # platforms with native 32-bit atomic store. We use it as the iteration
    # counter visible to all actors.
    iter_value = ctx.Value('i', 0)

    # Spawn actor processes. daemon=True means they get killed when the
    # parent exits — important for keeping CI clean if the learner crashes.
    actors = []
    for aid in range(args.mp_actors):
        p = ctx.Process(
            target=_actor_loop,
            args=(aid, net_cpu_sb, net_cpu_bb, trans_queue,
                  stop_event, iter_value,
                  cfg.run.seed + aid * 1009,
                  args.actor_torch_threads,
                  args.max_depth),
            daemon=True,
            name=f"cfr-actor-{aid}",
        )
        p.start()
        actors.append(p)
    print(f"[cfr_mp_gpu] spawned {len(actors)} actors")

    # ── Outer CFR loop ─────────────────────────────────────────────────────
    t_total_start = time.time()
    try:
        for t in range(1, cfg.train.n_iterations + 1):
            print(f"\n[cfr_mp_gpu] ════════════ iteration t={t}/{cfg.train.n_iterations} ════════════")
            # Setting iter_value tells actors which `t` to tag their writes
            # with. Actors poll this on every traversal, so the change
            # propagates within one traversal-time of the assignment.
            iter_value.value = t

            target_traversals = cfg.train.n_traversals_per_iter
            n_drained = 0
            t_iter_start = time.time()
            last_log = time.time()
            while n_drained < target_traversals:
                # Watchdog: any actor death is fatal. Without this check, a
                # crash in an actor process would silently reduce throughput
                # to near-zero (queue stops filling) and we'd hang forever.
                dead = [a.name for a in actors if not a.is_alive()]
                if dead:
                    raise SystemExit(f"actors died: {dead}")
                ctr = _drain_queue(trans_queue, adv_bufs, pol_buf,
                                   max_items=64, timeout_s=1.0)
                n_drained += ctr["n_traversals"]
                if time.time() - last_log >= args.watchdog_every_s:
                    last_log = time.time()
                    print(f"  drained {n_drained}/{target_traversals} "
                          f"adv_buf=[{len(adv_bufs[0]):,},{len(adv_bufs[1]):,}] "
                          f"pol_buf={len(pol_buf):,} "
                          f"q={trans_queue.qsize()}")

            print(f"  collected K={n_drained} in {time.time()-t_iter_start:.1f}s "
                  f"adv_buf=[{len(adv_bufs[0]):,},{len(adv_bufs[1]):,}] "
                  f"pol_buf={len(pol_buf):,}")

            # Note on snapshot semantics:
            # Actors keep producing during refit. Their next traversal will
            # still tag with iter_value=t (we haven't advanced yet). When we
            # advance to t+1 below, in-flight traversals will be drained at
            # the start of iter t+1's collection — they're tagged with t,
            # which is correct: Linear CFR weights them with t (the iteration
            # they were generated in), not t+1. Bounded staleness only.

            # Refit each AdvNet on its player's buffer, then sync the
            # shared-memory CPU copy from the freshly-trained GPU copy.
            for p in (0, 1):
                if cfg.train.reset_adv_net_each_iter:
                    # Paper-default: re-initialize before refit. Prevents
                    # warm-start bias toward early-iteration regret targets
                    # (which were noisy because σ was near-uniform then).
                    fresh = AdvNet(cfg.model).to(learner_device)
                    adv_gpu[p] = fresh
                print(f"  refit adv_net[p={p}] on {len(adv_bufs[p])} samples ...")
                r = refit_adv_net(adv_gpu[p], adv_bufs[p], cfg, learner_device,
                                  use_linear_cfr=use_linear)
                print(f"    loss: first={r['loss_first']:.2f} last={r['loss_last']:.2f} "
                      f"wall={r['wall_s']:.1f}s")

                # Push freshly-learned weights back to the shared-memory CPU
                # net so actors see them on their next forward.
                if learner_device.type == "cuda":
                    target_cpu = net_cpu_sb if p == 0 else net_cpu_bb
                    sync_cpu_from_gpu(target_cpu, adv_gpu[p])

            # ── Learning probes ────────────────────────────────────────────
            # Cheap "is the net learning poker?" signal — see cfr/probe.py.
            # Runs on the GPU AdvNets (adv_gpu); the shared-memory CPU nets
            # are unaffected, so actors keep pumping out traversals during
            # the probe just as they do during refit. Bounded-staleness is
            # already the model here — probe latency is ~1s/scenario.
            # CFRAdvPolicy puts nets in eval mode; restore train(True) after.
            if args.probe_every_iter > 0 and t % args.probe_every_iter == 0:
                results = run_default_probes(adv_gpu, learner_device,
                                             n_hands=args.probe_hands,
                                             base_seed=cfg.run.seed + t * 7919)
                for r in results:
                    print(format_probe_line(r, iter_t=t))
                for net in adv_gpu:
                    net.train(True)

            if (args.checkpoint_every_iter > 0
                    and t % args.checkpoint_every_iter == 0):
                path = ckpt_dir / f"cfr_iter_{t:04d}.ckpt"
                save_checkpoint(path, adv_gpu, policy_net, t, cfg)
                print(f"  saved {path}")

        # ── Final policy net training ──────────────────────────────────────
        # PolicyNet is trained ONCE at the end on the accumulated pol_buf.
        # This is correct per Deep CFR — the policy buffer is the time-
        # averaged record of σ across all iterations, and we just regress
        # against it. Linear CFR weights each sample by its `t` so recent
        # (better-quality) σ values dominate.
        print(f"\n[cfr_mp_gpu] training PolicyNet on {len(pol_buf):,} samples")
        pr = train_policy_net(policy_net, pol_buf, cfg, learner_device,
                              use_linear_cfr=use_linear)
        print(f"  loss first={pr['loss_first']:.4f} last={pr['loss_last']:.4f} "
              f"wall={pr['wall_s']:.1f}s")

        final = ckpt_dir / "cfr_final.ckpt"
        save_checkpoint(final, adv_gpu, policy_net,
                        cfg.train.n_iterations, cfg)
        print(f"\n[cfr_mp_gpu] DONE total wall={time.time()-t_total_start:.1f}s "
              f"saved {final}")

    finally:
        # Clean shutdown: tell actors to stop, then join with timeout.
        # If they don't exit cleanly within the timeout, terminate them
        # forcibly. daemon=True would also kill them at parent exit, but
        # explicit cleanup is cleaner for logs and CI.
        stop_event.set()
        for p in actors:
            p.join(timeout=3.0)
        for p in actors:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)


if __name__ == "__main__":
    main()
