"""Multi-process Deep CFR with a dedicated GPU inference server.

═══════════════════════════════════════════════════════════════════════════════
WHAT MAKES THIS DIFFERENT FROM cfr_mp_gpu.py
═══════════════════════════════════════════════════════════════════════════════

`cfr_mp_gpu.py`: actors do CPU inference using shared-memory CPU AdvNets.
                 Throughput O(N_actors × CPU_forward_time). Breaks at large
                 net sizes because CPU forward becomes the bottleneck.

`cfr_server.py`: actors send (player, x) RPC requests to a dedicated SERVER
                 process which holds AdvNets on GPU and BATCHES requests
                 across actors before each forward. Throughput is bounded
                 by GPU launch overhead amortized across the batch — much
                 higher when the net is large.

SHAPE OF THE SYSTEM:

   ┌── N actor procs (CPU) ──┐    ┌── server proc (GPU 1) ──┐    ┌── learner ─┐
   │  recursive traversal    │ →req│  loop:                 │    │ (main, GPU0) │
   │  on each yield-point:   │     │   accumulate up to     │    │  drains      │
   │     put((id,actor,x)    │     │   BATCH_MAX requests   │    │  result_q,   │
   │       on req_q          │     │   or wait TIMEOUT_S    │ ←sync│  refits      │
   │     block on reply_q    │ ←ans│   batch by player      │  CPU│  AdvNets,    │
   │  on traversal complete: │     │   ONE forward each     │  bus│  bumps wver  │
   │     put result on       │     │   dispatch via reply   │     │              │
   │     result_q            │     │   queues               │     │              │
   └─────────────────────────┘     └────────────────────────┘    └──────────────┘
       N processes                      1 process                    1 process

KEY DESIGN POINTS:

1. PER-ACTOR REPLY QUEUES.
   The request queue is multi-producer-single-consumer (N actors → 1 server).
   The reply path is single-producer-single-consumer per actor: one reply
   queue per actor, owned by the actor, written by the server. This avoids
   needing request IDs for matching — each actor has at most 1 outstanding
   request and just gets the next reply from its queue.

2. BATCHED DISPATCH BY PLAYER.
   Each actor's request includes which AdvNet to query (SB or BB). The server
   maintains TWO pending queues (one per net), batches each independently,
   runs ONE forward per net per round. SB and BB nets don't share weights,
   so they need separate batches.

3. WEIGHT SYNC VIA SHARED-MEMORY CPU MIRRORS.
   The server holds AdvNets on GPU. After learner refits, it copies fresh
   weights into a pair of shared-memory CPU AdvNets. The server periodically
   checks a `weight_version` counter; when it sees the version bump, it
   copies CPU shared → its GPU. This is the same pattern as cfr_mp_gpu.py's
   sync_cpu_from_gpu but with the server (not actors) as the consumer.

4. TWO-GPU TARGET: learner on GPU 0, server on GPU 1.
   Both GPUs run continuously: server during data collection, learner during
   refit. Maximum utilization. If you have only 1 GPU, set both --learner-device
   and --server-device to the same `cuda:0` — works but they'll fight for the
   GPU during collection.

═══════════════════════════════════════════════════════════════════════════════
PERFORMANCE NOTE: IPC BANDWIDTH IS THE LIKELY BOTTLENECK
═══════════════════════════════════════════════════════════════════════════════

Each request payload is ~3.3 KB (816 float32 in `x` + small header). At
target throughput (~100 traversals/sec across all actors × 5000 yields each
= 500k req/sec), that's ~1.6 GB/sec of IPC traffic — which mp.Queue (built
on OS pipes + per-message serialization) can struggle to sustain.

Options when this hurts:
  * Shared-memory ring buffer (skip serialization).
  * Per-actor "request slots" in a shared CUDA tensor (advanced).
  * Increase BATCH_MAX so each forward amortizes more requests.

The single-process coroutine driver in `cfr_coro.py` SIDESTEPS this entirely
because everything is in one address space. For target cluster scale, prefer
that variant unless you specifically need cross-process actor isolation
(e.g., for crash recovery).

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


NUM_ACTIONS = 11

# Sentinel: special request type that tells server "you're done, exit cleanly".
# We send one to the request queue at shutdown. None would also work, but a
# sentinel object is harder to confuse with a legitimate empty request.
_SHUTDOWN = ("__shutdown__",)


# ═══════════════════════════════════════════════════════════════════════════
# SERVER PROCESS
# ═══════════════════════════════════════════════════════════════════════════
#
# The server's job: drain the request queue, batch by player, run forwards,
# dispatch results back to per-actor reply queues. Reload weights from CPU
# shared mirrors when the version counter bumps.
# ═══════════════════════════════════════════════════════════════════════════


def _server_loop(server_device_str: str,
                 model_cfg,
                 net_cpu_sb,             # shared-memory mirror, written by learner
                 net_cpu_bb,
                 req_q: mp.Queue,
                 reply_qs: list,         # one per actor
                 weight_version,         # mp.Value('i') — bumped by learner
                 stop_event,
                 batch_max: int,
                 batch_timeout_s: float) -> None:
    """GPU inference server. Runs until stop_event is set or _SHUTDOWN sentinel.

    BATCH ACCUMULATION POLICY:
        We use a "fill or wait" rule: keep pulling requests until we either
        have BATCH_MAX or BATCH_TIMEOUT seconds have passed since the FIRST
        request. This trades latency (waiting fills more) for throughput
        (bigger batches amortize launch overhead).

        For Deep CFR with N=16 actors, BATCH_MAX=32 is a reasonable default
        since each actor has at most 1 in-flight request, so the natural
        ceiling is N. BATCH_TIMEOUT=0.001s (1ms) keeps tail latency bounded.

    SEPARATE QUEUES FOR SB vs BB:
        We don't pre-allocate two queues; we drain the one shared queue and
        bucket each request into a local list by its `player` field. Then
        we run TWO forwards (one per AdvNet), each over its bucket.

    WEIGHT RELOAD:
        Cheap: just check `weight_version.value` at the top of each round.
        If higher than what we last loaded, reload both nets from the CPU
        mirrors. Reload cost is ~70MB × 2 nets / PCIe ~25GB/s ≈ 6ms. Done
        once per outer iteration, so amortized over thousands of forwards.
    """
    # CUDA init happens INSIDE this process (forkserver guarantees we
    # didn't inherit a poisoned context). We materialize the GPU nets here.
    server_device = torch.device(server_device_str)
    if server_device.type == "cuda" and not torch.cuda.is_available():
        # Process won't be able to do its job. Bail.
        return

    # Construct server-side GPU AdvNets and load initial weights from the
    # CPU mirrors (which the learner pre-populated with random init weights
    # before spawning us).
    adv_gpu = [
        AdvNet(model_cfg).to(server_device),
        AdvNet(model_cfg).to(server_device),
    ]
    adv_gpu[0].load_state_dict(net_cpu_sb.state_dict())
    adv_gpu[1].load_state_dict(net_cpu_bb.state_dict())
    for n in adv_gpu:
        n.train(False)

    cached_version = int(weight_version.value)
    n_batches = 0
    n_requests_total = 0
    t_start = time.time()

    try:
        while not stop_event.is_set():
            # ── Reload weights if version bumped ─────────────────────────
            cur_version = int(weight_version.value)
            if cur_version > cached_version:
                # Weights were updated by the learner. Reload BOTH nets.
                # Use load_state_dict here — the server's GPU net isn't
                # shared with anyone, so we can rebind its parameters
                # freely (unlike the actor case in cfr_mp_gpu.py where
                # rebinding would break shared-memory aliasing).
                adv_gpu[0].load_state_dict(net_cpu_sb.state_dict())
                adv_gpu[1].load_state_dict(net_cpu_bb.state_dict())
                cached_version = cur_version

            # ── Accumulate requests for this batch ───────────────────────
            # Each request is (actor_id: int, player: int, x: np.ndarray).
            # We bucket by `player` because the two AdvNets need separate
            # forwards.
            sb_actor_ids:  list[int]        = []
            sb_xs:         list[np.ndarray] = []
            bb_actor_ids:  list[int]        = []
            bb_xs:         list[np.ndarray] = []

            # Block on first request — we have nothing to do until there is
            # one. After that, drain non-blocking until we have BATCH_MAX
            # OR the BATCH_TIMEOUT elapses since the first arrival.
            try:
                payload = req_q.get(timeout=0.1)
            except pyqueue.Empty:
                continue
            if payload == _SHUTDOWN:
                break

            actor_id, player, x = payload
            (sb_xs if player == 0 else bb_xs).append(x)
            (sb_actor_ids if player == 0 else bb_actor_ids).append(actor_id)
            batch_deadline = time.time() + batch_timeout_s

            while (len(sb_xs) + len(bb_xs)) < batch_max:
                remaining = max(0.0, batch_deadline - time.time())
                try:
                    payload = req_q.get(timeout=remaining)
                except pyqueue.Empty:
                    break
                if payload == _SHUTDOWN:
                    stop_event.set()
                    break
                actor_id, player, x = payload
                (sb_xs if player == 0 else bb_xs).append(x)
                (sb_actor_ids if player == 0 else bb_actor_ids).append(actor_id)

            # ── Run forwards ──────────────────────────────────────────────
            # Each net gets one forward over its bucket. If a bucket is
            # empty (e.g., all current requests are for SB), we skip it.
            if sb_xs:
                sb_batch = torch.from_numpy(
                    np.stack(sb_xs, axis=0)
                ).to(server_device, non_blocking=True)
                with torch.no_grad():
                    sb_out = adv_gpu[0](sb_batch).detach().cpu().numpy()
                # Dispatch each row to its actor's reply queue.
                for aid, row in zip(sb_actor_ids, sb_out):
                    # .copy() detaches from the batch ndarray's lifetime —
                    # otherwise the whole batch ndarray stays alive in the
                    # actor's reply queue until the actor consumes its row.
                    reply_qs[aid].put(np.ascontiguousarray(row,
                                                           dtype=np.float32))
                n_requests_total += len(sb_xs)

            if bb_xs:
                bb_batch = torch.from_numpy(
                    np.stack(bb_xs, axis=0)
                ).to(server_device, non_blocking=True)
                with torch.no_grad():
                    bb_out = adv_gpu[1](bb_batch).detach().cpu().numpy()
                for aid, row in zip(bb_actor_ids, bb_out):
                    reply_qs[aid].put(np.ascontiguousarray(row,
                                                           dtype=np.float32))
                n_requests_total += len(bb_xs)

            n_batches += 1

    except (KeyboardInterrupt, SystemExit):
        return
    finally:
        # Optional: print server-side throughput so you can see how big the
        # actual batches end up being. Useful for tuning batch_max / timeout.
        wall = time.time() - t_start
        if n_batches > 0:
            avg = n_requests_total / n_batches
            print(f"[server] processed {n_requests_total} requests in "
                  f"{n_batches} batches (avg B={avg:.1f}) over {wall:.1f}s "
                  f"= {n_requests_total/max(1,wall):.0f} req/s")


# ═══════════════════════════════════════════════════════════════════════════
# ACTOR PROCESS
# ═══════════════════════════════════════════════════════════════════════════
#
# Actors do CPU work only — recursion bookkeeping, regret matching, sampling.
# Every inference is an RPC to the server: put a request on req_q, block
# on reply_q, get back the regret prediction.
# ═══════════════════════════════════════════════════════════════════════════


def _legal_mask_from_obs(obs) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for at in obs.legal:
        mask[int(at)] = 1.0
    return mask


def _request_inference(actor_id: int,
                       player: int,
                       x: np.ndarray,
                       req_q: mp.Queue,
                       reply_q: mp.Queue) -> np.ndarray:
    """RPC to the server: request a forward, block on the reply.

    The actor's recursion is single-threaded — at any moment, an actor has
    at most ONE in-flight request. So .get() on its private reply queue
    will return THIS request's result (no need for request IDs to match
    replies to requests).

    Why no timeout on get()? If the server hangs, we WANT the actor to
    block forever — the parent process's watchdog will see the actor stop
    making progress and crash the whole run. A timeout that triggered on
    a slow batch would corrupt the protocol (the late reply would be
    consumed by the next request).
    """
    req_q.put((actor_id, player, x))
    return reply_q.get()


def _traverse_via_server(env,
                         traverser: int,
                         actor_id: int,
                         req_q: mp.Queue,
                         reply_q: mp.Queue,
                         t: int,
                         rng: np.random.Generator,
                         adv_writes_out: list,
                         pol_writes_out: list,
                         max_depth: int = DEFAULT_MAX_DEPTH):
    """Recursive traversal that delegates inference to the GPU server.

    Same algorithm as `traversal.traverse` and `cfr_mp_gpu._traverse_collect`,
    but every inference is an RPC. Each RPC blocks the caller for one batch's
    worth of time on the server (typically <1ms).
    """
    if env.is_terminal():
        return float(env.payoffs_bb()[traverser])

    obs   = env.observation()
    actor = int(env.to_act())
    x     = obs.x.astype(np.float32, copy=False).copy()
    legal = list(obs.legal)
    n_legal = len(legal)
    mask  = _legal_mask_from_obs(obs)

    # ── RPC TO SERVER ──────────────────────────────────────────────────
    # Send request, block on reply. The server batches this with other
    # actors' pending requests and runs one GPU forward.
    pred_r = _request_inference(actor_id, actor, x, req_q, reply_q)
    sigma  = regret_matching_np(pred_r, mask)
    pol_writes_out.append((x, sigma))

    if env.state().history_size >= max_depth:
        return float((sigma * pred_r * mask).sum()) * REGRET_SCALE

    if actor == traverser:
        action_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for at in legal:
            child = env.clone()
            child.step_action(at)
            v_a = _traverse_via_server(child, traverser, actor_id,
                                       req_q, reply_q, t, rng,
                                       adv_writes_out, pol_writes_out,
                                       max_depth=max_depth)
            action_values[int(at)] = v_a
        v_state = float((sigma * action_values).sum())
        regrets = (action_values - v_state) * mask / REGRET_SCALE
        adv_writes_out.append((x, regrets.astype(np.float32, copy=False)))
        return v_state

    # Opponent: external sampling.
    legal_int = np.array([int(at) for at in legal], dtype=np.int64)
    legal_probs = sigma[legal_int]
    z = legal_probs.sum()
    if z > 0.0:
        legal_probs = legal_probs / z
    else:
        legal_probs = np.full(n_legal, 1.0 / n_legal, dtype=np.float32)
    chosen_local_idx = int(rng.choice(n_legal, p=legal_probs))
    env.step(chosen_local_idx)
    return _traverse_via_server(env, traverser, actor_id,
                                req_q, reply_q, t, rng,
                                adv_writes_out, pol_writes_out,
                                max_depth=max_depth)


def _actor_loop(actor_id: int,
                req_q: mp.Queue,
                reply_q: mp.Queue,        # private reply queue for THIS actor
                result_q: mp.Queue,       # shared output channel to learner
                stop_event,
                iter_value,
                base_seed: int,
                torch_threads: int,
                max_depth: int) -> None:
    """Actor: run traversals forever, requesting inferences via RPC.

    NOTE: torch.set_num_threads(1) is still important here even though we
    don't do any local forwards. The PyTorch tensor creation in
    `np.stack` → `torch.from_numpy` paths can sometimes invoke OpenMP
    work; capping at 1 keeps process-level parallelism clean.
    """
    torch.set_num_threads(max(1, int(torch_threads)))
    env = pte.Env((base_seed ^ (actor_id * 0x9E3779B1)) & 0xFFFFFFFFFFFFFFFF)
    rng = np.random.default_rng(base_seed + actor_id * 7919)

    try:
        while not stop_event.is_set():
            t = int(iter_value.value)
            if t <= 0:
                stop_event.wait(0.05)
                continue

            traverser = int(rng.integers(0, 2))
            env.reset()
            adv_writes: list = []
            pol_writes: list = []
            _traverse_via_server(env, traverser, actor_id,
                                 req_q, reply_q, t, rng,
                                 adv_writes, pol_writes,
                                 max_depth=max_depth)

            try:
                result_q.put((traverser, adv_writes, pol_writes, t),
                             timeout=1.0)
            except pyqueue.Full:
                pass
    except (KeyboardInterrupt, SystemExit):
        return


# ═══════════════════════════════════════════════════════════════════════════
# WEIGHT SYNC HELPERS — same pattern as cfr_mp_gpu.py
# ═══════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def sync_cpu_from_gpu(net_cpu, net_gpu) -> None:
    """In-place copy of every named parameter / buffer from GPU to CPU mirror.

    Same rationale as cfr_mp_gpu.sync_cpu_from_gpu: we need copy_() (not
    load_state_dict) to preserve shared-memory aliasing on the CPU side.
    The server then reloads from this CPU mirror when its weight_version
    cache becomes stale.
    """
    cpu_params = dict(net_cpu.named_parameters())
    cpu_buffers = dict(net_cpu.named_buffers())
    for name, p in net_gpu.named_parameters():
        cpu_params[name].copy_(p.detach(), non_blocking=False)
    for name, b in net_gpu.named_buffers():
        if name in cpu_buffers:
            cpu_buffers[name].copy_(b.detach(), non_blocking=False)


# ═══════════════════════════════════════════════════════════════════════════
# LEARNER-SIDE QUEUE DRAINING — identical contract to cfr_mp_gpu.py
# ═══════════════════════════════════════════════════════════════════════════


def _drain_queue(result_q: mp.Queue,
                 adv_bufs: list[ReservoirBuffer],
                 pol_buf: ReservoirBuffer,
                 max_items: int,
                 timeout_s: float) -> dict:
    counters = {"n_traversals": 0,
                "adv_inserts": [0, 0],
                "pol_inserts": 0}
    deadline = time.time() + timeout_s
    items = 0
    while items < max_items:
        remaining = max(0.0, deadline - time.time())
        try:
            payload = result_q.get(timeout=remaining if items == 0 else 0.0)
        except pyqueue.Empty:
            break
        traverser, adv_writes, pol_writes, t = payload
        for x, regrets in adv_writes:
            adv_bufs[traverser].add(x, regrets, t)
            counters["adv_inserts"][traverser] += 1
        for x, sigma in pol_writes:
            pol_buf.add(x, sigma, t)
            counters["pol_inserts"] += 1
        counters["n_traversals"] += 1
        items += 1
    return counters


# ═══════════════════════════════════════════════════════════════════════════
# CLI / MAIN
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mp-actors", type=int, default=16)
    p.add_argument("--queue-maxsize", type=int, default=4096)
    p.add_argument("--learner-device", type=str, default="cuda:0",
                   help="device for AdvNet refit + PolicyNet training")
    p.add_argument("--server-device", type=str, default="cuda:1",
                   help="device for inference server (use a different GPU "
                        "than --learner-device for max throughput)")
    p.add_argument("--actor-torch-threads", type=int, default=1)
    p.add_argument("--n-iterations", type=int, default=100)
    p.add_argument("--n-traversals-per-iter", type=int, default=5000)
    p.add_argument("--adv-grad-steps", type=int, default=4000)
    p.add_argument("--policy-grad-steps", type=int, default=50000)
    p.add_argument("--adv-capacity", type=int, default=2_000_000)
    p.add_argument("--policy-capacity", type=int, default=4_000_000)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--no-linear-cfr", action="store_true")
    p.add_argument("--ckpt-dir", type=str, default="runs/cfr_server_latest")
    p.add_argument("--checkpoint-every-iter", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--watchdog-every-s", type=float, default=10.0)
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    p.add_argument("--batch-max", type=int, default=32,
                   help="server's max batch size before forcing a forward")
    p.add_argument("--batch-timeout-ms", type=float, default=1.0,
                   help="server's max wait time after first request, before "
                        "running a batch even if not full")
    p.add_argument("--smoke", action="store_true")
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
    cfg.model.hidden = args.hidden
    cfg.model.n_layers = args.n_layers
    use_linear = not args.no_linear_cfr

    if args.smoke:
        args.mp_actors = 2
        cfg.train.n_iterations = 2
        cfg.train.n_traversals_per_iter = 20
        cfg.optim.n_grad_steps_per_refit = 200
        cfg.train.policy_n_grad_steps = 500
        cfg.buffer.adv_capacity = 10_000
        cfg.buffer.policy_capacity = 10_000
        cfg.model.hidden = 128
        cfg.model.n_layers = 2
        args.checkpoint_every_iter = 1
        args.batch_max = 4

    torch.manual_seed(cfg.run.seed)
    learner_device = torch.device(args.learner_device)
    if learner_device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"--learner-device={args.learner_device} but CUDA unavailable")

    ckpt_dir = Path(cfg.run.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Build CPU mirror nets ──────────────────────────────────────────────
    # share_memory() must run before the nets are passed to .start() so each
    # worker receives the shared-memory handle (not a private copy).
    #
    # No CUDA-ordering discipline needed: forkserver fork+exec's a clean
    # helper, workers inherit from the helper, parent's CUDA state never
    # propagates. The server worker initializes its own CUDA context inside
    # itself when it constructs its GPU AdvNets and loads weights from
    # these mirrors at startup.
    net_cpu_sb = AdvNet(cfg.model); net_cpu_sb.share_memory()
    net_cpu_bb = AdvNet(cfg.model); net_cpu_bb.share_memory()

    ctx = mp.get_context("forkserver")

    # ── IPC primitives ──────────────────────────────────────────────────────
    # One shared request queue (all actors put → server gets).
    req_q: mp.Queue = ctx.Queue(maxsize=args.queue_maxsize)
    # Per-actor reply queues. Each actor gets its own. The server holds all
    # of them and writes to whichever the request originated from.
    reply_qs = [ctx.Queue(maxsize=2) for _ in range(args.mp_actors)]
    # Result queue: actors put completed traversal payloads; learner drains.
    result_q: mp.Queue = ctx.Queue(maxsize=args.queue_maxsize)

    stop_event = ctx.Event()
    iter_value = ctx.Value('i', 0)
    # weight_version: incremented by learner after each (CPU sync of) refit.
    # Server caches its last-loaded version and reloads when the value
    # increases. Initial value 0 = "use whatever was loaded at startup".
    weight_version = ctx.Value('i', 0)

    # ── Spawn server FIRST (before CUDA init in main) ──────────────────────
    server_proc = ctx.Process(
        target=_server_loop,
        args=(args.server_device,
              cfg.model,
              net_cpu_sb, net_cpu_bb,
              req_q, reply_qs,
              weight_version,
              stop_event,
              args.batch_max,
              args.batch_timeout_ms / 1000.0),
        daemon=True,
        name="cfr-server",
    )
    server_proc.start()
    print(f"[cfr_server] spawned inference server on {args.server_device}")

    # ── Spawn actor processes ──────────────────────────────────────────────
    actors = []
    for aid in range(args.mp_actors):
        p = ctx.Process(
            target=_actor_loop,
            args=(aid, req_q, reply_qs[aid], result_q,
                  stop_event, iter_value,
                  cfg.run.seed + aid * 1009,
                  args.actor_torch_threads,
                  args.max_depth),
            daemon=True,
            name=f"cfr-actor-{aid}",
        )
        p.start()
        actors.append(p)
    print(f"[cfr_server] spawned {len(actors)} actors")

    # ── NOW it's safe to init CUDA in main (for learner) ───────────────────
    if learner_device.type == "cuda":
        adv_gpu = [
            AdvNet(cfg.model).to(learner_device),
            AdvNet(cfg.model).to(learner_device),
        ]
        adv_gpu[0].load_state_dict(net_cpu_sb.state_dict())
        adv_gpu[1].load_state_dict(net_cpu_bb.state_dict())
    else:
        adv_gpu = [net_cpu_sb, net_cpu_bb]

    policy_net = PolicyNet(cfg.model).to(learner_device)

    print(f"[cfr_server] learner_device={learner_device}")
    print(f"[cfr_server] AdvNet params (each): {count_parameters(adv_gpu[0]):,}")
    print(f"[cfr_server] cfg.model={cfg.model}")
    print(f"[cfr_server] cfg.train={cfg.train}")

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

    # ── Outer CFR loop ─────────────────────────────────────────────────────
    t_total_start = time.time()
    try:
        for t in range(1, cfg.train.n_iterations + 1):
            print(f"\n[cfr_server] ════════════ iteration t={t}/{cfg.train.n_iterations} ════════════")
            iter_value.value = t

            target_traversals = cfg.train.n_traversals_per_iter
            n_drained = 0
            t_iter_start = time.time()
            last_log = time.time()
            while n_drained < target_traversals:
                # Watchdog covers actors AND server. If the server crashes,
                # all actors will hang on reply_q.get() and never produce
                # results — we'd hang forever without this check.
                dead_actors = [a.name for a in actors if not a.is_alive()]
                if dead_actors:
                    raise SystemExit(f"actors died: {dead_actors}")
                if not server_proc.is_alive():
                    raise SystemExit("server process died")

                ctr = _drain_queue(result_q, adv_bufs, pol_buf,
                                   max_items=64, timeout_s=1.0)
                n_drained += ctr["n_traversals"]
                if time.time() - last_log >= args.watchdog_every_s:
                    last_log = time.time()
                    print(f"  drained {n_drained}/{target_traversals} "
                          f"adv_buf=[{len(adv_bufs[0]):,},{len(adv_bufs[1]):,}] "
                          f"pol_buf={len(pol_buf):,} "
                          f"req_q={req_q.qsize()} result_q={result_q.qsize()}")

            print(f"  collected K={n_drained} in {time.time()-t_iter_start:.1f}s "
                  f"adv_buf=[{len(adv_bufs[0]):,},{len(adv_bufs[1]):,}] "
                  f"pol_buf={len(pol_buf):,}")

            # ── Refit AdvNets, sync to CPU mirrors, bump version ───────────
            for p in (0, 1):
                if cfg.train.reset_adv_net_each_iter:
                    fresh = AdvNet(cfg.model).to(learner_device)
                    adv_gpu[p] = fresh
                print(f"  refit adv_net[p={p}] on {len(adv_bufs[p])} samples ...")
                r = refit_adv_net(adv_gpu[p], adv_bufs[p], cfg, learner_device,
                                  use_linear_cfr=use_linear)
                print(f"    loss: first={r['loss_first']:.2f} last={r['loss_last']:.2f} "
                      f"wall={r['wall_s']:.1f}s")

                # Push to CPU mirror.
                if learner_device.type == "cuda":
                    target_cpu = net_cpu_sb if p == 0 else net_cpu_bb
                    sync_cpu_from_gpu(target_cpu, adv_gpu[p])

            # Bump version AFTER both nets are synced, so the server reload
            # picks up a consistent (SB, BB) pair, never half-updated.
            with weight_version.get_lock():
                weight_version.value += 1

            if (args.checkpoint_every_iter > 0
                    and t % args.checkpoint_every_iter == 0):
                path = ckpt_dir / f"cfr_iter_{t:04d}.ckpt"
                save_checkpoint(path, adv_gpu, policy_net, t, cfg)
                print(f"  saved {path}")

        # ── Final policy net training ──────────────────────────────────────
        print(f"\n[cfr_server] training PolicyNet on {len(pol_buf):,} samples")
        pr = train_policy_net(policy_net, pol_buf, cfg, learner_device,
                              use_linear_cfr=use_linear)
        print(f"  loss first={pr['loss_first']:.4f} last={pr['loss_last']:.4f} "
              f"wall={pr['wall_s']:.1f}s")

        final = ckpt_dir / "cfr_final.ckpt"
        save_checkpoint(final, adv_gpu, policy_net,
                        cfg.train.n_iterations, cfg)
        print(f"\n[cfr_server] DONE total wall={time.time()-t_total_start:.1f}s "
              f"saved {final}")

    finally:
        # ── Clean shutdown sequence ────────────────────────────────────────
        # 1. Stop actors (so they don't keep producing requests).
        # 2. Send shutdown sentinel to server (so it exits its blocking get).
        # 3. Join all processes with bounded timeouts.
        stop_event.set()
        try:
            req_q.put(_SHUTDOWN, timeout=1.0)
        except pyqueue.Full:
            pass

        for p in actors:
            p.join(timeout=3.0)
        server_proc.join(timeout=3.0)

        for p in actors + [server_proc]:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)


if __name__ == "__main__":
    main()
