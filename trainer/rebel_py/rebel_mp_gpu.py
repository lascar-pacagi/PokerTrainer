"""Multi-process ReBeL trainer: CPU actor pool + GPU learner.
ReBeL idea → scaling self-play to a cluster node (the test-time loop is the same
machinery minus the buffer writes) · doc §9.

Mirrors ``trainer/cfr/cfr_mp_gpu.py`` (read its header for the rationale of the
forkserver + shared-memory-net + copy_-sync design). The ReBeL-specific shape:

   ┌── N actor processes (CPU) ──┐        ┌── learner (main process, GPU) ─────┐
   │ each runs ReBeL self-play   │  ──Q──▶│ owns the GPU value net + optimizer │
   │ (RlRunner) with a shared-   │ (query,│ + reservoir. Drains queue, trains, │
   │ memory CPU value net; one   │ value) │ syncs the shared CPU net via copy_,│
   │ game = one queue payload of │ lists  │ evals exploitability, checkpoints  │
   │ (query,value) examples.     │        │ (+ TorchScript).                   │
   └─────────────────────────────┘        └────────────────────────────────────┘

Two games (``--game``):
  * ``river`` — Phase 2a (betting-depth limit, no chance): cheap, light memory.
  * ``turn``  — Phase 2b (turn→river, one chance node): the cross-street net.
    Each turn tree carries 48 per-card showdown matrices (~0.3 GB), so keep the
    per-actor board pool small until the river-card vectorization lands.

WHY CPU INFERENCE IN ACTORS: the value net is queried at batch≈(#net leaves)
inside a sequential CFR solve; tiny batched forwards are dominated by GPU launch
overhead, and actors share the net cheaply via shared memory (see cfr_mp_gpu).
"""
from __future__ import annotations

import os as _os
# Pin BLAS to one thread BEFORE numpy/torch load — in this process AND in the
# forkserver helper (which re-imports this module). A multi-threaded OpenBLAS
# pool does not survive process spawn: actors forked off such a pool segfault on
# their first matvec. (rebel_py/__init__ also sets these, but numpy is imported
# right below, so set them here too to guarantee ordering.)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import queue as pyqueue
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from .pbs import query_dim
from .models import ValueNet, NetValueFn
from .replay import QueryValueBuffer
from .recursive import RlRunner, recursive_strategy, SubgameParams, uniform_beliefs
from .exploitability import exploitability_bb
from .hand_index import NUM_HANDS


# ── tree builders per game ──────────────────────────────────────────────────

def _build_tree(game: str, seed: int, stack_bb: int, actions):
    from .public_tree import build_river_subgame, build_turn_subgame
    if game == "river":
        return build_river_subgame(seed=seed, starting_stack_bb=stack_bb,
                                   allowed_actions=actions)
    if game == "turn":
        return build_turn_subgame(seed=seed, starting_stack_bb=stack_bb,
                                  allowed_actions=actions)
    raise ValueError(f"unknown game {game!r}")


def _params_for(game: str, cfr_iters: int, rap: float) -> SubgameParams:
    if game == "river":
        return SubgameParams(depth_limit=2, num_iters=cfr_iters,
                             random_action_prob=rap, stop_at_chance=False)
    return SubgameParams(depth_limit=None, num_iters=cfr_iters,
                         random_action_prob=rap, stop_at_chance=True)


class _Collector:
    """A buffer-shaped sink: RlRunner calls .add(q, v); we ship the list."""
    __slots__ = ("items",)

    def __init__(self):
        self.items = []

    def add(self, q, v):
        self.items.append((q, v))


# ═══════════════════════════════════════════════════════════════════════════
# ACTOR PROCESS
# ═══════════════════════════════════════════════════════════════════════════

def _actor_loop(actor_id, net_cpu, trans_queue, stop_event, started_event,
                game, board_seeds, stack_bb, actions, cfr_iters, rap,
                base_seed, torch_threads, qdim, n_hidden, n_layers):
    torch.set_num_threads(max(1, int(torch_threads)))
    rng = np.random.default_rng(base_seed)
    # Per-actor pool of trees (one per board). Built once; sampled per game.
    trees = [_build_tree(game, s, stack_bb, actions) for s in board_seeds]
    # PRIVATE net: the actor forwards through its own copy, refreshed from the
    # shared net only BETWEEN games. This is the critical safety property — a
    # forward must never read a parameter the learner is mid-copy_ into (that
    # concurrent write+read on the same storage faults). The refresh itself may
    # read torn weights (bounded staleness), but it is never concurrent with a
    # forward on the same tensor.
    private = ValueNet(qdim, n_hidden=n_hidden, n_layers=n_layers)
    vfn = NetValueFn(private, "cpu")
    params = _params_for(game, cfr_iters, rap)
    started_event.set()
    import traceback, tempfile, os
    dumped = False
    try:
        while not stop_event.is_set():
            _refresh_from(private, net_cpu)        # safe point: between games
            # Per-game resilience: a rare numerical edge case in one sampled
            # path must not kill the actor (and trip the learner's watchdog).
            # Record the first traceback for diagnosis, then skip the game.
            try:
                col = _Collector()
                tree = trees[int(rng.integers(len(trees)))]
                RlRunner(tree, vfn, params, col, rng).step()
            except Exception:
                if not dumped:
                    with open(os.path.join(tempfile.gettempdir(),
                                           f"rebel_actor_{actor_id}.err"), "w") as fh:
                        traceback.print_exc(file=fh)
                    dumped = True
                continue
            try:
                trans_queue.put(col.items, timeout=1.0)
            except pyqueue.Full:
                pass
    except (KeyboardInterrupt, SystemExit):
        return


# ═══════════════════════════════════════════════════════════════════════════
# WEIGHT SYNC (copy_ preserves the shared-memory storage; see cfr_mp_gpu)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _refresh_from(dst, src) -> None:
    """Copy src→dst params/buffers (actor pulls the shared net into its private
    one). May read torn weights; never runs concurrent with a forward on dst."""
    dst_p = dict(dst.named_parameters()); dst_b = dict(dst.named_buffers())
    for name, p in src.named_parameters():
        dst_p[name].copy_(p.detach())
    for name, b in src.named_buffers():
        if name in dst_b:
            dst_b[name].copy_(b.detach())


@torch.no_grad()
def sync_cpu_from_gpu(net_cpu, net_gpu) -> None:
    cpu_params = dict(net_cpu.named_parameters())
    cpu_buffers = dict(net_cpu.named_buffers())
    for name, p in net_gpu.named_parameters():
        cpu_params[name].copy_(p.detach(), non_blocking=False)
    for name, b in net_gpu.named_buffers():
        if name in cpu_buffers:
            cpu_buffers[name].copy_(b.detach(), non_blocking=False)


def _drain_queue(trans_queue, buf, max_items, timeout_s) -> int:
    deadline = time.time() + timeout_s
    n_examples = 0
    items = 0
    while items < max_items:
        remaining = max(0.0, deadline - time.time())
        try:
            payload = trans_queue.get(timeout=remaining if items == 0 else 0.0)
        except pyqueue.Empty:
            break
        for q, v in payload:
            buf.add(q, v)
            n_examples += 1
        items += 1
    return n_examples


# ═══════════════════════════════════════════════════════════════════════════
# LEARNER / MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--game", choices=("river", "turn"), default="turn")
    p.add_argument("--mp-actors", type=int, default=8)
    p.add_argument("--queue-maxsize", type=int, default=2048)
    p.add_argument("--learner-device", type=str, default="cuda:0")
    p.add_argument("--actor-torch-threads", type=int, default=1)
    p.add_argument("--n-boards", type=int, default=4,
                   help="board pool size per actor (turn trees are ~0.3 GB each)")
    p.add_argument("--stack-bb", type=int, default=6)
    p.add_argument("--cfr-iters", type=int, default=256,
                   help="CFR iters per subgame solve (sets value-target accuracy)")
    p.add_argument("--random-action-prob", type=float, default=0.25)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--examples-per-epoch", type=int, default=4000)
    p.add_argument("--batches-per-epoch", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--buffer-cap", type=int, default=2_000_000)
    p.add_argument("--n-hidden", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-iters", type=int, default=256)
    p.add_argument("--eval-boards", type=int, default=2)
    p.add_argument("--ckpt-dir", type=str, default="runs/rebel_mp_gpu_latest")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--watchdog-every-s", type=float, default=15.0)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def _evaluate(trees, vfn, game, eval_iters):
    if game == "river":
        params = SubgameParams(depth_limit=2, num_iters=eval_iters,
                               random_action_prob=0.0, stop_at_chance=False)
    else:
        params = SubgameParams(depth_limit=None, num_iters=eval_iters,
                               random_action_prob=0.0, stop_at_chance=True)
    es = [exploitability_bb(t, recursive_strategy(t, vfn, params),
                            uniform_beliefs(t.board)) for t in trees]
    return float(np.mean(es))


def main():
    args = parse_args()
    actions = (0, 1, 10)
    if args.smoke:
        args.mp_actors = 2; args.n_boards = 1; args.n_epochs = 3
        args.examples_per_epoch = 200; args.batches_per_epoch = 16
        args.cfr_iters = 48; args.eval_every = 1; args.eval_iters = 64
        args.n_hidden = 64; args.n_layers = 2; args.checkpoint_every = 1
        args.learner_device = "cpu"

    torch.manual_seed(args.seed)
    device = torch.device(args.learner_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"--learner-device={args.learner_device} but CUDA unavailable")
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    qdim = query_dim()

    # Shared-memory CPU net for actors (share_memory BEFORE spawning).
    net_cpu = ValueNet(qdim, n_hidden=args.n_hidden, n_layers=args.n_layers)
    net_cpu.share_memory()
    ctx = mp.get_context("forkserver")

    # Learner net is ALWAYS a separate object from the shared actor net (even on
    # CPU): the optimizer mutates it in place, and actors must never read a net
    # mid-step. The only write actors observe is the periodic copy_ sync (a
    # per-tensor memcpy → bounded staleness, never a torn-realloc crash).
    net_gpu = ValueNet(qdim, n_hidden=args.n_hidden, n_layers=args.n_layers).to(device)
    net_gpu.load_state_dict(net_cpu.state_dict())
    opt = torch.optim.Adam(net_gpu.parameters(), lr=args.lr)
    lossf = torch.nn.SmoothL1Loss()
    buf = QueryValueBuffer(args.buffer_cap, qdim, NUM_HANDS, seed=args.seed)

    # Board seeds: a shared pool across actors (the net learns these boards).
    board_seeds = [args.seed * 100 + i for i in range(args.n_boards)]
    print(f"[rebel_mp_gpu] game={args.game} actors={args.mp_actors} device={device} "
          f"boards={args.n_boards} stack={args.stack_bb}bb cfr_iters={args.cfr_iters}")

    trans_queue = ctx.Queue(maxsize=args.queue_maxsize)
    stop_event = ctx.Event()

    def _spawn(aid):
        ev = ctx.Event()
        p = ctx.Process(target=_actor_loop, name=f"rebel-actor-{aid}", daemon=True,
                        args=(aid, net_cpu, trans_queue, stop_event, ev, args.game,
                              board_seeds, args.stack_bb, actions, args.cfr_iters,
                              args.random_action_prob, args.seed + aid * 1009,
                              args.actor_torch_threads, qdim, args.n_hidden, args.n_layers))
        p.start()
        return p, ev

    actors, started = [], []
    for aid in range(args.mp_actors):
        p, ev = _spawn(aid); actors.append(p); started.append(ev)
    for ev in started:                      # wait for actors to build their pools
        ev.wait(timeout=600)
    print(f"[rebel_mp_gpu] {len(actors)} actors ready")

    # Learner-side eval trees (the actors' first boards).
    eval_trees = [_build_tree(args.game, s, args.stack_bb, actions)
                  for s in board_seeds[:max(1, args.eval_boards)]]
    eval_vfn = NetValueFn(net_cpu, "cpu")

    t_start = time.time()
    try:
        print(f"epoch  0  exploit={_evaluate(eval_trees, eval_vfn, args.game, args.eval_iters):.5f} "
              f"bb/hand  (untrained)")
        last = float("nan")
        for epoch in range(1, args.n_epochs + 1):
            got = 0; last_log = time.time()
            while got < args.examples_per_epoch:
                # An actor may die on a rare C-level fault; respawn it rather
                # than abort the (possibly days-long) run.
                for i, a in enumerate(actors):
                    if not a.is_alive():
                        print(f"  [watchdog] respawning dead {a.name}")
                        actors[i], _ = _spawn(i)
                got += _drain_queue(trans_queue, buf, max_items=64, timeout_s=1.0)
                if time.time() - last_log >= args.watchdog_every_s:
                    last_log = time.time()
                    print(f"  draining {got}/{args.examples_per_epoch} buf={len(buf):,} "
                          f"q={trans_queue.qsize()}")
            net_gpu.train()
            for _ in range(args.batches_per_epoch):
                q, v = buf.sample(args.batch_size)
                loss = lossf(net_gpu(torch.as_tensor(q, device=device)),
                             torch.as_tensor(v, device=device))
                opt.zero_grad(); loss.backward(); opt.step(); last = float(loss.item())
            sync_cpu_from_gpu(net_cpu, net_gpu)       # actors see new weights

            if epoch % args.eval_every == 0 or epoch == args.n_epochs:
                e = _evaluate(eval_trees, eval_vfn, args.game, args.eval_iters)
                print(f"epoch {epoch:>3d}  buf={len(buf):>8,}  loss={last:.4f}  "
                      f"exploit={e:.5f} bb/hand  ({time.time()-t_start:.0f}s)")
            if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
                _save(net_gpu, qdim, ckpt_dir / f"rebel_{args.game}_{epoch:04d}.pt", args)
        _save(net_gpu, qdim, ckpt_dir / f"rebel_{args.game}_final.pt", args)
        print(f"[rebel_mp_gpu] DONE wall={time.time()-t_start:.0f}s")
    finally:
        stop_event.set()
        for p in actors:
            p.join(timeout=3.0)
        for p in actors:
            if p.is_alive():
                p.terminate(); p.join(timeout=1.0)


def _save(net, qdim, path: Path, args) -> None:
    """state_dict + a TorchScript trace (for a future C++/FFI consumer). Saves a
    fresh CPU copy so the GPU training net is never moved off-device mid-run."""
    cpu = ValueNet(qdim, n_hidden=args.n_hidden, n_layers=args.n_layers)
    cpu.load_state_dict({k: v.detach().cpu() for k, v in net.state_dict().items()})
    cpu.eval()
    torch.save({"state_dict": cpu.state_dict(), "query_dim": qdim,
                "game": args.game, "n_hidden": args.n_hidden,
                "n_layers": args.n_layers}, path)
    ts = torch.jit.trace(cpu, torch.zeros(1, qdim))
    ts.save(str(path) + ".ts")


if __name__ == "__main__":
    main()
