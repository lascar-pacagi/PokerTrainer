"""Multi-process DMC training: CPU actors + ONE GPU learner.

Architecture:
    N actor processes ── mp.Queue ──▶ Learner (main process)
       (read net_cpu                         │
        in shared memory,                    ├─▶ MCBuffer
        local env+rng)                       ├─▶ learn_step(net_gpu, ...) on CUDA
                                             ├─▶ sync net_cpu ⟵ net_gpu every N steps
                                             └─▶ periodic eval + ckpt

Two networks live in the parent process:
  * net_cpu  — allocated on CPU, share_memory_()'d, read by every actor for
               inference. Actors never touch CUDA.
  * net_gpu  — separate copy on the chosen CUDA device. The optimizer is
               built on this one. After every --actor-sync-every grad
               steps, in-place copy_() pushes net_gpu's parameters into
               net_cpu's shared-memory tensors. Atomic per-tensor; actors
               see eventually consistent weights (same property they'd
               have under all-CPU mp; DMC MC-target variance swamps the
               staleness).

Why forkserver and not fork: Python's forkserver spawns its helper via
fork+exec (the exec replaces the process image, so the helper boots up
clean regardless of whether the parent had CUDA initialised). Worker
actors are then forked from the helper — never directly from this
process — so they never inherit CUDA state. PyTorch officially endorses
forkserver as the CUDA-safe start method.

CLI:
    python -m dmc.dmc_mp_gpu --mp-actors 8 --learner-device cuda:0 \
                             --actor-sync-every 16

This module is the GPU-learner variant of dmc_mp.py. The CPU-only
multi-process trainer in dmc_mp.py is kept as-is.
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

from .actor import PlayerTrajectory, choose_action, schedule_epsilon, schedule_reward_clip
from .buffers import MCBuffer
from .config import DMCConfig
from .learner import build_optimizer, learn_step
from .models import DMCNet, count_parameters

from evaluate.match import play_match
from evaluate.policies import (
    CallingStationPolicy, CheckFoldPolicy, ModelPolicy, RandomPolicy,
)


OPPONENT_FACTORIES = {
    "random":          RandomPolicy,
    "check_fold":      CheckFoldPolicy,
    "calling_station": CallingStationPolicy,
}


# ─── Actor process entry point ──────────────────────────────────────────────

def _actor_loop(actor_id: int,
                net: DMCNet,
                trans_queue: mp.Queue,
                stop_event,
                pause_event,
                eps_shared,
                ban_allin_pre_shared,
                cfg: DMCConfig,
                base_seed: int,
                torch_threads: int) -> None:
    """Play hands forever; push each hand's (xs, as_, rs) batch to the queue.

    Runs in a child process. `net` is the CPU-shared model — every actor
    reads the same parameter memory; the learner writes into it asynchronously
    via the per-N-steps sync from net_gpu.

    Each actor pins itself to `torch_threads` torch CPU threads (default 1).
    Single-state forward through the MLP is too small to benefit from
    intra-op parallelism, and the default of (cores) threads × N actors
    massively oversubscribes the box and starves the main process during
    eval — exactly the failure mode that hung the 200K run on 2026-05-02.

    `pause_event` is set by the main process during eval; actors wait
    between hands while it's set so eval has the CPU to itself. We never
    abandon a mid-hand — pause is checked at hand boundaries only.
    """
    torch.set_num_threads(max(1, int(torch_threads)))

    env = pte.Env((base_seed ^ (actor_id * 0x9E3779B1)) & 0xFFFFFFFFFFFFFFFF)
    rng = np.random.default_rng(base_seed + actor_id * 7919)
    device = torch.device("cpu")
    net.train(False)

    PREFLOP_BIT = 104   # x[104] is the preflop one-hot bit.

    try:
        while not stop_event.is_set():
            # Pause check at hand boundaries — never mid-decision.
            while pause_event.is_set() and not stop_event.is_set():
                stop_event.wait(0.05)
            if stop_event.is_set():
                break

            env.reset()
            eps = float(eps_shared.value)
            ban_allin_pre = bool(ban_allin_pre_shared.value)
            traj = {pte.Player.SB: PlayerTrajectory(),
                    pte.Player.BB: PlayerTrajectory()}

            while not env.is_terminal():
                actor = env.to_act()
                obs   = env.observation()
                ban = ()
                if ban_allin_pre and obs.x[PREFLOP_BIT] >= 0.5:
                    ban = (pte.ActionType.ALL_IN,)
                idx   = choose_action(obs, net, eps, device, rng, ban_actions=ban)
                traj[actor].add(obs.x, obs.a[idx])
                env.step(idx)

            payoffs = env.payoffs_bb()
            for p in (pte.Player.SB, pte.Player.BB):
                t = traj[p]
                if len(t) == 0:
                    continue
                xs = np.stack(t.xs)
                as_ = np.stack(t.as_)
                r = float(payoffs[int(p)])
                try:
                    trans_queue.put((xs, as_, r), timeout=1.0)
                except pyqueue.Full:
                    pass
    except (KeyboardInterrupt, SystemExit):
        return


# ─── Weight-sync ───────────────────────────────────────────────────────────

@torch.no_grad()
def sync_cpu_from_gpu(net_cpu: DMCNet, net_gpu: DMCNet) -> None:
    """Copy net_gpu's params + buffers into net_cpu's shared-memory tensors.

    In-place copy_() preserves the shared-memory storage of net_cpu's
    tensors, so actors keep reading from the same physical memory — they
    just see fresher numbers next time they fetch a parameter. No locks,
    no Queue traffic; the only cost is a single host-side D2H transfer
    per parameter (small for our ~1.4M param net).
    """
    cpu_params = dict(net_cpu.named_parameters())
    cpu_buffers = dict(net_cpu.named_buffers())
    for name, p in net_gpu.named_parameters():
        cpu_params[name].copy_(p.detach(), non_blocking=False)
    for name, b in net_gpu.named_buffers():
        if name in cpu_buffers:
            cpu_buffers[name].copy_(b.detach(), non_blocking=False)


# ─── Learner (main process) ─────────────────────────────────────────────────

def _drain_queue(trans_queue: mp.Queue, buf: MCBuffer,
                 max_items: int, timeout_s: float) -> int:
    n_trans = 0
    items = 0
    deadline = time.time() + timeout_s
    while items < max_items:
        remaining = max(0.0, deadline - time.time())
        try:
            xs, as_, r = trans_queue.get(timeout=remaining if items == 0 else 0.0)
        except pyqueue.Empty:
            break
        buf.append_trajectory(xs, as_, r)
        n_trans += xs.shape[0]
        items += 1
    return n_trans


def _run_eval_suite(net_cpu: DMCNet,
                    opponent_names: list[str],
                    n_hands: int, base_seed: int, step: int,
                    duplicated: bool = True) -> dict:
    """Evaluate net_cpu against the named baselines.

    Eval always runs on CPU (cheap for tiny batches, no GPU transfer
    overhead per inference). The caller is responsible for syncing
    net_cpu from net_gpu before calling this.
    """
    cpu_device = torch.device("cpu")
    model = ModelPolicy(net_cpu, cpu_device, name="model")
    results: dict[str, float] = {}
    for i, name in enumerate(opponent_names):
        opp = OPPONENT_FACTORIES[name]()
        seed = (base_seed + 1_000_003 * i) & 0xFFFFFFFFFFFFFFFF
        r = play_match(model, opp, n_hands=n_hands, base_seed=seed,
                       duplicated=duplicated)
        results[name] = r.a_mbb_per_100
        mode = "dup" if r.duplicated else "seq"
        print(f"[eval]  step={step}  vs {name:<16} "
              f"{r.a_mbb_per_100:+9.0f} mbb/100  "
              f"(SE {r.a_std_err_mbb_per_100:.0f})  "
              f"{r.n_hands} hands [{mode}] in {r.elapsed_s:.1f}s")
        print(r.slot_summary(per_street=True))
    net_cpu.train(True)
    return results


def save_checkpoint(path: Path, net_gpu: DMCNet, optim: torch.optim.Optimizer,
                    step: int, cfg: DMCConfig) -> None:
    """Save net_gpu's state_dict (the canonical, training copy)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step":  step,
        "model": {k: v.detach().cpu() for k, v in net_gpu.state_dict().items()},
        "optim": optim.state_dict(),
        "cfg":   cfg,
    }, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Multi-process actor pool
    p.add_argument("--mp-actors", type=int, default=4,
                   help="number of actor subprocesses")
    p.add_argument("--queue-maxsize", type=int, default=8192)
    p.add_argument("--drain-per-step", type=int, default=64)
    # Learner side
    p.add_argument("--learner-device", type=str, default="cuda:0",
                   help="torch device for the learner (cuda:N or cpu). The "
                        "actors stay on CPU regardless.")
    p.add_argument("--actor-sync-every", type=int, default=16,
                   help="copy net_gpu → net_cpu every N grad steps (so "
                        "actors pick up fresh weights). Set to 1 for the "
                        "tightest sync at the cost of more D2H transfers.")
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--buffer-capacity", type=int, default=200_000)
    p.add_argument("--min-buffer", type=int, default=4_000)
    p.add_argument("--grad-steps-per-iter", type=int, default=32)
    # Eval
    p.add_argument("--eval-every-steps", type=int, default=500)
    p.add_argument("--eval-hands", type=int, default=500)
    p.add_argument("--eval-opponents", type=str,
                   default="random,check_fold,calling_station")
    p.add_argument("--eval-seed", type=int, default=0xEEEE)
    p.add_argument("--no-duplicated-eval", action="store_true")
    # Bookkeeping
    p.add_argument("--ckpt-dir", type=str, default="runs/latest_mp_gpu")
    p.add_argument("--checkpoint-every-steps", type=int, default=500)
    p.add_argument("--watchdog-every-iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every-steps", type=int, default=100)
    p.add_argument("--smoke", action="store_true",
                   help="tiny run: 2 actors, 200 steps, small buffer.")
    # Schedules
    p.add_argument("--epsilon-start", type=float, default=None)
    p.add_argument("--epsilon-end", type=float, default=None)
    p.add_argument("--epsilon-decay-steps", type=int, default=None)
    p.add_argument("--reward-clip", type=float, default=None,
                   help="MC return clip in BB. START value if a schedule is set.")
    p.add_argument("--reward-clip-end", type=float, default=None,
                   help="if set, schedule clip linearly from --reward-clip "
                        "to this over --reward-clip-decay-steps")
    p.add_argument("--reward-clip-decay-steps", type=int, default=0)
    p.add_argument("--slot-log-every", type=int, default=0,
                   help="every N grad steps, print buffer slot distribution")
    p.add_argument("--no-allin-until-step", type=int, default=0,
                   help="forbid ALL_IN preflop in actors' rollout policy "
                        "until learner reaches step N. 0 disables.")
    p.add_argument("--actor-torch-threads", type=int, default=1,
                   help="torch.set_num_threads() inside each actor process. "
                        "Default 1 — single-state forward gets no benefit "
                        "from multi-threading, and the default of (cores) × "
                        "(N actors) oversubscribes the CPU and starves the "
                        "main process during eval.")
    p.add_argument("--pause-actors-during-eval", type=int, default=1,
                   help="1 (default) = freeze actors at hand boundaries "
                        "while eval runs, so eval has the CPU to itself. "
                        "0 = let actors keep running (will slow eval).")
    # Model architecture overrides (otherwise cfg.model defaults are used).
    p.add_argument("--arch", type=str, default=None,
                   choices=("mlp_v1", "resmlp_v1"))
    p.add_argument("--mlp-layers", type=int, default=None)
    p.add_argument("--mlp-hidden", type=int, default=None)
    p.add_argument("--mlp-expansion", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DMCConfig()
    cfg.run.device = args.learner_device
    cfg.run.seed = args.seed
    cfg.run.ckpt_dir = args.ckpt_dir
    cfg.learner.max_grad_steps = args.max_steps

    if args.smoke:
        args.mp_actors = 2
        args.max_steps = 200
        cfg.learner.max_grad_steps = 200
        args.buffer_capacity = 4_000
        args.min_buffer = 256
        args.grad_steps_per_iter = 8
        args.eval_every_steps = 0
        args.checkpoint_every_steps = 0
        args.log_every_steps = 20
        args.actor_sync_every = 8
    if args.epsilon_start is not None:
        cfg.actor.epsilon_start = args.epsilon_start
    if args.epsilon_end is not None:
        cfg.actor.epsilon_end = args.epsilon_end
    if args.epsilon_decay_steps is not None:
        cfg.actor.epsilon_decay_steps = args.epsilon_decay_steps
    if args.arch is not None:
        cfg.model.arch = args.arch
    if args.mlp_layers is not None:
        cfg.model.mlp_layers = args.mlp_layers
    if args.mlp_hidden is not None:
        cfg.model.mlp_hidden = args.mlp_hidden
    if args.mlp_expansion is not None:
        cfg.model.mlp_expansion = args.mlp_expansion

    torch.manual_seed(cfg.run.seed)

    # Sanity-check shape constants against the compiled engine.
    assert cfg.model.x_dim == pte.X_DIM, (cfg.model.x_dim, pte.X_DIM)
    assert cfg.model.a_dim == pte.A_DIM, (cfg.model.a_dim, pte.A_DIM)

    # ── Step 1: build the CPU shared net BEFORE any CUDA init ──────────────
    # This is the actors' inference-time view of the model. The optimizer
    # is built on net_gpu (below); net_cpu only gets written via in-place
    # copy_() during the periodic sync.
    net_cpu = DMCNet(cfg.model)
    net_cpu.share_memory()

    # ── Step 2: get the multiprocessing context (forkserver is CUDA-safe
    # because Python's `forkserver` start method spawns its helper via
    # fork+exec, which discards the parent's CUDA state regardless of
    # when the first ctx.Process.start() runs). Workers are then forked
    # from the (clean) helper, never directly from this process.
    ctx = mp.get_context("forkserver")
    learner_device = torch.device(args.learner_device)

    # ── Step 3: build the GPU learner net ───────────────────────────────────
    if learner_device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                f"--learner-device={args.learner_device} but torch reports "
                f"no CUDA available")
        net_gpu = DMCNet(cfg.model).to(learner_device)
        net_gpu.load_state_dict(net_cpu.state_dict())
    else:
        # CPU learner — net_gpu aliases net_cpu, sync becomes a no-op.
        net_gpu = net_cpu

    optim = build_optimizer(net_gpu, cfg.optim)

    buf = MCBuffer(
        capacity=args.buffer_capacity,
        x_dim=cfg.model.x_dim,
        a_dim=cfg.model.a_dim,
        rng=np.random.default_rng(cfg.run.seed + 1),
        reward_clip=args.reward_clip,
    )

    ckpt_dir = Path(cfg.run.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    eval_opponents: list[str] = []
    if args.eval_every_steps > 0:
        eval_opponents = [s.strip() for s in args.eval_opponents.split(",") if s.strip()]
        bad = [n for n in eval_opponents if n not in OPPONENT_FACTORIES]
        if bad:
            raise SystemExit(f"unknown eval opponents: {bad}")

    print(f"[dmc_mp_gpu] actors={args.mp_actors} "
          f"learner_device={learner_device} "
          f"params={count_parameters(net_gpu):,}  ckpt_dir={ckpt_dir}")
    print(f"[dmc_mp_gpu] cfg.model={cfg.model}")
    print(f"[dmc_mp_gpu] cfg.learner={cfg.learner}")
    print(f"[dmc_mp_gpu] cfg.actor={cfg.actor}")
    print(f"[dmc_mp_gpu] sync net_cpu ⟵ net_gpu every {args.actor_sync_every} grad steps")
    if eval_opponents:
        print(f"[dmc_mp_gpu] eval every {args.eval_every_steps} steps "
              f"vs {eval_opponents} ({args.eval_hands} hands each)")

    # ── Step 4: spawn actors ────────────────────────────────────────────────
    trans_queue: mp.Queue = ctx.Queue(maxsize=args.queue_maxsize)
    stop_event = ctx.Event()
    pause_event = ctx.Event()   # set by learner during eval; cleared otherwise
    eps_shared = ctx.Value('d', cfg.actor.epsilon_start)
    ban_allin_pre = 1 if args.no_allin_until_step > 0 else 0
    ban_allin_pre_shared = ctx.Value('i', ban_allin_pre)

    actors = []
    for aid in range(args.mp_actors):
        p = ctx.Process(
            target=_actor_loop,
            args=(aid, net_cpu, trans_queue, stop_event, pause_event,
                  eps_shared, ban_allin_pre_shared,
                  cfg, cfg.actor.base_seed, args.actor_torch_threads),
            daemon=True,
            name=f"actor-{aid}",
        )
        p.start()
        actors.append(p)
    print(f"[dmc_mp_gpu] spawned {len(actors)} actors  "
          f"eps_init={cfg.actor.epsilon_start:.3f} → "
          f"{cfg.actor.epsilon_end:.3f} over "
          f"{cfg.actor.epsilon_decay_steps} steps  "
          f"ban_allin_pre_until={args.no_allin_until_step}")

    step = 0
    iter_idx = 0
    t_start = time.time()
    total_transitions = 0
    loss_ema: float | None = None
    last_eval_step = -1
    sync_count = 0

    try:
        while step < cfg.learner.max_grad_steps:
            iter_idx += 1

            # Update shared epsilon for actors before draining.
            eps_shared.value = schedule_epsilon(step, cfg.actor)

            # Curriculum: lift the preflop ALL_IN ban once threshold passed.
            if (args.no_allin_until_step > 0
                    and ban_allin_pre_shared.value == 1
                    and step >= args.no_allin_until_step):
                ban_allin_pre_shared.value = 0
                print(f"[dmc_mp_gpu] step={step}: lifting preflop ALL_IN ban")

            # Watchdog: detect dead actors (silent crash → empty queue forever).
            if (args.watchdog_every_iters > 0
                    and iter_idx % args.watchdog_every_iters == 0):
                dead = [a.name for a in actors if not a.is_alive()]
                if dead:
                    raise SystemExit(
                        f"[dmc_mp_gpu] actors died: {dead} — "
                        f"aborting at step={step}")

            n_new = _drain_queue(trans_queue, buf,
                                 max_items=args.drain_per_step,
                                 timeout_s=0.5)
            total_transitions += n_new

            if len(buf) < args.min_buffer:
                if iter_idx % 20 == 0:
                    print(f"[dmc_mp_gpu] iter={iter_idx} buf={len(buf)} "
                          f"(<{args.min_buffer}) — warming up  "
                          f"qsize≈{trans_queue.qsize()}")
                continue

            for _ in range(args.grad_steps_per_iter):
                if step >= cfg.learner.max_grad_steps:
                    break
                buf.set_reward_clip(schedule_reward_clip(
                    step, args.reward_clip, args.reward_clip_end,
                    args.reward_clip_decay_steps))
                batch = buf.sample(cfg.learner.batch_size)
                s = learn_step(net_gpu, optim, batch, learner_device,
                               cfg.optim.grad_clip)
                step += 1
                loss_ema = (s["loss"] if loss_ema is None
                            else 0.98 * loss_ema + 0.02 * s["loss"])

                # Sync to actors. For learner_device==cpu, net_gpu IS
                # net_cpu, so the function does nothing material.
                if (learner_device.type == "cuda"
                        and step % args.actor_sync_every == 0):
                    sync_cpu_from_gpu(net_cpu, net_gpu)
                    sync_count += 1

                if step % args.log_every_steps == 0:
                    dt = time.time() - t_start
                    tps = total_transitions / max(dt, 1e-6)
                    sps = step / max(dt, 1e-6)
                    print(f"[dmc_mp_gpu] step={step} buf={len(buf)} "
                          f"loss_ema={loss_ema:.3f} "
                          f"grad={s['grad_norm']:.2f} "
                          f"eps={eps_shared.value:.3f} "
                          f"trans={total_transitions} "
                          f"trans/s={tps:.0f}  steps/s={sps:.1f}  "
                          f"qsize={trans_queue.qsize()}  "
                          f"syncs={sync_count}  wall={dt:.1f}s")

                if (args.slot_log_every > 0
                        and step % args.slot_log_every == 0
                        and step > 0):
                    print(MCBuffer.format_slot_distribution(buf.slot_distribution()))

                if (eval_opponents
                        and args.eval_every_steps > 0
                        and step % args.eval_every_steps == 0
                        and step != last_eval_step):
                    # Always sync before eval so net_cpu reflects the
                    # latest learner state, regardless of where we are
                    # in the actor-sync schedule.
                    if learner_device.type == "cuda":
                        sync_cpu_from_gpu(net_cpu, net_gpu)
                        sync_count += 1
                    last_eval_step = step
                    eval_t0 = time.time()
                    if args.pause_actors_during_eval:
                        pause_event.set()
                        # Brief settle so any in-flight hand can finish
                        # before we start hammering the CPU for eval.
                        time.sleep(0.5)
                        print(f"[dmc_mp_gpu] step={step}: actors paused for eval")
                    try:
                        _run_eval_suite(net_cpu, eval_opponents,
                                        args.eval_hands, args.eval_seed, step,
                                        duplicated=not args.no_duplicated_eval)
                    finally:
                        if args.pause_actors_during_eval:
                            pause_event.clear()
                            print(f"[dmc_mp_gpu] step={step}: eval done in "
                                  f"{time.time() - eval_t0:.1f}s — actors resumed")

                if (args.checkpoint_every_steps > 0
                        and step % args.checkpoint_every_steps == 0
                        and step > 0):
                    path = ckpt_dir / f"weights_{step:08d}.ckpt"
                    save_checkpoint(path, net_gpu, optim, step, cfg)
                    print(f"[dmc_mp_gpu] saved {path}")

        # Final eval + checkpoint.
        if eval_opponents and step != last_eval_step:
            if learner_device.type == "cuda":
                sync_cpu_from_gpu(net_cpu, net_gpu)
            if args.pause_actors_during_eval:
                pause_event.set()
                time.sleep(0.5)
            try:
                _run_eval_suite(net_cpu, eval_opponents, args.eval_hands,
                                args.eval_seed, step,
                                duplicated=not args.no_duplicated_eval)
            finally:
                if args.pause_actors_during_eval:
                    pause_event.clear()
        final = ckpt_dir / f"weights_final_{step:08d}.ckpt"
        save_checkpoint(final, net_gpu, optim, step, cfg)
        print(f"[dmc_mp_gpu] done — saved {final}  steps={step}  "
              f"trans={total_transitions}  syncs={sync_count}  "
              f"wall={time.time()-t_start:.1f}s")

    finally:
        stop_event.set()
        for p in actors:
            p.join(timeout=3.0)
        for p in actors:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)


if __name__ == "__main__":
    main()
