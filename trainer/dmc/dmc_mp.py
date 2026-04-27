"""Multi-process DMC training (actor-learner, CPU).

Architecture:
    N actor processes ── mp.Queue ──▶ Learner (main process)
       (shared net,                          │
        local env+rng)                       ├─▶ MCBuffer
                                             ├─▶ learn_step(net, optim, ...)
                                             └─▶ periodic eval + ckpt

The model is allocated on CPU and made shared via `net.share_memory_()`
BEFORE spawning actors. After fork, every process has Python-wrapping of
the same parameter memory: actors read (inference) and the learner writes
(optim.step()). There's no explicit weight sync — actors see "eventually
consistent" weights, which is fine for DMC (MC target variance dominates).

Each actor batches an entire hand's transitions (per-player) into one
Queue.put() call to amortize IPC overhead. The learner drains the queue
as fast as it can and writes into the same MCBuffer used by dmc.dmc.

CLI:
    python -m dmc.dmc_mp --mp-actors 4
    python -m dmc.dmc_mp --mp-actors 8 --max-steps 10000 --eval-every-steps 500
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
                eps_shared,
                ban_allin_pre_shared,
                cfg: DMCConfig,
                base_seed: int) -> None:
    """Play hands forever; push each hand's (xs, as_, rs) batch to the queue.

    Runs in a child process. The shared `net` lives in shared memory; calling
    `choose_action` reads current weights without any explicit sync.
    `eps_shared` is an mp.Value('d') updated by the learner each iteration to
    reflect the linear epsilon decay schedule; the actor reads it once per
    hand (cheap, contention-free for a single double).
    `ban_allin_pre_shared` is an mp.Value('i'): 1 = exclude ALL_IN from
    preflop legal actions during rollout; 0 = unrestricted. Toggled by the
    learner once `step >= --no-allin-until-step`.
    """
    # Fork inherits the parent's imported modules, so pte is already loaded.
    # Give each actor its own Env with a disjoint seed stream.
    env = pte.Env((base_seed ^ (actor_id * 0x9E3779B1)) & 0xFFFFFFFFFFFFFFFF)
    rng = np.random.default_rng(base_seed + actor_id * 7919)
    device = torch.device("cpu")
    net.train(False)   # inference mode for the actor's copy

    # Street offset in the encoded state: x[104] is the preflop one-hot bit.
    PREFLOP_BIT = 104

    try:
        while not stop_event.is_set():
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
            # Pack each player's trajectory into stacked numpy arrays and ship
            # in one Queue.put to amortize IPC. Learner will append_trajectory.
            for p in (pte.Player.SB, pte.Player.BB):
                t = traj[p]
                if len(t) == 0:
                    continue
                xs = np.stack(t.xs)       # (T, X_DIM)
                as_ = np.stack(t.as_)     # (T, A_DIM)
                r = float(payoffs[int(p)])
                try:
                    trans_queue.put((xs, as_, r), timeout=1.0)
                except pyqueue.Full:
                    # Learner is falling behind; drop this batch. With a large
                    # queue this should essentially never happen, but don't
                    # stall the actor on backpressure.
                    pass
    except (KeyboardInterrupt, SystemExit):
        return


# ─── Learner (main process) ─────────────────────────────────────────────────

def _drain_queue(trans_queue: mp.Queue, buf: MCBuffer,
                 max_items: int, timeout_s: float) -> int:
    """Move up to max_items (xs, as_, r) batches from the queue into the buffer.

    Returns the total number of transitions appended.
    """
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


def _run_eval_suite(net: DMCNet, device: torch.device,
                    opponent_names: list[str],
                    n_hands: int, base_seed: int, step: int,
                    duplicated: bool = True) -> dict:
    model = ModelPolicy(net, device, name="model")
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
    net.train(True)
    return results


def save_checkpoint(path: Path, net: DMCNet, optim: torch.optim.Optimizer,
                    step: int, cfg: DMCConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step":  step,
        "model": net.state_dict(),
        "optim": optim.state_dict(),
        "cfg":   cfg,
    }, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mp-actors", type=int, default=4,
                   help="number of actor subprocesses")
    p.add_argument("--queue-maxsize", type=int, default=8192,
                   help="max pending trajectory batches in the actor->learner queue")
    p.add_argument("--drain-per-step", type=int, default=64,
                   help="max trajectory batches drained per learner tick")
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--buffer-capacity", type=int, default=200_000)
    p.add_argument("--min-buffer", type=int, default=4_000)
    p.add_argument("--grad-steps-per-iter", type=int, default=32,
                   help="gradient steps per drain/learn cycle")
    p.add_argument("--eval-every-steps", type=int, default=500)
    p.add_argument("--eval-hands", type=int, default=500)
    p.add_argument("--eval-opponents", type=str,
                   default="random,check_fold,calling_station")
    p.add_argument("--eval-seed", type=int, default=0xEEEE)
    p.add_argument("--no-duplicated-eval", action="store_true")
    p.add_argument("--ckpt-dir", type=str, default="runs/latest_mp")
    p.add_argument("--checkpoint-every-steps", type=int, default=500,
                   help="save weights_<step>.ckpt every N grad steps (0 = end-only)")
    p.add_argument("--watchdog-every-iters", type=int, default=20,
                   help="how often to verify all actors are still alive")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every-steps", type=int, default=100)
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end run: small buffer, few steps, 2 actors")
    p.add_argument("--epsilon-start", type=float, default=None)
    p.add_argument("--epsilon-end", type=float, default=None)
    p.add_argument("--epsilon-decay-steps", type=int, default=None)
    p.add_argument("--reward-clip", type=float, default=None,
                   help="symmetric clip on MC terminal return (BB). START "
                        "value if a schedule is set, else constant.")
    p.add_argument("--reward-clip-end", type=float, default=None,
                   help="if set, schedule clip linearly from --reward-clip to "
                        "this value over --reward-clip-decay-steps")
    p.add_argument("--reward-clip-decay-steps", type=int, default=0,
                   help="grad-steps over which to interpolate clip. 0 = constant.")
    p.add_argument("--slot-log-every", type=int, default=0,
                   help="every N grad steps, print buffer slot distribution")
    p.add_argument("--no-allin-until-step", type=int, default=0,
                   help="forbid ALL_IN preflop in the actor's rollout policy "
                        "until the learner reaches step N. Eval policy is "
                        "untouched. 0 disables the curriculum.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DMCConfig()
    cfg.run.device = "cpu"
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
    if args.epsilon_start is not None:
        cfg.actor.epsilon_start = args.epsilon_start
    if args.epsilon_end is not None:
        cfg.actor.epsilon_end = args.epsilon_end
    if args.epsilon_decay_steps is not None:
        cfg.actor.epsilon_decay_steps = args.epsilon_decay_steps

    torch.manual_seed(cfg.run.seed)

    # Build the (shared) network on CPU and move its params into shared memory.
    device = torch.device("cpu")
    net = DMCNet(cfg.model).to(device)
    net.share_memory()
    optim = build_optimizer(net, cfg.optim)

    # Sanity-check shape constants against the compiled engine.
    assert cfg.model.x_dim == pte.X_DIM
    assert cfg.model.a_dim == pte.A_DIM

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

    print(f"[dmc_mp] actors={args.mp_actors} params={count_parameters(net):,} "
          f"ckpt_dir={ckpt_dir}")
    print(f"[dmc_mp] cfg.model={cfg.model}")
    print(f"[dmc_mp] cfg.learner={cfg.learner}")
    print(f"[dmc_mp] cfg.actor={cfg.actor}")
    if eval_opponents:
        print(f"[dmc_mp] eval every {args.eval_every_steps} steps "
              f"vs {eval_opponents} ({args.eval_hands} hands each)")

    # ── Spawn actors ────────────────────────────────────────────────────────
    # `forkserver`: a tiny helper process snapshots a clean Python state
    # (without the parent's torch CUDA init), then forks each worker from it.
    # Plain `fork` deadlocks here because torch 2.x with a CUDA build
    # initializes CUDA at import time, and fork after CUDA init is unsafe.
    # `spawn` would also work but pays full Python startup per actor.
    ctx = mp.get_context("forkserver")
    trans_queue: mp.Queue = ctx.Queue(maxsize=args.queue_maxsize)
    stop_event = ctx.Event()
    # Shared epsilon: learner writes, actors read. 'd' = C double in shm.
    eps_shared = ctx.Value('d', cfg.actor.epsilon_start)
    # Shared curriculum flag: 1 = forbid ALL_IN preflop in rollout, 0 = allow.
    ban_allin_pre = 1 if args.no_allin_until_step > 0 else 0
    ban_allin_pre_shared = ctx.Value('i', ban_allin_pre)

    actors = []
    for aid in range(args.mp_actors):
        p = ctx.Process(
            target=_actor_loop,
            args=(aid, net, trans_queue, stop_event, eps_shared,
                  ban_allin_pre_shared, cfg, cfg.actor.base_seed),
            daemon=True,
            name=f"actor-{aid}",
        )
        p.start()
        actors.append(p)
    print(f"[dmc_mp] spawned {len(actors)} actors  "
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

    try:
        while step < cfg.learner.max_grad_steps:
            iter_idx += 1

            # Update shared epsilon for actors before we drain (so the next
            # batch of transitions reflects the current schedule). The Value
            # write is atomic for a double on x86_64; no lock needed.
            eps_shared.value = schedule_epsilon(step, cfg.actor)

            # Curriculum: lift the preflop ALL_IN ban once the learner has
            # passed the threshold. After lift, actors fall back to normal
            # legal-action set on the very next hand.
            if (args.no_allin_until_step > 0
                    and ban_allin_pre_shared.value == 1
                    and step >= args.no_allin_until_step):
                ban_allin_pre_shared.value = 0
                print(f"[dmc_mp] step={step}: lifting preflop ALL_IN ban")

            # Watchdog: periodically verify all actors are alive. A silent
            # crash would otherwise leave the learner spinning on an empty
            # queue forever.
            if (args.watchdog_every_iters > 0
                    and iter_idx % args.watchdog_every_iters == 0):
                dead = [a.name for a in actors if not a.is_alive()]
                if dead:
                    raise SystemExit(
                        f"[dmc_mp] actors died: {dead} — aborting at step={step}")

            # Drain transitions produced by actors (blocks briefly if empty).
            n_new = _drain_queue(trans_queue, buf,
                                 max_items=args.drain_per_step,
                                 timeout_s=0.5)
            total_transitions += n_new

            if len(buf) < args.min_buffer:
                if iter_idx % 20 == 0:
                    print(f"[dmc_mp] iter={iter_idx} buf={len(buf)} "
                          f"(<{args.min_buffer}) — warming up  "
                          f"qsize≈{trans_queue.qsize()}")
                continue

            # Learner: fire off grad_steps_per_iter gradient steps.
            for _ in range(args.grad_steps_per_iter):
                if step >= cfg.learner.max_grad_steps:
                    break
                buf.set_reward_clip(schedule_reward_clip(
                    step, args.reward_clip, args.reward_clip_end,
                    args.reward_clip_decay_steps))
                batch = buf.sample(cfg.learner.batch_size)
                s = learn_step(net, optim, batch, device, cfg.optim.grad_clip)
                step += 1
                loss_ema = (s["loss"] if loss_ema is None
                            else 0.98 * loss_ema + 0.02 * s["loss"])

                if step % args.log_every_steps == 0:
                    dt = time.time() - t_start
                    tps = total_transitions / max(dt, 1e-6)
                    sps = step / max(dt, 1e-6)
                    print(f"[dmc_mp] step={step} buf={len(buf)} "
                          f"loss_ema={loss_ema:.3f} "
                          f"grad={s['grad_norm']:.2f} "
                          f"eps={eps_shared.value:.3f} "
                          f"trans={total_transitions} "
                          f"trans/s={tps:.0f}  steps/s={sps:.1f}  "
                          f"qsize={trans_queue.qsize()}  wall={dt:.1f}s")

                if (args.slot_log_every > 0
                        and step % args.slot_log_every == 0
                        and step > 0):
                    print(MCBuffer.format_slot_distribution(buf.slot_distribution()))

                if (eval_opponents
                        and args.eval_every_steps > 0
                        and step % args.eval_every_steps == 0
                        and step != last_eval_step):
                    last_eval_step = step
                    _run_eval_suite(net, device, eval_opponents,
                                    args.eval_hands, args.eval_seed, step,
                                    duplicated=not args.no_duplicated_eval)

                if (args.checkpoint_every_steps > 0
                        and step % args.checkpoint_every_steps == 0
                        and step > 0):
                    path = ckpt_dir / f"weights_{step:08d}.ckpt"
                    save_checkpoint(path, net, optim, step, cfg)
                    print(f"[dmc_mp] saved {path}")

        # Final eval + checkpoint
        if eval_opponents and step != last_eval_step:
            _run_eval_suite(net, device, eval_opponents, args.eval_hands,
                            args.eval_seed, step,
                            duplicated=not args.no_duplicated_eval)
        final = ckpt_dir / f"weights_final_{step:08d}.ckpt"
        save_checkpoint(final, net, optim, step, cfg)
        print(f"[dmc_mp] done — saved {final}  steps={step}  "
              f"trans={total_transitions}  wall={time.time()-t_start:.1f}s")

    finally:
        # Clean shutdown: signal actors, then join with a deadline.
        stop_event.set()
        for p in actors:
            p.join(timeout=3.0)
        for p in actors:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)


if __name__ == "__main__":
    main()
