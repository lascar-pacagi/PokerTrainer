"""Main DMC training entry point (single-process).

Loop, per iteration:
    1. Rollout: play rollout_hands hands, append transitions to the buffer.
    2. Learn:   take grad_steps_per_iter gradient steps on sampled batches.
    3. Log:     every log_every_steps, print a one-line rollup.
    4. Eval:    every eval_every_steps, play N hands vs fixed baselines.
    5. Ckpt:    every checkpoint_every_steps, save weights + optim state.

Phase 1.1 runs this single-process. Phase 1.2 will fork multiple actors that
push transitions into a shared buffer and one learner that consumes them.

CLI:
    python -m dmc.dmc --smoke           # tiny end-to-end sanity run on CPU
    python -m dmc.dmc --device cuda:0   # real training
    python -m dmc.dmc --eval-every-steps 200 --eval-hands 500
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import pokertrainer_engine as pte

from .actor import rollout, schedule_epsilon, schedule_reward_clip
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end run: few hands, a few grad steps, CPU.")
    p.add_argument("--device", type=str, default=None, help="override run.device")
    p.add_argument("--ckpt-dir", type=str, default=None, help="override run.ckpt_dir")
    p.add_argument("--seed", type=int, default=None, help="override run.seed")
    p.add_argument("--max-steps", type=int, default=None,
                   help="override learner.max_grad_steps")
    p.add_argument("--rollout-hands", type=int, default=128,
                   help="hands per rollout phase (before each learner cycle)")
    p.add_argument("--grad-steps-per-iter", type=int, default=128,
                   help="gradient steps per learner cycle")
    p.add_argument("--buffer-capacity", type=int, default=200_000,
                   help="transitions held in the replay buffer")
    p.add_argument("--min-buffer", type=int, default=2_000,
                   help="minimum transitions in buffer before learning starts")
    p.add_argument("--eval-every-steps", type=int, default=200,
                   help="run baseline eval every N grad steps (0 disables)")
    p.add_argument("--eval-hands", type=int, default=500,
                   help="hands per eval match per opponent")
    p.add_argument("--eval-opponents", type=str,
                   default="random,check_fold,calling_station",
                   help="comma-separated subset of "
                        + ",".join(OPPONENT_FACTORIES.keys()))
    p.add_argument("--eval-seed", type=int, default=0xEEEE,
                   help="separate seed stream for eval hands")
    p.add_argument("--epsilon-start", type=float, default=None,
                   help="override actor.epsilon_start")
    p.add_argument("--epsilon-end", type=float, default=None,
                   help="override actor.epsilon_end")
    p.add_argument("--epsilon-decay-steps", type=int, default=None,
                   help="override actor.epsilon_decay_steps")
    p.add_argument("--no-duplicated-eval", action="store_true",
                   help="disable duplicate-dealing eval (use seat-alternation only)")
    p.add_argument("--train-against", type=str, default="self",
                   choices=["self"] + list(OPPONENT_FACTORIES.keys()),
                   help="opponent for the rollout phase. 'self' = self-play "
                        "(default). Otherwise a fixed baseline; the model "
                        "learns to maximally exploit it. Eval suite is "
                        "unchanged (always model vs each baseline).")
    p.add_argument("--reward-clip", type=float, default=None,
                   help="symmetric MC return clip (BB), START value if a "
                        "schedule is set, else constant. ~25 is the canonical "
                        "tight value for HU 100bb (see project_reward_clip_sweep).")
    p.add_argument("--reward-clip-end", type=float, default=None,
                   help="if set, schedule the clip linearly from --reward-clip "
                        "to this value over --reward-clip-decay-steps. Use a "
                        "large value (e.g. 200) to effectively unclip late in "
                        "training.")
    p.add_argument("--reward-clip-decay-steps", type=int, default=0,
                   help="grad-steps over which to interpolate clip. 0 = no "
                        "schedule (constant --reward-clip).")
    p.add_argument("--slot-log-every", type=int, default=0,
                   help="every N grad steps, print buffer slot distribution. "
                        "0 disables.")
    p.add_argument("--checkpoint-every-steps", type=int, default=None,
                   help="save weights_<step>.ckpt every N grad steps. "
                        "0 = end-only (just weights_final). overrides "
                        "cfg.learner.checkpoint_every_steps.")
    # ─ Model architecture overrides
    p.add_argument("--arch", type=str, default=None,
                   choices=["mlp_v1", "resmlp_v1"],
                   help="model trunk: mlp_v1 (legacy) or resmlp_v1 "
                        "(LN+residual, deeper). overrides cfg.model.arch.")
    p.add_argument("--mlp-layers", type=int, default=None,
                   help="# of layers (mlp_v1) or # of residual blocks "
                        "(resmlp_v1). overrides cfg.model.mlp_layers.")
    p.add_argument("--mlp-hidden", type=int, default=None,
                   help="hidden width. overrides cfg.model.mlp_hidden.")
    p.add_argument("--mlp-expansion", type=int, default=None,
                   help="FFN expansion ratio for resmlp_v1 (default 4). "
                        "overrides cfg.model.mlp_expansion.")
    return p.parse_args()


def apply_smoke_overrides(cfg: DMCConfig, args: argparse.Namespace) -> None:
    """Shrink cfg for a laptop-safe sanity run."""
    cfg.run.device = "cpu"
    cfg.learner.max_grad_steps = 20
    cfg.learner.log_every_steps = 5
    cfg.learner.checkpoint_every_steps = 10_000  # skip in smoke
    args.rollout_hands = 16
    args.grad_steps_per_iter = 5
    args.buffer_capacity = 4_000
    args.min_buffer = 64
    args.eval_every_steps = 0                    # skip in smoke


def save_checkpoint(path: Path, net: DMCNet, optim: torch.optim.Optimizer,
                    step: int, cfg: DMCConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step":  step,
        "model": net.state_dict(),
        "optim": optim.state_dict(),
        "cfg":   cfg,
    }, path)


def run_eval_suite(net: DMCNet, device: torch.device,
                   opponent_names: list[str],
                   n_hands: int, base_seed: int,
                   step: int, duplicated: bool = True) -> dict:
    """Play n_hands vs each named baseline and print a compact rollup.

    Returns {opponent_name: mbb_per_100}. Restores net.train(True) on exit so
    the learner can keep stepping without surprises.
    """
    model = ModelPolicy(net, device, name="model")
    results: dict[str, float] = {}
    # Fresh per-opponent seed so eval hands don't overlap between opponents
    # (and reproducible across runs given the same eval-seed).
    for i, name in enumerate(opponent_names):
        factory = OPPONENT_FACTORIES[name]
        opp = factory()
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


def main() -> None:
    args = parse_args()
    cfg = DMCConfig()

    if args.smoke:
        apply_smoke_overrides(cfg, args)
    if args.device is not None:
        cfg.run.device = args.device
    if args.ckpt_dir is not None:
        cfg.run.ckpt_dir = args.ckpt_dir
    if args.seed is not None:
        cfg.run.seed = args.seed
    if args.max_steps is not None:
        cfg.learner.max_grad_steps = args.max_steps
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
    if args.checkpoint_every_steps is not None:
        cfg.learner.checkpoint_every_steps = args.checkpoint_every_steps

    torch.manual_seed(cfg.run.seed)
    rng = np.random.default_rng(cfg.run.seed)

    device = torch.device(cfg.run.device)
    net = DMCNet(cfg.model).to(device)
    optim = build_optimizer(net, cfg.optim)

    # Sanity-check shape constants against the compiled engine.
    assert cfg.model.x_dim == pte.X_DIM, (cfg.model.x_dim, pte.X_DIM)
    assert cfg.model.a_dim == pte.A_DIM, (cfg.model.a_dim, pte.A_DIM)

    env = pte.Env(cfg.actor.base_seed)
    buf = MCBuffer(
        capacity=args.buffer_capacity,
        x_dim=cfg.model.x_dim,
        a_dim=cfg.model.a_dim,
        rng=np.random.default_rng(cfg.run.seed + 1),
        reward_clip=args.reward_clip,
    )

    ckpt_dir = Path(cfg.run.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Parse eval opponent list once up-front so a typo fails fast.
    eval_opponents: list[str] = []
    if args.eval_every_steps > 0:
        eval_opponents = [s.strip() for s in args.eval_opponents.split(",") if s.strip()]
        unknown = [n for n in eval_opponents if n not in OPPONENT_FACTORIES]
        if unknown:
            raise SystemExit(f"unknown eval opponent(s): {unknown}. "
                             f"known: {list(OPPONENT_FACTORIES)}")

    # Build the rollout opponent policy if we're training against a baseline.
    train_opponent = None
    if args.train_against != "self":
        train_opponent = OPPONENT_FACTORIES[args.train_against]()

    print(f"[dmc] device={device} params={count_parameters(net):,} "
          f"ckpt_dir={ckpt_dir}")
    print(f"[dmc] cfg.model={cfg.model}")
    print(f"[dmc] cfg.optim={cfg.optim}")
    print(f"[dmc] cfg.learner={cfg.learner}")
    print(f"[dmc] cfg.actor={cfg.actor}")
    print(f"[dmc] train_against={args.train_against}")
    if eval_opponents:
        print(f"[dmc] eval every {args.eval_every_steps} steps "
              f"vs {eval_opponents} ({args.eval_hands} hands each)")

    step = 0
    iter_idx = 0
    t_start = time.time()
    hands_total = 0
    loss_ema: float | None = None
    last_eval_step = -1

    while step < cfg.learner.max_grad_steps:
        iter_idx += 1
        # --- rollout ---
        eps = schedule_epsilon(step, cfg.actor)
        ro = rollout(env, net, buf,
                     n_hands=args.rollout_hands,
                     epsilon=eps,
                     device=device,
                     rng=rng,
                     opponent_policy=train_opponent)
        hands_total += ro["hands"]

        if len(buf) < args.min_buffer:
            print(f"[dmc] iter={iter_idx} buf={len(buf)} "
                  f"(<{args.min_buffer}) — warming up")
            continue

        # --- learn ---
        for _ in range(args.grad_steps_per_iter):
            if step >= cfg.learner.max_grad_steps:
                break
            buf.set_reward_clip(schedule_reward_clip(
                step, args.reward_clip, args.reward_clip_end,
                args.reward_clip_decay_steps))
            batch = buf.sample(cfg.learner.batch_size)
            stats = learn_step(net, optim, batch, device, cfg.optim.grad_clip)
            step += 1
            loss_ema = (stats["loss"] if loss_ema is None
                        else 0.98 * loss_ema + 0.02 * stats["loss"])

            if step % cfg.learner.log_every_steps == 0:
                dt = time.time() - t_start
                print(f"[dmc] step={step} hands={hands_total} buf={len(buf)} "
                      f"loss={stats['loss']:.4f} loss_ema={loss_ema:.4f} "
                      f"grad={stats['grad_norm']:.2f} "
                      f"mean_pred={stats['mean_pred']:+.3f} "
                      f"mean_tgt={stats['mean_target']:+.3f} "
                      f"eps={eps:.3f} "
                      f"wall={dt:.1f}s")

            if (args.slot_log_every > 0
                    and step % args.slot_log_every == 0
                    and step > 0):
                print(MCBuffer.format_slot_distribution(buf.slot_distribution()))

            if (eval_opponents
                    and args.eval_every_steps > 0
                    and step % args.eval_every_steps == 0
                    and step != last_eval_step):
                last_eval_step = step
                run_eval_suite(net, device,
                               eval_opponents,
                               args.eval_hands,
                               args.eval_seed,
                               step,
                               duplicated=not args.no_duplicated_eval)

            if (cfg.learner.checkpoint_every_steps > 0
                    and step % cfg.learner.checkpoint_every_steps == 0
                    and step > 0):
                path = ckpt_dir / f"weights_{step:08d}.ckpt"
                save_checkpoint(path, net, optim, step, cfg)
                print(f"[dmc] saved {path}")

    # Final eval + checkpoint.
    if eval_opponents and step != last_eval_step:
        run_eval_suite(net, device, eval_opponents, args.eval_hands,
                       args.eval_seed, step,
                       duplicated=not args.no_duplicated_eval)
    final = ckpt_dir / f"weights_final_{step:08d}.ckpt"
    save_checkpoint(final, net, optim, step, cfg)
    print(f"[dmc] done — saved {final}  steps={step}  hands={hands_total}  "
          f"wall={time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
