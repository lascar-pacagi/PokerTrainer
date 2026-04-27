"""Policy inspection v1 — text-based, no matplotlib.

Four views:
  preflop_range     — 13×13 grid of starting-hand classes (AA, AKs, AKo, ..., 22)
                      showing the model's dominant action and VPIP rate.
  postflop_strength — action distribution per (street × made-hand strength)
                      bucket. Tells you what the model does with top pair on
                      flop, with trips, with nothing — the "is the postflop
                      policy sane?" diagnostic.
  hand_trace        — play one hand (model vs check_fold by default), print
                      each decision with the model's Q-values across legal
                      actions.
  q_values          — for a fresh seed, dump the model's Q-vector for SB's
                      first preflop decision. Useful to spot specific hand
                      classes.

Usage:
    PYTHONPATH=engine/build:trainer python -m inspect_policy \\
        --ckpt runs/gpu_100k_clip10/weights_final_00100000.ckpt \\
        --view preflop_range --n-hands 50000

    PYTHONPATH=engine/build:trainer python -m inspect_policy \\
        --ckpt ... --view postflop_strength --n-hands 100000

    PYTHONPATH=engine/build:trainer python -m inspect_policy \\
        --ckpt ... --view hand_trace --seed 42 --opponent calling_station
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import pokertrainer_engine as pte

import torch.serialization as _ts
from dmc.config import (
    ActorConfig, DMCConfig, LearnerConfig, ModelConfig, OptimConfig, RunConfig,
)
from dmc.models import DMCNet


_ts.add_safe_globals([
    DMCConfig, ModelConfig, OptimConfig,
    ActorConfig, LearnerConfig, RunConfig,
])


# ─── Card / hand-class helpers ──────────────────────────────────────────────
RANK_CHARS = "23456789TJQKA"   # rank index 0..12
SUIT_CHARS = "cdhs"            # suit index 0..3


def hand_class(c1: int, c2: int) -> tuple[int, int, str]:
    """Return (high_rank, low_rank, kind) for two card indices.

    kind ∈ {'pair', 'suited', 'offsuit'}. high_rank/low_rank are the standard
    13-class indices (0=2, 12=A) with high_rank >= low_rank.
    """
    r1, s1 = pte.rank_of(c1), pte.suit_of(c1)
    r2, s2 = pte.rank_of(c2), pte.suit_of(c2)
    if r1 < r2:
        r1, r2, s1, s2 = r2, r1, s2, s1
    if r1 == r2:
        return r1, r2, "pair"
    return (r1, r2, "suited" if s1 == s2 else "offsuit")


# ─── Model loading ──────────────────────────────────────────────────────────
def load_model(ckpt_path: Path, device: torch.device) -> DMCNet:
    blob = torch.load(ckpt_path, map_location=device)
    cfg = blob.get("cfg")
    if cfg is not None and hasattr(cfg, "model"):
        mcfg = cfg.model
        if not isinstance(mcfg, ModelConfig):
            mcfg = ModelConfig(
                x_dim=int(mcfg.x_dim), a_dim=int(mcfg.a_dim),
                mlp_hidden=int(mcfg.mlp_hidden), mlp_layers=int(mcfg.mlp_layers),
                arch=getattr(mcfg, "arch", "mlp_v1"),
                mlp_expansion=int(getattr(mcfg, "mlp_expansion", 4)),
            )
    else:
        mcfg = DMCConfig().model
    net = DMCNet(mcfg).to(device)
    net.load_state_dict(blob["model"])
    net.train(False)
    print(f"[inspect] loaded {ckpt_path}  step={blob.get('step', '?')}  "
          f"x={mcfg.x_dim} a={mcfg.a_dim} hidden={mcfg.mlp_hidden}")
    return net


# ─── Greedy argmax helper ───────────────────────────────────────────────────
@torch.no_grad()
def greedy_action(net: DMCNet, obs, device: torch.device) -> tuple[int, np.ndarray]:
    """Return (legal_idx, q_values_per_legal)."""
    x = torch.from_numpy(np.asarray(obs.x)).to(device)
    a = torch.from_numpy(np.asarray(obs.a)).to(device)
    vals = net.score_legal(x, a)
    q = vals.cpu().numpy()
    return int(np.argmax(q)), q


# ─── Action labelling ───────────────────────────────────────────────────────
ACTION_SHORT = {
    int(pte.ActionType.FOLD):       "F",
    int(pte.ActionType.CHECK_CALL): "C",
    int(pte.ActionType.RAISE_25):   "R25",
    int(pte.ActionType.RAISE_33):   "R33",
    int(pte.ActionType.RAISE_50):   "R50",
    int(pte.ActionType.RAISE_75):   "R75",
    int(pte.ActionType.RAISE_100):  "R1x",
    int(pte.ActionType.RAISE_150):  "R15",
    int(pte.ActionType.RAISE_200):  "R2x",
    int(pte.ActionType.RAISE_300):  "R3x",
    int(pte.ActionType.ALL_IN):     "AI",
}

# ANSI color codes for action categories. Optional (--no-color).
ANSI = {
    "FOLD":  "\033[2;90m",   # dim grey
    "CALL":  "\033[2;36m",   # cyan
    "RAISE": "\033[1;33m",   # bold yellow
    "AI":    "\033[1;31m",   # bold red
    "RST":   "\033[0m",
}


def colorize(label: str, action_int: int, use_color: bool) -> str:
    if not use_color:
        return label
    if action_int == int(pte.ActionType.FOLD):
        return f"{ANSI['FOLD']}{label}{ANSI['RST']}"
    if action_int == int(pte.ActionType.CHECK_CALL):
        return f"{ANSI['CALL']}{label}{ANSI['RST']}"
    if action_int == int(pte.ActionType.ALL_IN):
        return f"{ANSI['AI']}{label}{ANSI['RST']}"
    return f"{ANSI['RAISE']}{label}{ANSI['RST']}"


# ─── Preflop range view ─────────────────────────────────────────────────────
def view_preflop_range(net: DMCNet, device: torch.device,
                       n_hands: int, seed: int, use_color: bool) -> None:
    """Sample N random hands, record the model's argmax over preflop SB-to-act
    states, bucket by starting-hand class, render a 13×13 grid.
    """
    # buckets[(high_rank, low_rank, kind)] -> {"actions": Counter, "n": int}
    counts = {}  # keyed by (high, low, kind), value = dict[action_int -> count]
    rng = np.random.default_rng(seed)
    print(f"[preflop_range] sampling {n_hands} preflop SB states…")
    env = pte.Env(seed)
    for h in range(n_hands):
        env.reset(int(rng.integers(0, 2**62)))
        # Always SB to act on a fresh deal preflop. Sanity-check anyway.
        if env.is_terminal():
            continue
        obs = env.observation()
        s = env.state()
        # SB's hole when SB is to_act — that's the to_act player's hole.
        hole = list(s.hole[int(s.to_act)])
        hi, lo, kind = hand_class(hole[0], hole[1])

        idx, _q = greedy_action(net, obs, device)
        action_int = int(obs.legal[idx])

        key = (hi, lo, kind)
        bucket = counts.setdefault(key, {"actions": {}, "n": 0})
        bucket["actions"][action_int] = bucket["actions"].get(action_int, 0) + 1
        bucket["n"] += 1

    # Render. Standard chart layout: rows = hi rank descending (A..2), cols = lo rank.
    # Diagonal = pairs. Upper triangle = suited. Lower triangle = offsuit.
    print()
    header = "       " + " ".join(f"{RANK_CHARS[12 - c]:>5}" for c in range(13))
    print(header)
    for r in range(13):  # row = high-rank index, displayed top-down A→2
        hi_rank = 12 - r
        line_parts = [f" {RANK_CHARS[hi_rank]} |"]
        for c in range(13):
            lo_rank = 12 - c
            # Pairs on diagonal, suited above (col > row in display = lo_rank < hi_rank
            # but the standard chart has suited in upper triangle when row=hi, col=lo
            # AND lo > hi visually means... let's just match the convention:
            # cell (row=A, col=K) where col is to the right of row → suited if col rank
            # is *lower* in our display (since A→2 left to right). Hmm.
            # Standard chart: rows = both hand cards' high rank, cols = the other rank.
            # cell on diagonal = pair. If col rank < row rank → it's "AK" type with the
            # lower card; convention: upper-right triangle = suited, lower-left = offsuit.
            # Display: rows top→bottom A,K,...,2; cols left→right A,K,...,2.
            # cell(row=A, col=K) — A is higher than K → AK; convention-wise, this
            # cell in upper-right triangle means suited.
            display_hi = 12 - r
            display_lo = 12 - c
            if display_hi == display_lo:
                kind = "pair"
                hi_used, lo_used = display_hi, display_hi
            elif c > r:
                # upper triangle = suited; the high rank is the row's rank
                kind = "suited"
                hi_used, lo_used = display_hi, display_lo
            else:
                # lower triangle = offsuit; the high rank is the col's rank
                kind = "offsuit"
                hi_used, lo_used = display_lo, display_hi
            key = (hi_used, lo_used, kind)
            bucket = counts.get(key)
            if not bucket or bucket["n"] == 0:
                line_parts.append(" - - ")
                continue
            actions = bucket["actions"]
            dominant = max(actions, key=actions.get)
            label = ACTION_SHORT.get(dominant, "?")
            vpip = 1.0 - actions.get(int(pte.ActionType.FOLD), 0) / bucket["n"]
            cell = f"{label:>3}{int(vpip*100):>2d}"
            line_parts.append(colorize(cell, dominant, use_color))
        print("  ".join(line_parts))

    print()
    print("Legend: each cell shows DOMINANT_ACTION + VPIP%.")
    print("  F = fold, C = check/call, R<X> = raise to X% pot, AI = all-in.")
    print("  Diagonal = pairs, upper-right = suited, lower-left = offsuit.")
    if use_color:
        print(f"  Color: {ANSI['FOLD']}fold{ANSI['RST']}  "
              f"{ANSI['CALL']}call{ANSI['RST']}  "
              f"{ANSI['RAISE']}raise{ANSI['RST']}  "
              f"{ANSI['AI']}all-in{ANSI['RST']}")


# ─── Postflop strength view ────────────────────────────────────────────────
# Bucket rank_category strings into a small ordered set for compact display.
_STRENGTH_BUCKETS = [
    ("high_card",  ("High Card",)),
    ("pair",       ("One Pair", "Pair")),  # engine returns "One Pair"
    ("two_pair",   ("Two Pair",)),
    ("trips_set",  ("Three of a Kind",)),
    ("straight",   ("Straight",)),
    ("flush_plus", ("Flush", "Full House", "Four of a Kind",
                    "Straight Flush", "Royal Flush")),
]
_STRENGTH_LABELS = [b[0] for b in _STRENGTH_BUCKETS]


def _bucket_for_category(cat: str) -> str:
    for label, members in _STRENGTH_BUCKETS:
        if cat in members:
            return label
    return "high_card"  # fallback for any unexpected category


def _evaluate_visible(he, hole, board_visible) -> int:
    """Best-5 score from however many cards are visible (5/6/7)."""
    cards = list(hole) + list(board_visible)
    n = len(cards)
    if n == 5:
        return he.evaluate5(cards)
    if n == 7:
        return he.evaluate7(cards)
    if n == 6:
        # Best 5 of 6 by trying each single-card removal.
        best = 9999
        for skip in range(6):
            sub = [cards[i] for i in range(6) if i != skip]
            score = he.evaluate5(sub)
            if score < best:
                best = score
        return best
    # Pre-flop or showdown — caller shouldn't request strength on these.
    return 9999


_N_VISIBLE_BY_STREET = {0: 0, 1: 3, 2: 4, 3: 5}


def view_postflop_strength(net: DMCNet, device: torch.device,
                           n_hands: int, seed: int, use_color: bool) -> None:
    """Sample N random hands; for every model decision on flop/turn/river,
    classify made-hand strength and aggregate the model's chosen action.
    """
    he = pte.HandEvaluator.load_or_generate("")

    # counts[street][bucket] -> {action_int: count}
    counts: dict[int, dict[str, dict[int, int]]] = {
        s: {b: {} for b in _STRENGTH_LABELS} for s in (1, 2, 3)
    }

    rng = np.random.default_rng(seed)
    print(f"[postflop_strength] sampling {n_hands} random hands "
          f"(self-play rollouts; greedy model both seats)…")

    env = pte.Env(seed)
    for h in range(n_hands):
        env.reset(int(rng.integers(0, 2**62)))
        while not env.is_terminal():
            s = env.state()
            street_int = int(s.street)
            obs = env.observation()
            idx, _q = greedy_action(net, obs, device)
            chosen = int(obs.legal[idx])

            if street_int in counts:  # postflop only
                hole = list(s.hole[int(s.to_act)])
                n_visible = _N_VISIBLE_BY_STREET[street_int]
                board_vis = list(s.board)[:n_visible]
                score = _evaluate_visible(he, hole, board_vis)
                cat = pte.HandEvaluator.rank_category(score)
                bucket = _bucket_for_category(cat)
                bucket_counts = counts[street_int][bucket]
                bucket_counts[chosen] = bucket_counts.get(chosen, 0) + 1

            env.step(idx)

    # Render: one table per street.
    streets = {1: "flop", 2: "turn", 3: "river"}
    short = ["F", "C", "R25", "R33", "R50", "R75", "R1x", "R15", "R2x", "R3x", "AI"]
    for street_int, street_name in streets.items():
        total_in_street = sum(
            sum(b.values()) for b in counts[street_int].values()
        )
        print()
        print(f"== {street_name}  (n={total_in_street} model decisions) ==")
        # Header
        header = f"  {'strength':<11}"
        for s in short:
            header += f"{s:>6}"
        header += f"{'n':>7}"
        print(header)
        for bucket_label in _STRENGTH_LABELS:
            row = counts[street_int][bucket_label]
            n = sum(row.values())
            line = f"  {bucket_label:<11}"
            if n == 0:
                line += "       (no samples)" + "  " * (11 - 5)
                line += f"{0:>7}"
                print(line)
                continue
            for ai, lbl in enumerate(short):
                frac = row.get(ai, 0) / n
                cell = f"{frac:>5.2f}"
                if frac >= 0.10:
                    cell = colorize(cell, ai, use_color)
                line += f" {cell}"
            line += f"{n:>7}"
            print(line)

    print()
    print("Each cell = fraction of decisions in that strength bucket that")
    print("picked that action. Cells highlighted ≥ 0.10. Rows sum to 1.0.")


# ─── Hand trace view ───────────────────────────────────────────────────────
def view_hand_trace(net: DMCNet, device: torch.device,
                    seed: int, opponent_name: str, use_color: bool) -> None:
    """Play ONE hand: model in SB seat, opponent (or model self) in BB seat.
    Print each decision with the full Q-vector.
    """
    from evaluate.policies import (
        CallingStationPolicy, CheckFoldPolicy, RandomPolicy,
    )
    OPP = {
        "self":            None,
        "check_fold":      CheckFoldPolicy(),
        "calling_station": CallingStationPolicy(),
        "random":          RandomPolicy(),
    }
    if opponent_name not in OPP:
        print(f"[trace] unknown opponent: {opponent_name}")
        return
    opp = OPP[opponent_name]

    env = pte.Env(seed)
    env.reset(seed)

    rng = np.random.default_rng(seed)

    print(f"[hand_trace] seed={seed:#x}  model=SB  opp={opponent_name}")
    s = env.state()
    print(f"  SB hole: {pte.card_to_string(s.hole[0][0])} {pte.card_to_string(s.hole[0][1])}")
    print(f"  BB hole: {pte.card_to_string(s.hole[1][0])} {pte.card_to_string(s.hole[1][1])}")
    print(f"  board:   {' '.join(pte.card_to_string(c) for c in s.board)}")

    decision = 0
    while not env.is_terminal():
        actor = env.to_act()
        obs = env.observation()
        s = env.state()
        street_name = ["pre", "flop", "turn", "rvr"][int(s.street)]
        is_model = (actor == pte.Player.SB)

        if is_model or opp is None:
            idx, q = greedy_action(net, obs, device)
            chosen = int(obs.legal[idx])
            who = "SB/model"
            n_visible = {0: 0, 1: 3, 2: 4, 3: 5}.get(int(s.street), 5)
            board_str = " ".join(pte.card_to_string(c) for c in list(s.board)[:n_visible]) or "—"
            print(f"\n  [{decision:>2d} {street_name:>4s}] {who} pot={s.pot_chips/100:.1f}BB  "
                  f"to_call={s.to_call_chips()/100:.1f}BB  board={board_str}")
            for li, lt in enumerate(obs.legal):
                marker = "→" if li == idx else " "
                cell = colorize(ACTION_SHORT[int(lt)], int(lt), use_color)
                print(f"      {marker} {cell:>15}  q={q[li]:+.3f}")
        else:
            idx = opp.choose(obs, rng)
            chosen = int(obs.legal[idx])
            print(f"\n  [{decision:>2d} {street_name:>4s}] BB/{opponent_name}: "
                  f"{ACTION_SHORT[chosen]}")
        env.step(idx)
        decision += 1

    bb = env.payoffs_bb()
    print(f"\n  terminal: SB={bb[0]:+.2f} BB  BB={bb[1]:+.2f} BB")


# ─── Q-values dump ──────────────────────────────────────────────────────────
def view_q_values(net: DMCNet, device: torch.device,
                  seed: int, use_color: bool) -> None:
    env = pte.Env(seed)
    env.reset(seed)
    obs = env.observation()
    s = env.state()
    print(f"[q_values] seed={seed:#x}")
    print(f"  SB hole: {pte.card_to_string(s.hole[0][0])} {pte.card_to_string(s.hole[0][1])}")
    print(f"  to_act = SB (preflop, first decision)")
    idx, q = greedy_action(net, obs, device)
    print(f"  legal actions and Q-values:")
    for li, lt in enumerate(obs.legal):
        marker = "→" if li == idx else " "
        cell = colorize(ACTION_SHORT[int(lt)], int(lt), use_color)
        print(f"    {marker} {cell:>15}  q={q[li]:+.3f}")


# ─── CLI ───────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--view", type=str, default="preflop_range",
                   choices=["preflop_range", "postflop_strength",
                            "hand_trace", "q_values"])
    p.add_argument("--n-hands", type=int, default=50_000,
                   help="for preflop_range view")
    p.add_argument("--seed", type=int, default=0xC0FFEE,
                   help="for hand_trace / q_values view, or sampling seed")
    p.add_argument("--opponent", type=str, default="check_fold",
                   help="for hand_trace view")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    net = load_model(Path(args.ckpt), device)
    use_color = not args.no_color and sys.stdout.isatty()

    if args.view == "preflop_range":
        view_preflop_range(net, device, args.n_hands, args.seed, use_color)
    elif args.view == "postflop_strength":
        view_postflop_strength(net, device, args.n_hands, args.seed, use_color)
    elif args.view == "hand_trace":
        view_hand_trace(net, device, args.seed, args.opponent, use_color)
    elif args.view == "q_values":
        view_q_values(net, device, args.seed, use_color)
    return 0


if __name__ == "__main__":
    sys.exit(main())
