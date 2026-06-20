"""Reservoir buffer for Deep CFR (token-sequence variant).

Stores `(tokens, decision_pos, target_vec, t)` quadruples.
  * `tokens` is the variable-length int16 token sequence from
    `cfr/tokenize.py` (cast back to int64 on sample for GPU upload).
  * `decision_pos` is the index of the [DECISION] token in `tokens`.
  * `target_vec` is the per-action regret/strategy vector.
  * `t` is the CFR iteration index for Linear CFR weighting.

Implementation notes:

* Algorithm R reservoir sampling (Vitter 1985) — when the buffer is full,
  an incoming entry replaces a uniformly-chosen older entry with
  probability `capacity / n_seen_total`. The empirical distribution stays
  uniform over the entire insert stream, which Deep CFR's convergence
  proof relies on.

* Variable-length token sequences are stored as a Python list of int16
  ndarrays. We could pre-pad to MAX_SEQ_LEN for contiguous storage, but
  most sequences are <30 tokens vs MAX_SEQ_LEN=160 — variable-length
  storage saves ~5x RAM at cluster scale (20M entries × avg-30 tokens ×
  int16 ≈ 1.2 GB; padded would be ~6 GB).

* On `sample(B)` we pad to the longest in-batch sequence and convert to
  int64 (PyTorch embedding expects Long).

* Linear CFR weighting: each entry carries `t`, and the refit loss
  multiplies per-sample MSE by `t/mean(t)` so later iterations dominate.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class _Sample:
    """A torch-free batch returned by ReservoirBuffer.sample()."""
    tokens:       np.ndarray   # (B, L_max) int64, PAD=0
    pad_mask:     np.ndarray   # (B, L_max) bool, True at PAD positions
    decision_pos: np.ndarray   # (B,)       int64
    target:       np.ndarray   # (B, num_actions) float32
    t:            np.ndarray   # (B,)       float32


class ReservoirBuffer:
    """Algorithm R reservoir over (token sequence, decision_pos, target, t).

    Invariants:
        * `n_seen` = total inserts attempted (grows forever).
        * `n_filled = min(n_seen, capacity)` = currently-stored entries.
        * Every entry in the buffer is a uniform sample from the stream of
          `n_seen` inputs, regardless of insertion order.

    Token sequences are stored as int16 (vocab is 106 → fits comfortably)
    and lifted to int64 only at sample time.
    """

    # Padding token id; must equal cfr.tokenize.PAD == 0.
    PAD_ID = 0

    def __init__(self,
                 capacity: int,
                 num_actions: int,
                 rng: np.random.Generator | None = None):
        self.capacity = int(capacity)
        self.num_actions = int(num_actions)
        # Variable-length per-entry token sequences. None until populated.
        self.tokens:   list[np.ndarray | None] = [None] * self.capacity
        self.dpos     = np.zeros((self.capacity,),                 dtype=np.int32)
        self.target   = np.zeros((self.capacity, self.num_actions), dtype=np.float32)
        self.t        = np.zeros((self.capacity,),                 dtype=np.float32)
        self.n_seen   = 0
        self.n_filled = 0
        self.rng = rng or np.random.default_rng()

    def __len__(self) -> int:
        return self.n_filled

    def _write(self, slot: int,
               tokens: np.ndarray,
               dpos: int,
               target: np.ndarray,
               t: int) -> None:
        # Defensive copy via int16 cast — caller may have given us int64,
        # and we want our own storage so subsequent caller mutations don't
        # leak in.
        self.tokens[slot] = np.asarray(tokens, dtype=np.int16).copy()
        self.dpos[slot]   = int(dpos)
        self.target[slot] = target
        self.t[slot]      = float(t)

    def add(self,
            tokens: np.ndarray,
            dpos: int,
            target: np.ndarray,
            t: int) -> None:
        """Insert a single (tokens, dpos, target, t) entry."""
        if self.n_seen < self.capacity:
            self._write(self.n_seen, tokens, dpos, target, t)
            self.n_filled += 1
        else:
            j = self.rng.integers(0, self.n_seen + 1)
            if j < self.capacity:
                self._write(int(j), tokens, dpos, target, t)
        self.n_seen += 1

    def sample(self, batch_size: int) -> _Sample:
        """Uniformly sample `batch_size` entries, padded to max-in-batch."""
        assert self.n_filled > 0, "sample from empty buffer"
        idx = self.rng.integers(0, self.n_filled, size=batch_size)
        # Gather sequences first to compute max length.
        seqs = [self.tokens[i] for i in idx]
        L_max = max(s.shape[0] for s in seqs)
        tokens_batch   = np.full((batch_size, L_max), self.PAD_ID, dtype=np.int64)
        pad_mask_batch = np.ones((batch_size, L_max), dtype=bool)
        for k, s in enumerate(seqs):
            L = s.shape[0]
            tokens_batch[k, :L]   = s   # int16 → int64 automatic
            pad_mask_batch[k, :L] = False
        return _Sample(
            tokens=tokens_batch,
            pad_mask=pad_mask_batch,
            decision_pos=self.dpos[idx].astype(np.int64, copy=True),
            target=self.target[idx].copy(),
            t=self.t[idx].copy(),
        )

    def state_dict(self) -> dict:
        """Checkpointing. Stores only the active region."""
        n = self.n_filled
        return {
            "capacity":    self.capacity,
            "num_actions": self.num_actions,
            "n_seen":      self.n_seen,
            "n_filled":    self.n_filled,
            "tokens":      [self.tokens[i] for i in range(n)],
            "dpos":        self.dpos[:n].copy(),
            "target":      self.target[:n].copy(),
            "t":           self.t[:n].copy(),
        }

    def load_state_dict(self, sd: dict) -> None:
        assert sd["capacity"]    == self.capacity
        assert sd["num_actions"] == self.num_actions
        n = int(sd["n_filled"])
        for i in range(n):
            self.tokens[i] = sd["tokens"][i]
        self.dpos[:n]   = sd["dpos"]
        self.target[:n] = sd["target"]
        self.t[:n]      = sd["t"]
        self.n_seen   = int(sd["n_seen"])
        self.n_filled = n


# ═══════════════════════════════════════════════════════════════════════════
# DISK PERSISTENCE (for exact checkpoint resume — buffers ARE the training data)
# ═══════════════════════════════════════════════════════════════════════════
#
# The ragged per-entry token lists are flattened into one int16 array + a
# lengths array (CSR-style), so a multi-GB reservoir saves/loads as plain numpy
# without per-entry pickling. Several buffers share one .npz (keys prefixed by
# name). Written atomically (tmp + os.replace) so a crash mid-write can't leave a
# half-file that breaks the next resume.


def _flatten_tokens(buf: "ReservoirBuffer") -> tuple[np.ndarray, np.ndarray]:
    n = buf.n_filled
    toks = [np.asarray(buf.tokens[i], dtype=np.int16) for i in range(n)]
    lengths = np.array([t.shape[0] for t in toks], dtype=np.int32)
    concat = (np.concatenate(toks) if toks
              else np.zeros(0, dtype=np.int16))
    return concat, lengths


def save_buffers(path: str, named: dict[str, "ReservoirBuffer"],
                 iteration: int) -> None:
    """Atomically write a set of named reservoir buffers to one .npz."""
    # Pure-numeric .npz (no object arrays → no pickle): load_buffers iterates the
    # caller's buffer dict, so the names need not be stored in the file.
    out: dict[str, np.ndarray] = {
        "__iteration": np.array([iteration], dtype=np.int64),
    }
    for name, buf in named.items():
        concat, lengths = _flatten_tokens(buf)
        n = buf.n_filled
        out[f"{name}__capacity"]    = np.array([buf.capacity], dtype=np.int64)
        out[f"{name}__num_actions"] = np.array([buf.num_actions], dtype=np.int64)
        out[f"{name}__n_seen"]      = np.array([buf.n_seen], dtype=np.int64)
        out[f"{name}__n_filled"]    = np.array([n], dtype=np.int64)
        out[f"{name}__tok_concat"]  = concat
        out[f"{name}__tok_len"]     = lengths
        out[f"{name}__dpos"]        = buf.dpos[:n].copy()
        out[f"{name}__target"]      = buf.target[:n].copy()
        out[f"{name}__t"]           = buf.t[:n].copy()
    tmp = f"{path}.tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(tmp, **out)                       # np.savez appends .npz to tmp
    os.replace(tmp + ".npz", path)


def load_buffers(path: str, named: dict[str, "ReservoirBuffer"]) -> int:
    """Refill the given buffers in place from a .npz written by save_buffers.

    Returns the saved iteration. Capacity/num_actions must match the buffers
    constructed by the caller (assert) — a fresh run constructs them from the
    same CLI flags, so this holds; if a flag changed, fail loudly rather than
    silently corrupt the reservoir semantics."""
    with np.load(path) as z:                   # pure-numeric file → no allow_pickle
        iteration = int(z["__iteration"][0])
        for name, buf in named.items():
            if f"{name}__n_filled" not in z:
                print(f"[buffers] WARNING: '{name}' absent in {path}; left empty.")
                continue
            cap = int(z[f"{name}__capacity"][0])
            na = int(z[f"{name}__num_actions"][0])
            assert cap == buf.capacity, (
                f"buffer '{name}' capacity {cap} != configured {buf.capacity}; "
                "match --adv-capacity/--policy-capacity to the saved run.")
            assert na == buf.num_actions, name
            n = int(z[f"{name}__n_filled"][0])
            lengths = z[f"{name}__tok_len"]
            concat = z[f"{name}__tok_concat"]
            offs = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
            for i in range(n):
                buf.tokens[i] = concat[offs[i]:offs[i + 1]].astype(np.int16).copy()
            buf.dpos[:n]   = z[f"{name}__dpos"]
            buf.target[:n] = z[f"{name}__target"]
            buf.t[:n]      = z[f"{name}__t"]
            buf.n_filled = n
            buf.n_seen = int(z[f"{name}__n_seen"][0])
    return iteration
