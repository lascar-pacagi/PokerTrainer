"""Bridge between pokertrainer_engine (C++) and PyTorch tensors.

The engine returns numpy float32 arrays via pybind11:
    obs.x : (X_DIM,)            float32   # includes flat action history + valid-mask
    obs.a : (n_legal, A_DIM)    float32
    obs.legal_idx : (n_legal,)  int8  — ActionType indices in 0..9

This module keeps engine ⇆ torch conversions in one place so the rest of the
DMC code stays agnostic to where the numpy arrays came from.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


def obs_to_tensors(obs: Any, device: torch.device | str = "cpu") -> dict:
    """Convert a pokertrainer_engine.EncodedState to a dict of torch tensors."""
    return {
        "x":         torch.from_numpy(obs.x).to(device),
        "a":         torch.from_numpy(obs.a).to(device),
        "legal_idx": torch.from_numpy(np.asarray(obs.legal_idx, dtype=np.int64)).to(device),
    }


def pick_action_row(obs_a: np.ndarray, taken_idx: int) -> np.ndarray:
    """Return the A_DIM-vector row of `obs.a` for the action actually taken.

    `taken_idx` is the index within the legal-actions vector (0..n_legal-1),
    NOT the ActionType id.
    """
    return obs_a[taken_idx].copy()
