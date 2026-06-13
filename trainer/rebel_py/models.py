"""ReBeL value network (and a numpy adapter the solver queries).

Faithful to ``trainer/rebel/rebel/cfvpy/models.py`` (``Net2``): a GELU MLP whose
output layer is scaled by 0.01 at init so the net's initial predictions are near
zero. The paper trains a *single* value net across streets — the street/board is
part of the query — so one ``ValueNet`` instance serves every depth-limited leaf.

Input  : the PBS query (see ``pbs.encode_query``), dim = ``query_dim``.
Output : per-hand counterfactual values for the *traverser*, assuming the
         opponent's belief vector in the query is normalized to sum 1
         (the solver rescales by the opponent's true reach mass at the leaf).

The ``NetValueFn`` adapter is the bridge the solver expects: it takes a numpy
``(N, query_dim)`` batch of queries and returns numpy ``(N, NUM_HANDS)`` values,
handling device placement and eval/no-grad. Training lives in ``train.py``.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .hand_index import NUM_HANDS


class GELU(nn.Module):
    def forward(self, x):
        return nn.functional.gelu(x)


def build_mlp(*, n_in: int, n_hidden: int, n_layers: int,
              out_size: int | None = None, use_layer_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = n_in
    for _ in range(n_layers):
        layers.append(nn.Linear(last, n_hidden))
        if use_layer_norm:
            layers.append(nn.LayerNorm(n_hidden))
        layers.append(GELU())
        last = n_hidden
    if out_size is not None:
        layers.append(nn.Linear(last, out_size))
    return nn.Sequential(*layers)


class ValueNet(nn.Module):
    """MLP value net: query_dim → NUM_HANDS. Output ×0.01 init (≈0 predictions)."""

    def __init__(self, query_dim: int, *, n_hidden: int = 512, n_layers: int = 3,
                 use_layer_norm: bool = True, out_size: int = NUM_HANDS):
        super().__init__()
        self.query_dim = query_dim
        self.out_size = out_size
        self.body = build_mlp(n_in=query_dim, n_hidden=n_hidden, n_layers=n_layers,
                              use_layer_norm=use_layer_norm)
        self.head = nn.Linear(n_hidden if n_layers > 0 else query_dim, out_size)
        with torch.no_grad():
            self.head.weight.data *= 0.01
            self.head.bias.data *= 0.01

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(q))


class NetValueFn:
    """Numpy callable the solver uses at net-leaf nodes.

    ``__call__(queries[N, qdim]) -> values[N, NUM_HANDS]``. Pure inference: eval
    mode, no grad, on the net's device. Card-removal / opp-reach scaling is the
    solver's job — this returns the net's raw (normalized-belief) predictions.
    """

    def __init__(self, net: ValueNet, device: str | torch.device = "cpu"):
        self.net = net.to(device)
        self.device = torch.device(device)

    def __call__(self, queries: np.ndarray) -> np.ndarray:
        if queries.shape[0] == 0:
            return np.zeros((0, self.net.out_size), dtype=np.float64)
        self.net.eval()
        with torch.no_grad():
            q = torch.as_tensor(queries, dtype=torch.float32, device=self.device)
            v = self.net(q)
        return v.double().cpu().numpy()
