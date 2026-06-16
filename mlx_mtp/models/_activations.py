"""Tiny pure-mlx activations + MLP, vendored to sever mlx_lm.

`swiglu` and the Qwen3 dense `MLP` are one-liners; reimplemented here on mlx.nn
so nothing in mlx-mtp imports mlx_lm at runtime.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def swiglu(gate: mx.array, x: mx.array) -> mx.array:
    return nn.silu(gate) * x


class Qwen3MLP(nn.Module):
    """Dense gate/up/down SwiGLU MLP (== mlx_lm.models.qwen3.MLP), for the DFlash drafter."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
