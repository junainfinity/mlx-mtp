"""Tiny pure-mlx activations + MLP on Apple mlx.nn.

`swiglu` and the Qwen3 dense `MLP` are one-liners, implemented here directly on
mlx.nn so the package depends only on Apple MLX at runtime.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def swiglu(gate: mx.array, x: mx.array) -> mx.array:
    return nn.silu(gate) * x


class Qwen3MLP(nn.Module):
    """Dense gate/up/down SwiGLU MLP (Qwen3-style), for the DFlash drafter."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
