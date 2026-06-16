"""Native embedded MTP (multi-token-prediction) head for mlx-mtp.

This is the piece omlx used to monkey-patch in at runtime. Here it is a first-class
`nn.Module` built INTO the Qwen3.5 LanguageModel, matching the exact 15 `mtp.*`
checkpoint tensors:

  pre_fc_norm_embedding.weight, pre_fc_norm_hidden.weight, fc.weight,
  layers.0.input_layernorm.weight, layers.0.post_attention_layernorm.weight,
  layers.0.self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}.weight,
  layers.0.mlp.{gate_proj,up_proj,down_proj}.weight, norm.weight

The single MTP layer is a full-attention `Qwen3_5DecoderLayer` (full_attention_interval=1),
so its self_attn carries the gated/partial-RoPE projections that match the checkpoint
shapes (q_proj=[12288,5120]=24*256*2, o_proj=[5120,6144]=24*256).

Forward mirrors mlx_vlm's Qwen3_5MTPDraftModel._forward_hidden, specialized to the
engine's D1 (one-draft-per-round) usage: a fresh KVCache each round, position 0.
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_mtp.models.base import create_attention_mask
from mlx_mtp.models.cache import KVCache
from mlx_mtp.models.qwen3_5.config import TextConfig
from mlx_mtp.models.qwen3_5.language import Qwen3_5DecoderLayer


class MTPHead(nn.Module):
    def __init__(self, text_config: TextConfig):
        super().__init__()
        if "moe" in getattr(text_config, "model_type", ""):
            raise NotImplementedError(
                "MTPHead supports dense Qwen3.5 only; MoE MTP checkpoints are not wired."
            )
        H = text_config.hidden_size
        n_layers = int(getattr(text_config, "mtp_num_hidden_layers", 1))

        self.fc = nn.Linear(2 * H, H, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(H, eps=text_config.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(H, eps=text_config.rms_norm_eps)
        # interval=1 => layer 0 is FULL attention (matches mtp.layers.0.self_attn.*)
        layer_config = replace(
            text_config, num_hidden_layers=n_layers, full_attention_interval=1
        )
        self.layers = [Qwen3_5DecoderLayer(args=layer_config, layer_idx=0)
                       for _ in range(n_layers)]
        self.norm = nn.RMSNorm(H, eps=text_config.rms_norm_eps)

        # bindings (set by .bind, NOT owned weights — shared with the target LM)
        self._embed = None
        self._embed_scale: float = 1.0
        self._lm_head = None

    # ---- binding to the host LanguageModel (shared embed + separate lm_head) ----
    def bind(self, language_model) -> "MTPHead":
        inner = language_model.model  # Qwen3_5Model
        self._embed = inner.embed_tokens
        self._embed_scale = float(getattr(inner, "embed_scale", 1.0))
        self._lm_head = getattr(language_model, "lm_head", None) or inner.embed_tokens.as_linear
        return self

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for _ in self.layers]

    def _forward_hidden(self, token_embed, hidden, cache, position_ids) -> mx.array:
        h = mx.concatenate(
            [self.pre_fc_norm_embedding(token_embed), self.pre_fc_norm_hidden(hidden)],
            axis=-1,
        )
        h = self.fc(h)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, cache):
            mask = (
                create_attention_mask(h, layer_cache)
                if layer_cache is not None
                else ("causal" if h.shape[1] > 1 else None)
            )
            h = layer(h, mask=mask, cache=layer_cache, position_ids=position_ids)
        return self.norm(h)

    def mtp_forward(self, pre_norm_hidden: mx.array, token: mx.array,
                    cache: Optional[List[KVCache]]) -> mx.array:
        """Draft logits for the token after `token`, given the last layer's pre-norm hidden.

        D1 usage: `cache` is a fresh per-round KVCache list, so position starts at 0.
        """
        te = self._embed(token.astype(mx.int32)) * self._embed_scale
        length = te.shape[1]
        position_ids = mx.arange(length, dtype=mx.int32)[None, :]
        h = self._forward_hidden(te, pre_norm_hidden, cache, position_ids)
        return self._lm_head(h)
