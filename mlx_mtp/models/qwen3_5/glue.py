"""Top-level multimodal glue for Qwen3.5 (Qwen3_5ForConditionalGeneration), pure-mlx.

Folds the qwen3_vl base Model into a single self-contained class: vision tower +
language model + image-feature scatter. The key change vs mlx_vlm: `sanitize()`
PRESERVES the 15 MTP tensors (remapping `mtp.` -> `language_model.mtp.`) instead of
dropping them, and applies the RMSNorm +1.0 convention to the MTP norms too, so the
native embedded MTP head loads strictly.
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_mtp.models.base import InputEmbeddingsFeatures
from mlx_mtp.models.qwen3_5.config import ModelConfig
from mlx_mtp.models.qwen3_5.language import LanguageModel
from mlx_mtp.models.qwen3_5.vision import VisionModel


def masked_scatter(final_embedding, image_mask_expanded, scaled_image_features):
    shape = final_embedding.shape
    feats = mx.flatten(scaled_image_features)
    flat = mx.flatten(final_embedding)
    mask_flat = mx.flatten(image_mask_expanded)
    positions = mx.array(np.where(mask_flat)[0], mx.uint32)
    flat[positions] = feats
    return mx.reshape(flat, shape)


def sanitize_key(key):
    if key.startswith("model.language_model.visual"):
        key = key.replace("model.language_model.visual", "vision_tower", 1)
    elif key.startswith("model.language_model"):
        key = key.replace("model.language_model", "language_model.model", 1)
    elif key.startswith("model.visual"):
        key = key.replace("model.visual", "vision_tower", 1)
    elif key.startswith("mtp."):
        # PRESERVE the MTP head (was dropped by mlx_vlm); bind to the native head.
        key = key.replace("mtp.", "language_model.mtp.", 1)
    elif key.startswith("lm_head"):
        key = key.replace("lm_head", "language_model.lm_head", 1)
    return key


# RMSNorm weights stored with the (1 + w) convention -> add 1.0 at load. Covers the
# main model AND the MTP head norms (pre_fc_norm_*, mtp.norm, mtp layer norms).
_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    ".pre_fc_norm_embedding.weight",
    ".pre_fc_norm_hidden.weight",
    ".mtp.norm.weight",
)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    # ---- multimodal embedding ----
    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        if pixel_values is None:
            pixel_values = kwargs.get("pixel_values_videos", None)

        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None:
            self.language_model._position_ids = None
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        hidden_states, _ = self.vision_tower(pixel_values, grid_thw)
        inputs_embeds, _ = self.merge_input_ids_with_image_features(
            hidden_states, inputs_embeds, input_ids,
            self.config.image_token_index, self.config.video_token_index,
        )
        if image_grid_thw is not None or video_grid_thw is not None:
            position_ids, rope_deltas = self.language_model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, mask
            )
            self.language_model._position_ids = position_ids
            self.language_model._rope_deltas = rope_deltas
        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    @staticmethod
    def merge_input_ids_with_image_features(
        image_features, inputs_embeds, input_ids, image_token_index, video_token_index
    ):
        special = (input_ids == image_token_index) | (input_ids == video_token_index)
        n_tokens = special.sum()
        special = mx.broadcast_to(special[..., None], inputs_embeds.shape)
        if special.sum() != image_features.size:
            raise ValueError(
                f"Image features and image tokens do not match: tokens {n_tokens}, "
                f"features {image_features.shape[0]}"
            )
        inputs_embeds = masked_scatter(inputs_embeds, special, image_features)
        return inputs_embeds, special

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        feats = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        kwargs.update({"pixel_values": pixel_values, **feats.to_dict()})
        return self.language_model(input_ids, mask=mask, cache=cache, **kwargs)

    # ---- weight key remap (+ MTP preserved + RMSNorm +1.0) ----
    def sanitize(self, weights):
        if self.config.text_config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        out = {}
        for key, value in weights.items():
            key = sanitize_key(key)
            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            # vision patch_embed Conv3d: PyTorch (out, in, kD, kH, kW) -> mlx (out, kD, kH, kW, in).
            # Guard against double-permute on reload (mlx layout already has in_channels last & small).
            if key.endswith("patch_embed.proj.weight") and value.ndim == 5 and value.shape[1] == self.vision_tower.patch_embed.in_channels:
                value = value.transpose(0, 2, 3, 4, 1)
            if any(key.endswith(sfx) for sfx in _NORM_SUFFIXES) and value.ndim == 1:
                value = value + 1.0
            out[key] = value
        return out
