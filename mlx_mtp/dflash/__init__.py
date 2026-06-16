"""DFlash block-diffusion drafter for mlx-mtp (pure mlx).

DFlash is the optional EXTERNAL drafter path (e.g. z-lab/Qwen3.6-27B-DFlash): a
separate checkpoint that drafts a block of tokens from the target's hidden states at
`target_layer_ids`. Used standalone (`dflash_generate`) or fused per-round with the
native MTP head (`mlx_mtp.hybrid.hybrid_generate`). Requires the drafter weights on disk.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Tuple

import mlx.core as mx

from mlx_mtp.dflash.config import DFlashConfig
from mlx_mtp.dflash.model import DFlashDraftModel

__all__ = ["DFlashConfig", "DFlashDraftModel", "load_dflash_drafter"]


def load_dflash_drafter(path: str) -> Tuple[DFlashDraftModel, str]:
    """Load a DFlash drafter checkpoint -> (drafter, 'dflash'). Bind to the target via reset()."""
    cfg = json.loads((Path(path) / "config.json").read_text())
    drafter = DFlashDraftModel(DFlashConfig.from_dict(cfg))
    weights = {}
    for f in sorted(glob.glob(str(Path(path) / "*.safetensors"))):
        weights.update(mx.load(f))
    if hasattr(drafter, "sanitize"):
        weights = drafter.sanitize(weights)
    drafter.load_weights(list(weights.items()), strict=False)
    mx.eval(drafter.parameters())
    return drafter, "dflash"
