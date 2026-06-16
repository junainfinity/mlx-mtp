"""Pure-mlx loader for mlx-mtp checkpoints (bf16 or mxfp4/mxfp8).

load(path) -> (model, processor, config). Uses only mlx.core/mlx.nn + safetensors +
the tokenizer boundary. Rebuilds QuantizedLinear/QuantizedEmbedding only where the
checkpoint actually carries `.scales` (vision/MTP/SSM stay fp16), driven by the saved
`quantization` config — keeping the quantize<->reload contract exact.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_mtp.models.qwen3_5.config import ModelConfig
from mlx_mtp.models.qwen3_5.glue import Model


def load_config(path: str) -> dict:
    return json.loads((Path(path) / "config.json").read_text())


def build_model_from_config(cfg: dict) -> Model:
    # ensure the MTP layer count survives into TextConfig (top-level mirror)
    tcfg = cfg.get("text_config", {})
    if "mtp_num_hidden_layers" not in tcfg and "mtp_num_hidden_layers" in cfg:
        tcfg = {**tcfg, "mtp_num_hidden_layers": cfg["mtp_num_hidden_layers"]}
        cfg = {**cfg, "text_config": tcfg}
    mc = ModelConfig.from_dict(cfg)
    return Model(mc)


def load(path: str) -> Tuple[Model, object, dict]:
    cfg = load_config(path)
    model = build_model_from_config(cfg)

    # 1. gather weights
    weights = {}
    for f in sorted(glob.glob(str(Path(path) / "*.safetensors"))):
        weights.update(mx.load(f))

    # 2. sanitize (key remap, conv1d, RMSNorm+1.0, MTP preserved)
    weights = model.sanitize(weights)

    # 3. rebuild quantized layers exactly where the checkpoint has .scales
    qcfg = cfg.get("quantization")
    if qcfg:
        quantized_paths = {k[: -len(".scales")] for k in weights if k.endswith(".scales")}

        def class_predicate(p, m):
            return p in quantized_paths and hasattr(m, "to_quantized")

        nn.quantize(
            model,
            group_size=qcfg["group_size"],
            bits=qcfg["bits"],
            mode=qcfg.get("mode", "affine"),
            class_predicate=class_predicate,
        )

    # 4. strict load (config drives which params exist)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())

    # 5. bind the native MTP head (shared embed + lm_head)
    lm = model.language_model
    if hasattr(lm, "mtp"):
        lm.mtp.bind(lm)

    # 6. tokenizer / processor at the I/O boundary
    from mlx_mtp.tokenizer import load_processor

    processor = load_processor(path)
    return model, processor, cfg
