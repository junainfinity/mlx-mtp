"""mlx-mtp MXFP4 quantizer — vision-preserving, MTP/SSM-protected, DFlash-ready.

Quantizes the language model to **MXFP4** (OCP microscaling FP4: 4-bit E2M1 + a
shared E8M0 scale per group of 32) while keeping the parts that hate 4-bit in fp16:

  * **vision tower / projector  → fp16**  (the hard requirement)
  * **MTP head                 → fp16**  (preserve native MTP if the checkpoint has it)
  * **hybrid-SSM-sensitive      → fp16**  (a_log / dt_bias / conv1d / ssm_* — Qwen3.5 GDN)
  * everything else (attention + MLP linears, lm_head) → mxfp4

DFlash works regardless (speculation is quant-agnostic on the target). This is the
mxfp4 sibling of the affine oQ quantizer; group_size is forced to 32 (mxfp4's only
legal group). On load, mlx_vlm keeps the skipped layers fp16 automatically (they have
no `.scales`), driven by the saved `quantization` config.

    .venv-omlx/bin/python -m mlx_mtp.mxfp4_quantize \
        --src "<bf16 model dir>" --out "<out dir>"
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.utils import load, save_weights, save_config, skip_multimodal_module

GROUP = 32  # mxfp4's only supported group size
BITS = 4

# Non-weight files to copy verbatim so the output loads + chats correctly.
_COPY = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
         "chat_template.jinja", "preprocessor_config.json", "processor_config.json",
         "generation_config.json", "special_tokens_map.json", "added_tokens.json")

# Qwen3.5 hybrid-SSM params that are very sensitive to low-bit quantization.
_SSM_SENSITIVE = ("a_log", "dt_bias", "conv1d", "ssm_alpha", "ssm_beta",
                  "time_decay", "time_faaaa", "A_log")


def _protected(path: str) -> str | None:
    """Return a reason to keep this module fp16, or None to quantize it."""
    if skip_multimodal_module(path):
        return "vision"
    if "mtp" in path.lower():
        return "mtp"
    if any(s in path for s in _SSM_SENSITIVE):
        return "ssm"
    return None


def quantize(src: str, out: str) -> dict:
    t0 = time.time()
    print(f">> mxfp4 quantize: LM->mxfp4 (g{GROUP}), vision+MTP+SSM->fp16", flush=True)
    model, _processor = load(src, lazy=True)

    stats = {"quantized": 0, "fp16_vision": 0, "fp16_mtp": 0, "fp16_ssm": 0, "fp16_other": 0}

    def predicate(path, module):
        reason = _protected(path)
        if reason:
            stats[f"fp16_{reason}"] += 1
            return False
        if not hasattr(module, "to_quantized"):
            return False
        w = getattr(module, "weight", None)
        if w is None or w.shape[-1] % GROUP != 0:
            stats["fp16_other"] += 1
            return False
        stats["quantized"] += 1
        return True

    nn.quantize(model, group_size=GROUP, bits=BITS, mode="mxfp4", class_predicate=predicate)

    out_p = Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    save_weights(out_p, model, donate_weights=True)

    # config: record the mxfp4 quantization + flag vision as skipped so reload keeps fp16
    cfg = json.loads((Path(src) / "config.json").read_text())
    q = {"group_size": GROUP, "bits": BITS, "mode": "mxfp4"}
    cfg["quantization"] = q
    cfg["quantization_config"] = q
    cfg.setdefault("vision_config", {})["skip_vision"] = True
    # If the source has no MTP head, don't let the arch declare one (it would expect
    # 15 phantom params and fail strict load). Disable MTP in the config in that case.
    if stats["fp16_mtp"] == 0:
        for c in (cfg, cfg.get("text_config", {})):
            if "mtp_num_hidden_layers" in c:
                c["mtp_num_hidden_layers"] = 0
        cfg["mlx_mtp"] = {"version": 1, "backend": "mxfp4", "vision_fp16": True,
                          "mtp_preserved": False, "note": "source had no MTP head"}
    save_config(cfg, out_p / "config.json")
    for f in _COPY:
        s = Path(src) / f
        if s.exists():
            shutil.copy2(s, out_p / f)

    dt = time.time() - t0
    print(f">> done in {dt:.1f}s | {json.dumps(stats)}", flush=True)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="full-precision (bf16) source model dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    quantize(a.src, a.out)
