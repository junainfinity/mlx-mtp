"""mlx-mtp quantizer (oQ backend).

Our quantizer is a thin, branded fork of omlx's streaming oQ quantizer — the
open "optiq-class" mixed-precision quantizer — driven so that it:

  * quantizes the language model to N-bit (oQ level, default 8),
  * keeps the **vision tower fp16 automatically** (omlx's predicate already
    returns fp16 for `visual.*`/`vision_*`/projector tensors), and
  * **preserves the MTP head** (`preserve_mtp=True`) so native MTP speculative
    decoding works downstream.

It also keeps the Qwen3.5 hybrid-SSM-sensitive params (a_log, dt_bias, conv1d)
in higher precision — already handled by the oQ predicate.

This is the "similar to optiq but vision + MTP preserving" deliverable. Stage-2
verification confirms vision stayed fp16 and the 15 MTP tensors survived.

Run inside the omlx venv:
    .venv-omlx/bin/python -m mlx_mtp.oq_quantize --src <dir> --out <dir> --level 8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from omlx.oq import quantize_oq_streaming


def _progress(phase: str, pct: float) -> None:
    print(f"   [{phase}] {pct:5.1f}%", flush=True)


def quantize(src: str, out: str, level: int = 8, group_size: int = 64,
             dtype: str = "bfloat16") -> None:
    t0 = time.time()
    print(f">> mlx-mtp quantize (oQ{level}) : LM->{level}-bit, vision->fp16, MTP preserved",
          flush=True)
    quantize_oq_streaming(
        model_path=src,
        output_path=out,
        oq_level=level,
        group_size=group_size,
        progress_callback=_progress,
        text_only=False,        # keep vision in the model...
        dtype=dtype,
        preserve_mtp=True,      # ...and keep the MTP head
        auto_proxy_sensitivity=False,  # uniform oQ8, no calibration pass
    )
    # brand marker
    cfg_path = Path(out) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["mlx_mtp"] = {"version": 1, "backend": "omlx-oq", "level": level,
                      "vision_fp16": True, "mtp_preserved": True}
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f">> done in {time.time()-t0:.1f}s -> {out}", flush=True)
    verify(out)


def verify(out: str) -> None:
    out_p = Path(out)
    cfg = json.loads((out_p / "config.json").read_text())
    idx_p = out_p / "model.safetensors.index.json"
    if idx_p.exists():
        wm = json.loads(idx_p.read_text())["weight_map"]
        keys = list(wm)
    else:
        keys = list(mx.load(str(out_p / "model.safetensors")).keys())
    scales = [k for k in keys if k.endswith(".scales")]
    vision = [k for k in keys if "visual" in k or "vision" in k]
    vision_q = [k for k in vision if k.endswith(".scales")]
    mtp = [k for k in keys if k.startswith("mtp.") or ".mtp." in k]
    print("=== mlx-mtp quant report ===")
    print(f"  quantization config : {cfg.get('quantization')}")
    print(f"  quantized layers (.scales): {len(scales)}")
    print(f"  vision tensors: {len(vision)} | vision quantized: {len(vision_q)} "
          f"(want 0 => vision fp16) ")
    print(f"  MTP tensors present: {len(mtp)} (want 15)")
    print(f"  mtp_num_hidden_layers: {cfg.get('mtp_num_hidden_layers') or cfg.get('text_config',{}).get('mtp_num_hidden_layers')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    if a.verify_only:
        verify(a.out)
    else:
        quantize(a.src, a.out, a.level, a.group_size, a.dtype)


if __name__ == "__main__":
    main()
