"""mlx-mtp — DFlash drafter integration + 3-way comparison.

Runs vanilla AR, native-MTP (our engine), and DFlash (external block-diffusion
drafter via mlx_vlm) on the same 8-bit quant, for an apples-to-apples TPS
comparison. Prereq for the DFlash+MTP hybrid.
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx

from mlx_mtp.engine import (
    load_model, vanilla_generate, mtp_generate, _lm, _prompt_ids, _eos_ids,
)
from mlx_vlm import stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.speculative.drafters import load_drafter


def load_dflash_drafter(path):
    drafter, kind = load_drafter(path, kind="dflash")
    return drafter, kind


def dflash_generate(model, processor, config, drafter, text, max_tokens=128):
    """DFlash speculative decode via mlx_vlm's generate path (draft_kind=dflash)."""
    prompt = apply_chat_template(processor, config, text, num_images=0)
    t0 = time.perf_counter()
    ttft = None
    out = ""
    last = None
    for r in stream_generate(model, processor, prompt, image=None,
                             max_tokens=max_tokens, temperature=0.0,
                             draft_model=drafter, draft_kind="dflash"):
        if ttft is None:
            ttft = time.perf_counter() - t0
        out += r.text
        last = r
    wall = time.perf_counter() - t0
    return {
        "mode": "dflash", "text": out, "ttft_s": ttft, "wall_s": wall,
        "gen_tokens": getattr(last, "generation_tokens", None),
        "tps": getattr(last, "generation_tps", None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--prompt", default="Write a short paragraph about the city of Tokyo.")
    ap.add_argument("--max-tokens", type=int, default=128)
    a = ap.parse_args()

    print(f">> loading target {a.model}", flush=True)
    model, processor, config = load_model(a.model)
    print(f">> loading DFlash drafter {a.drafter}", flush=True)
    drafter, kind = load_dflash_drafter(a.drafter)
    print(f">> drafter kind={kind}, block_size={getattr(drafter.config, 'block_size', '?')}",
          flush=True)

    # warmup
    vanilla_generate(model, processor, config, "Hello.", max_tokens=8)

    print("\n--- vanilla ---", flush=True)
    rv = vanilla_generate(model, processor, config, a.prompt, a.max_tokens)
    print(f"vanilla: {rv['tokens']} tok @ {rv['tps']:.2f} tok/s", flush=True)

    print("\n--- native MTP ---", flush=True)
    rm = mtp_generate(model, processor, config, a.prompt, a.max_tokens)
    print(f"mtp: {rm['tokens']} tok @ {rm['tps']:.2f} tok/s | accept {rm['accept_rate']*100:.0f}%",
          flush=True)

    print("\n--- DFlash ---", flush=True)
    rd = dflash_generate(model, processor, config, drafter, a.prompt, a.max_tokens)
    print(rd["text"][:200], flush=True)
    print(f"dflash: {rd['gen_tokens']} tok @ {rd['tps']:.2f} tok/s", flush=True)

    print("\n================ 3-way (vs vanilla) ================", flush=True)
    print(f"  vanilla : {rv['tps']:.2f} tok/s (1.00x)", flush=True)
    print(f"  MTP     : {rm['tps']:.2f} tok/s ({rm['tps']/rv['tps']:.2f}x)", flush=True)
    print(f"  DFlash  : {rd['tps']:.2f} tok/s ({rd['tps']/rv['tps']:.2f}x)", flush=True)
    print("====================================================", flush=True)


if __name__ == "__main__":
    main()
