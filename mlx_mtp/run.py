"""mlx-mtp runner — load the quant (with MTP head attached) and run/benchmark.

Vanilla AR generation (TTFT + decode TPS) and a vision test, on the mlx-mtp
8-bit quant. (Native-MTP speculative decode lives in engine.py.)
"""
from __future__ import annotations

import argparse
import time

# Attach the MTP head to mlx_vlm's qwen3_5 LanguageModel so language_model.mtp.*
# weights load cleanly (and are available to the MTP engine).
from omlx.patches.mlx_vlm_mtp import (
    apply_mlx_vlm_mtp_runtime_patch,
    set_mtp_attach_enabled,
)

set_mtp_attach_enabled(True)
apply_mlx_vlm_mtp_runtime_patch()

from mlx_vlm import load, stream_generate  # noqa: E402
from mlx_vlm.prompt_utils import apply_chat_template  # noqa: E402
from mlx_vlm.utils import load_config  # noqa: E402


def _gen(model, processor, config, prompt_text, image=None, max_tokens=128,
         temp=0.0):
    n_images = 1 if image else 0
    prompt = apply_chat_template(processor, config, prompt_text, num_images=n_images)
    images = [image] if image else None
    t0 = time.perf_counter()
    ttft = None
    text = ""
    last = None
    ntok = 0
    for r in stream_generate(model, processor, prompt, image=images,
                             max_tokens=max_tokens, temperature=temp):
        if ttft is None:
            ttft = time.perf_counter() - t0
        text += r.text
        ntok += 1
        last = r
    wall = time.perf_counter() - t0
    return {
        "text": text,
        "ttft_s": ttft,
        "wall_s": wall,
        "gen_tokens": getattr(last, "generation_tokens", ntok),
        "gen_tps": getattr(last, "generation_tps", None),
        "prompt_tokens": getattr(last, "prompt_tokens", None),
        "prompt_tps": getattr(last, "prompt_tps", None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt", default="Explain what multi-token prediction is in 2 sentences.")
    a = ap.parse_args()

    print(f">> loading {a.model}", flush=True)
    t0 = time.perf_counter()
    model, processor = load(a.model, trust_remote_code=True)
    config = load_config(a.model, trust_remote_code=True)
    print(f">> loaded in {time.perf_counter()-t0:.1f}s", flush=True)
    print(">> has MTP head attached:",
          hasattr(getattr(model, "language_model", model), "mtp"), flush=True)

    print("\n=== TEXT (vanilla AR) ===", flush=True)
    rt = _gen(model, processor, config, a.prompt, image=None, max_tokens=a.max_tokens)
    print(f"TTFT: {rt['ttft_s']*1000:.0f} ms | gen {rt['gen_tokens']} tok @ "
          f"{rt['gen_tps']:.2f} tok/s | prompt {rt['prompt_tokens']} tok "
          f"@ {rt['prompt_tps']:.1f} tok/s", flush=True)
    print("OUT:", rt["text"][:400], flush=True)

    if a.image:
        print("\n=== VISION (8-bit quant) ===", flush=True)
        rv = _gen(model, processor, config,
                  "Describe this image. What shapes and colors do you see?",
                  image=a.image, max_tokens=a.max_tokens)
        print(f"TTFT: {rv['ttft_s']*1000:.0f} ms | gen {rv['gen_tokens']} tok @ "
              f"{rv['gen_tps']:.2f} tok/s", flush=True)
        print("CAPTION:", rv["text"][:500], flush=True)


if __name__ == "__main__":
    main()
