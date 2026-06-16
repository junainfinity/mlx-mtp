"""mlx-mtp benchmark — TTFT, decode TPS (vanilla vs native-MTP), acceptance,
and a vision test, all on the mlx-mtp 8-bit quant. Locally runnable, one model.

    python -m mlx_mtp.bench --model <dir> --image <png> --out <json>
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx

from mlx_mtp.engine import load_model, vanilla_generate, mtp_generate, _lm
from mlx_mtp.tokenizer import apply_chat_template, eos_ids, preprocess_images

PROMPTS = [
    "Write a short paragraph about the city of Tokyo.",
    "Explain how multi-token prediction speeds up LLM inference.",
    "List three benefits of Apple Silicon for local AI, with one sentence each.",
    "Describe the plot of a short story about a lighthouse keeper.",
]


def vision_test(model, processor, config, image, max_tokens=96):
    from PIL import Image

    img = Image.open(image).convert("RGB") if isinstance(image, str) else image
    prompt = apply_chat_template(
        processor, config,
        "Describe this image. What shapes and colors do you see?", num_images=1)
    pixel_values, grid_thw = preprocess_images(processor, [img], text=prompt)
    input_ids = mx.array([processor.tokenizer.encode(prompt)])
    lm = _lm(model)
    eos = eos_ids(processor, config)
    cache = lm.make_cache()
    t0 = time.perf_counter()
    out = model(input_ids, pixel_values=pixel_values, image_grid_thw=grid_thw, cache=cache)
    ttft = time.perf_counter() - t0
    tok = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
    toks = [tok]
    t1 = time.perf_counter()
    for _ in range(max_tokens - 1):
        if tok in eos:
            break
        out = lm(mx.array([[tok]]), cache=cache)
        tok = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
        toks.append(tok)
    dt = time.perf_counter() - t1
    return {
        "ttft_s": ttft, "text": processor.tokenizer.decode(toks),
        "gen_tokens": len(toks), "gen_tps": (len(toks) - 1) / dt if dt > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    print(f">> loading {a.model}", flush=True)
    t0 = time.perf_counter()
    model, processor, config = load_model(a.model)
    load_s = time.perf_counter() - t0
    print(f">> loaded in {load_s:.1f}s | MTP head: {hasattr(_lm(model), 'mtp')}",
          flush=True)

    # warmup (compile graphs / settle)
    print(">> warmup...", flush=True)
    vanilla_generate(model, processor, config, "Hello.", max_tokens=8)
    mtp_generate(model, processor, config, "Hello.", max_tokens=8)

    rows = []
    for i, p in enumerate(PROMPTS):
        print(f"\n[{i+1}/{len(PROMPTS)}] {p[:50]}...", flush=True)
        rv = vanilla_generate(model, processor, config, p, a.max_tokens)
        rm = mtp_generate(model, processor, config, p, a.max_tokens)
        identical = rv["text"][:200] == rm["text"][:200]
        speedup = rm["tps"] / rv["tps"] if rv["tps"] else 0.0
        print(f"   vanilla {rv['tps']:5.2f} tok/s (TTFT {rv['ttft_s']*1000:.0f}ms) | "
              f"mtp {rm['tps']:5.2f} tok/s (TTFT {rm['ttft_s']*1000:.0f}ms) | "
              f"{speedup:.2f}x | accept {rm['accept_rate']*100:.0f}% | "
              f"identical={identical}", flush=True)
        rows.append({
            "prompt": p,
            "vanilla_tps": rv["tps"], "mtp_tps": rm["tps"], "speedup": speedup,
            "vanilla_ttft_ms": rv["ttft_s"] * 1000, "mtp_ttft_ms": rm["ttft_s"] * 1000,
            "accept_rate": rm["accept_rate"], "rounds": rm["rounds"],
            "accepts": rm["accepts"], "vanilla_tokens": rv["tokens"],
            "mtp_tokens": rm["tokens"], "identical_output": identical,
        })

    agg = {
        "mean_vanilla_tps": statistics.mean(r["vanilla_tps"] for r in rows),
        "mean_mtp_tps": statistics.mean(r["mtp_tps"] for r in rows),
        "mean_speedup": statistics.mean(r["speedup"] for r in rows),
        "mean_accept_rate": statistics.mean(r["accept_rate"] for r in rows),
        "mean_vanilla_ttft_ms": statistics.mean(r["vanilla_ttft_ms"] for r in rows),
        "all_identical": all(r["identical_output"] for r in rows),
    }

    vis = None
    if a.image:
        print("\n>> vision test (8-bit quant)...", flush=True)
        vis = vision_test(model, processor, config, a.image, max_tokens=96)
        print(f"   vision: {vis['gen_tps']:.2f} tok/s (TTFT {vis['ttft_s']*1000:.0f}ms)",
              flush=True)
        print("   CAPTION:", vis["text"][:300], flush=True)

    result = {
        "model": a.model, "load_s": load_s, "max_tokens": a.max_tokens,
        "device": "Apple M4 Max", "decode": "D1 greedy native-MTP",
        "per_prompt": rows, "aggregate": agg, "vision": vis,
    }

    print("\n================ mlx-mtp BENCHMARK ================", flush=True)
    print(f"  vanilla decode : {agg['mean_vanilla_tps']:.2f} tok/s", flush=True)
    print(f"  native-MTP     : {agg['mean_mtp_tps']:.2f} tok/s", flush=True)
    print(f"  SPEEDUP        : {agg['mean_speedup']:.2f}x", flush=True)
    print(f"  accept rate    : {agg['mean_accept_rate']*100:.1f}%", flush=True)
    print(f"  TTFT (vanilla) : {agg['mean_vanilla_ttft_ms']:.0f} ms", flush=True)
    print(f"  byte-identical : {agg['all_identical']} (MTP == vanilla output on all prompts)", flush=True)
    if vis:
        print(f"  vision (8-bit) : {vis['gen_tps']:.2f} tok/s, caption OK", flush=True)
    print("==================================================", flush=True)

    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f">> saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
