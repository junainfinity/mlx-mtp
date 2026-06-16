"""mlx-mtp runner — load a pure-mlx checkpoint and run text (+vision) generation.

Vanilla AR and native-MTP speculative decoding via mlx_mtp.engine; optional vision
smoke via the model's multimodal forward. No mlx_vlm / omlx.
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx

from mlx_mtp.engine import load_model, vanilla_generate, mtp_generate, _lm
from mlx_mtp.tokenizer import apply_chat_template, eos_ids, preprocess_images


def _vision_generate(model, processor, config, prompt_text, image, max_tokens=128):
    from PIL import Image

    img = Image.open(image).convert("RGB")
    prompt = apply_chat_template(processor, config, prompt_text, num_images=1)
    pixel_values, grid_thw = preprocess_images(processor, [img], text=prompt)
    input_ids = mx.array([processor.tokenizer.encode(prompt)])
    lm = _lm(model)
    eos = eos_ids(processor, config)
    cache = lm.make_cache()
    out = model(input_ids, pixel_values=pixel_values, image_grid_thw=grid_thw, cache=cache)
    tok = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
    toks = [tok]
    for _ in range(max_tokens - 1):
        if tok in eos:
            break
        out = lm(mx.array([[tok]]), cache=cache)
        tok = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
        toks.append(tok)
    return processor.tokenizer.decode(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt", default="Explain what multi-token prediction is in 2 sentences.")
    a = ap.parse_args()

    print(f">> loading {a.model}", flush=True)
    t0 = time.perf_counter()
    model, processor, config = load_model(a.model)
    print(f">> loaded in {time.perf_counter()-t0:.1f}s", flush=True)
    print(">> MTP head attached:", hasattr(_lm(model), "mtp"), flush=True)

    print("\n=== TEXT (vanilla AR) ===", flush=True)
    rt = vanilla_generate(model, processor, config, a.prompt, a.max_tokens)
    print(f"{rt['tokens']} tok @ {rt['tps']:.2f} tok/s | TTFT {rt['ttft_s']*1000:.0f}ms", flush=True)
    print("OUT:", rt["text"][:400], flush=True)

    print("\n=== TEXT (native MTP) ===", flush=True)
    rm = mtp_generate(model, processor, config, a.prompt, a.max_tokens)
    print(f"{rm['tokens']} tok @ {rm['tps']:.2f} tok/s | accept {rm['accept_rate']*100:.0f}% "
          f"({rm['accepts']}/{rm['rounds']})", flush=True)
    if rt["tps"] > 0:
        print(f">> MTP speedup: {rm['tps']/rt['tps']:.2f}x", flush=True)

    if a.image:
        print("\n=== VISION ===", flush=True)
        cap = _vision_generate(model, processor, config,
                               "Describe this image in detail.", a.image, a.max_tokens)
        print("CAPTION:", cap[:500], flush=True)


if __name__ == "__main__":
    main()
