"""mlx-mtp engine — native embedded-MTP speculative decoding for Qwen3.5/3.6 VLMs.

Pure mlx.core/mlx.nn (+ tokenizer boundary). The MTP head is now built into the model
(mlx_mtp.models.qwen3_5), so there is NO omlx runtime patch and NO mlx_vlm import.

  draft  : d = lm.mtp_forward(pre_norm_hidden_t, token_t)        # embedded MTP head predicts t+1
  verify : run target on [token_t, d] in one forward, capturing logits + pre-norm hidden + gdn states
  accept : greedy — accept d iff d == argmax(target logits at pos 0)
  rollback: lm.rollback_speculative_cache(...) restores BOTH the KV cache and the GDN/SSM state.

D1 (one draft/round): accept => +2 tokens, reject => +1, at the cost of ~one target
forward + one tiny MTP forward. Vanilla AR shares the loader for an apples-to-apples TPS.
"""
from __future__ import annotations

import time
from typing import Optional

import mlx.core as mx

from mlx_mtp.loader import load as _load
from mlx_mtp.loader import load_config
from mlx_mtp.tokenizer import apply_chat_template, eos_ids as _eos_set, prompt_ids


def _lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def load_model(path):
    model, processor, config = _load(path)
    return model, processor, config


def vanilla_generate(model, processor, config, text, max_tokens=128):
    lm = _lm(model)
    eos = _eos_set(processor, config)
    ids, _ = prompt_ids(processor, config, text)
    cache = lm.make_cache()
    t0 = time.perf_counter()
    out = lm(ids, cache=cache)
    tok = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
    mx.eval(tok)
    ttft = time.perf_counter() - t0
    toks = [tok]
    t1 = time.perf_counter()
    for _ in range(max_tokens - 1):
        if tok in eos:
            break
        out = lm(mx.array([[tok]]), cache=cache)
        tok = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
        toks.append(tok)
    decode_t = time.perf_counter() - t1
    text_out = processor.tokenizer.decode(toks) if hasattr(processor, "tokenizer") else processor.decode(toks)
    n = len(toks)
    return {
        "mode": "vanilla", "text": text_out, "tokens": n,
        "ttft_s": ttft, "decode_s": decode_t,
        "tps": (n - 1) / decode_t if decode_t > 0 else 0.0,
    }


def mtp_generate(model, processor, config, text, max_tokens=128):
    """Native embedded-MTP D1 speculative decode (greedy)."""
    lm = _lm(model)
    eos = _eos_set(processor, config)
    last_idx = len(lm.model.layers) - 1
    ids, _ = prompt_ids(processor, config, text)
    cache = lm.make_cache()

    t0 = time.perf_counter()
    out = lm(ids, cache=cache, capture_layer_ids=[last_idx])
    hidden = out.hidden_states[0][:, -1:, :]            # pre-norm hidden_{n-1}
    cur = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())  # bonus token t0
    mx.eval(cur)
    ttft = time.perf_counter() - t0

    toks = [cur]
    rounds = accepts = 0
    t1 = time.perf_counter()
    while len(toks) < max_tokens:
        if cur in eos:
            break
        rounds += 1
        # draft 1 token with the embedded MTP head (fresh stateless cache)
        mtp_cache = lm.make_mtp_cache()
        mlogits = lm.mtp_forward(hidden, mx.array([[cur]]), mtp_cache)
        d = int(mx.argmax(mlogits[:, -1, :], axis=-1).item())
        # verify: target forward on [cur, d]
        vout = lm(mx.array([[cur, d]]), cache=cache, capture_layer_ids=[last_idx])
        vlogits = vout.logits
        vhidden = vout.hidden_states[0]
        r = int(mx.argmax(vlogits[:, 0, :], axis=-1).item())
        if d == r:
            s = int(mx.argmax(vlogits[:, 1, :], axis=-1).item())
            toks.append(d)
            if d not in eos:
                toks.append(s)
            cur = s
            hidden = vhidden[:, 1:2, :]
            accepts += 1
            lm.rollback_speculative_cache(cache, vout.gdn_states, 1, 2)
        else:
            toks.append(r)
            cur = r
            hidden = vhidden[:, 0:1, :]
            lm.rollback_speculative_cache(cache, vout.gdn_states, 0, 2)
    decode_t = time.perf_counter() - t1
    text_out = processor.tokenizer.decode(toks) if hasattr(processor, "tokenizer") else processor.decode(toks)
    n = len(toks)
    return {
        "mode": "mtp", "text": text_out, "tokens": n,
        "ttft_s": ttft, "decode_s": decode_t,
        "tps": (n - 1) / decode_t if decode_t > 0 else 0.0,
        "rounds": rounds, "accepts": accepts,
        "accept_rate": accepts / rounds if rounds else 0.0,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Write a short paragraph about the city of Tokyo.")
    ap.add_argument("--max-tokens", type=int, default=128)
    a = ap.parse_args()
    model, processor, config = load_model(a.model)
    print(">> MTP head:", hasattr(_lm(model), "mtp"), flush=True)
    print("\n--- vanilla ---", flush=True)
    rv = vanilla_generate(model, processor, config, a.prompt, a.max_tokens)
    print(rv["text"][:300], flush=True)
    print(f"vanilla: {rv['tokens']} tok | TTFT {rv['ttft_s']*1000:.0f}ms | {rv['tps']:.2f} tok/s", flush=True)
    print("\n--- mtp ---", flush=True)
    rm = mtp_generate(model, processor, config, a.prompt, a.max_tokens)
    print(rm["text"][:300], flush=True)
    print(f"mtp: {rm['tokens']} tok | TTFT {rm['ttft_s']*1000:.0f}ms | {rm['tps']:.2f} tok/s | "
          f"accept {rm['accept_rate']*100:.1f}% ({rm['accepts']}/{rm['rounds']})", flush=True)
    if rv["tps"] > 0:
        print(f"\n>> MTP speedup: {rm['tps']/rv['tps']:.2f}x", flush=True)
