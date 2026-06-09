"""mlx-mtp engine — native embedded-MTP speculative decoding for Qwen3.5/3.6 VLMs.

This is the novel piece: neither mlx_vlm nor omlx wires the *embedded* MTP head
as the drafter for a VLM (both use an external gemma4 assistant). We drive the
model's own `language_model.mtp` head directly:

  draft  : d = mtp_forward(pre_norm_hidden_t, token_{t})            # predicts t+1
  verify : run target on [token_t, d] (one forward), capturing
           logits + pre-norm hidden + gdn (SSM) states
  accept : greedy — accept d iff d == argmax(target logits at pos 0)
           (probability-ratio min(1,p/q) for temp>0)
  rollback: language_model.rollback_speculative_cache(...) restores BOTH the
           KV cache and the Mamba/GDN SSM+conv state on rejection.

D1 (one draft / round): accept ⇒ +2 tokens (draft + bonus), reject ⇒ +1, each
at the cost of ~one target forward + one tiny MTP forward. Vanilla AR baseline
shares the same loader for an apples-to-apples TPS comparison.
"""
from __future__ import annotations

import time
from typing import Optional

import mlx.core as mx

from omlx.patches.mlx_vlm_mtp import (
    apply_mlx_vlm_mtp_runtime_patch,
    set_mtp_attach_enabled,
)

set_mtp_attach_enabled(True)
apply_mlx_vlm_mtp_runtime_patch()

from mlx_vlm import load  # noqa: E402
from mlx_vlm.prompt_utils import apply_chat_template  # noqa: E402
from mlx_vlm.utils import load_config  # noqa: E402


def _lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def _eos_ids(processor, config):
    ids = set()
    tok = getattr(processor, "tokenizer", processor)
    for v in (getattr(tok, "eos_token_id", None), config.get("eos_token_id")):
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, (list, tuple)):
            ids.update(int(x) for x in v)
    # qwen <|im_end|>
    try:
        ie = tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(ie, int) and ie >= 0:
            ids.add(ie)
    except Exception:
        pass
    return ids


def _prompt_ids(processor, config, text, n_images=0):
    prompt = apply_chat_template(processor, config, text, num_images=n_images)
    tok = getattr(processor, "tokenizer", processor)
    return mx.array([tok.encode(prompt)]), prompt


def vanilla_generate(model, processor, config, text, max_tokens=128):
    lm = _lm(model)
    eos = _eos_ids(processor, config)
    ids, _ = _prompt_ids(processor, config, text)
    cache = lm.make_cache()
    t0 = time.perf_counter()
    out = lm(ids, cache=cache)
    logits = out.logits[:, -1, :]
    tok = int(mx.argmax(logits, axis=-1).item())
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
    text_out = getattr(processor, "tokenizer", processor).decode(toks)
    n = len(toks)
    return {
        "mode": "vanilla", "text": text_out, "tokens": n,
        "ttft_s": ttft, "decode_s": decode_t,
        "tps": (n - 1) / decode_t if decode_t > 0 else 0.0,
    }


def mtp_generate(model, processor, config, text, max_tokens=128):
    """Native embedded-MTP D1 speculative decode (greedy)."""
    lm = _lm(model)
    eos = _eos_ids(processor, config)
    last_idx = len(lm.model.layers) - 1
    ids, _ = _prompt_ids(processor, config, text)
    cache = lm.make_cache()

    t0 = time.perf_counter()
    out = lm(ids, cache=cache, capture_layer_ids=[last_idx])
    logits = out.logits[:, -1, :]
    hidden = out.hidden_states[0][:, -1:, :]          # pre-norm hidden_{n-1}
    cur = int(mx.argmax(logits, axis=-1).item())       # bonus token t0
    mx.eval(cur)
    ttft = time.perf_counter() - t0

    toks = [cur]
    rounds = 0
    accepts = 0
    t1 = time.perf_counter()
    while len(toks) < max_tokens:
        if cur in eos:
            break
        rounds += 1
        # --- draft 1 token with the embedded MTP head ---
        mtp_cache = lm.make_mtp_cache()
        mlogits = lm.mtp_forward(hidden, mx.array([[cur]]), mtp_cache)
        d = int(mx.argmax(mlogits[:, -1, :], axis=-1).item())
        # --- verify: target forward on [cur, d] ---
        vout = lm(mx.array([[cur, d]]), cache=cache, capture_layer_ids=[last_idx])
        vlogits = vout.logits          # [1, 2, V]
        vhidden = vout.hidden_states[0]  # [1, 2, H] pre-norm
        r = int(mx.argmax(vlogits[:, 0, :], axis=-1).item())
        if d == r:
            # accept: confirm d (t+1) and bonus s (t+2)
            s = int(mx.argmax(vlogits[:, 1, :], axis=-1).item())
            toks.append(d)
            if d not in eos:
                toks.append(s)
            cur = s
            hidden = vhidden[:, 1:2, :]
            accepts += 1
            lm.rollback_speculative_cache(cache, vout.gdn_states, 1, 2)
        else:
            # reject: confirm r (t+1), drop d
            toks.append(r)
            cur = r
            hidden = vhidden[:, 0:1, :]
            lm.rollback_speculative_cache(cache, vout.gdn_states, 0, 2)
    decode_t = time.perf_counter() - t1
    text_out = getattr(processor, "tokenizer", processor).decode(toks)
    n = len(toks)
    return {
        "mode": "mtp", "text": text_out, "tokens": n,
        "ttft_s": ttft, "decode_s": decode_t,
        "tps": (n - 1) / decode_t if decode_t > 0 else 0.0,
        "rounds": rounds, "accepts": accepts,
        "accept_rate": accepts / rounds if rounds else 0.0,
    }


def load_model(path):
    model, processor = load(path, trust_remote_code=True)
    config = load_config(path, trust_remote_code=True)
    return model, processor, config


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
    print(f"vanilla: {rv['tokens']} tok | TTFT {rv['ttft_s']*1000:.0f}ms | "
          f"{rv['tps']:.2f} tok/s", flush=True)
    print("\n--- mtp ---", flush=True)
    rm = mtp_generate(model, processor, config, a.prompt, a.max_tokens)
    print(rm["text"][:300], flush=True)
    print(f"mtp: {rm['tokens']} tok | TTFT {rm['ttft_s']*1000:.0f}ms | "
          f"{rm['tps']:.2f} tok/s | accept {rm['accept_rate']*100:.1f}% "
          f"({rm['accepts']}/{rm['rounds']})", flush=True)
    print(f"\n>> MTP speedup: {rm['tps']/rv['tps']:.2f}x", flush=True)
