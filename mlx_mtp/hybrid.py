"""mlx-mtp — DFlash + native-MTP HYBRID speculative decoding.

Both drafters share the same round shape (draft block -> verify [bonus, block]
in one target forward -> _speculative_walk greedy accept -> rollback_speculative_cache).
They differ only in the draft step and their state:
  * native MTP  : drafts 1 token via the embedded head; STATELESS across rounds
                  (fresh mtp_cache each round).
  * DFlash      : drafts a block_size-1 block via the external diffusion drafter;
                  carries its own draft_cache across rounds.

The hybrid runs a single unified loop and picks the drafter PER ROUND by an EWMA
of tokens-emitted-per-round (an adaptive bandit). Because DFlash's draft_cache
desyncs if we skip rounds, we reset() it when switching back to DFlash, and use a
min-dwell so the reset cost is amortized. Each round we capture the UNION of the
layers both drafters need, so whichever is chosen next round has its hidden ready.

This is the concrete answer to "can DFlash and native MTP be used together": yes,
via per-round adaptive selection over a shared verify/rollback loop.
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx

from mlx_mtp.engine import load_model, vanilla_generate, mtp_generate, _lm
from mlx_mtp.tokenizer import eos_ids as _eos_ids
from mlx_mtp.tokenizer import prompt_ids as _prompt_ids
from mlx_mtp.dflash import load_dflash_drafter
from mlx_mtp.speculative import _speculative_walk


def _argmax_sampler(logits):
    return mx.argmax(logits, axis=-1)


def dflash_generate(model, processor, config, drafter, text, max_tokens=128,
                    block_size=16):
    """Standalone DFlash block-diffusion speculative decode (greedy).

    Shares the MTP verify/rollback loop: draft a (block_size-1)-token block from the
    target hidden at the drafter's target_layer_ids, verify [bonus, block] in one
    target forward, greedily accept via _speculative_walk, then roll back KV+GDN state.
    """
    lm = _lm(model)
    eos = _eos_ids(processor, config)
    cap = sorted(set(drafter.config.target_layer_ids))
    ids, _ = _prompt_ids(processor, config, text)
    cache = lm.make_cache()

    t0 = time.perf_counter()
    out = lm(ids, cache=cache, capture_layer_ids=cap)
    h_df = mx.concatenate([h[:, -1:, :] for h in out.hidden_states], axis=-1)
    cur = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
    mx.eval(cur)
    ttft = time.perf_counter() - t0

    toks = [cur]
    draft_cache = drafter.reset(model)
    t1 = time.perf_counter()
    while len(toks) < max_tokens:
        if cur in eos:
            break
        budget = max_tokens - len(toks)
        draft_tokens = drafter.draft_block(cur, h_df, draft_cache, block_size,
                                           _argmax_sampler, mx.int32)
        verify_in = mx.concatenate(
            [mx.array([[cur]], dtype=mx.int32), draft_tokens.astype(mx.int32)], axis=1)
        vout = lm(verify_in, cache=cache, capture_layer_ids=cap)
        target_tokens = mx.argmax(vout.logits, axis=-1)
        accepted, new = _speculative_walk(draft_tokens, target_tokens, budget)
        for tk in new:
            toks.append(int(tk))
        cur = int(toks[-1])
        h_full = mx.concatenate(vout.hidden_states, axis=-1)
        h_df = h_full[:, accepted:accepted + 1, :]
        lm.rollback_speculative_cache(cache, vout.gdn_states, accepted, block_size)
        mx.eval(cur)
    decode_t = time.perf_counter() - t1
    text_out = processor.tokenizer.decode(toks) if hasattr(processor, "tokenizer") else processor.decode(toks)
    n = len(toks)
    return {"mode": "dflash", "text": text_out, "tokens": n, "ttft_s": ttft,
            "decode_s": decode_t, "tps": (n - 1) / decode_t if decode_t > 0 else 0.0}


def hybrid_generate(model, processor, config, drafter, text,
                    max_tokens=128, min_dwell=4, dflash_block=16):
    """Adaptive per-round DFlash+MTP hybrid (greedy)."""
    lm = _lm(model)
    eos = _eos_ids(processor, config)
    last_idx = len(lm.model.layers) - 1
    tgt_layers = list(drafter.config.target_layer_ids)
    cap = sorted(set(tgt_layers) | {last_idx})
    i_mtp = cap.index(last_idx)
    i_df = [cap.index(l) for l in tgt_layers]

    def split_hidden(hs):
        h_mtp = hs[i_mtp]
        h_df = mx.concatenate([hs[j] for j in i_df], axis=-1)
        return h_mtp, h_df

    ids, _ = _prompt_ids(processor, config, text)
    cache = lm.make_cache()

    t0 = time.perf_counter()
    out = lm(ids, cache=cache, capture_layer_ids=cap)
    h_mtp, h_df = split_hidden(out.hidden_states)
    h_mtp = h_mtp[:, -1:, :]; h_df = h_df[:, -1:, :]
    cur = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
    mx.eval(cur)
    ttft = time.perf_counter() - t0

    toks = [cur]
    # bandit state: EWMA of tokens/SECOND (wall-clock throughput, not tokens/round —
    # DFlash's big blocks win tokens/round but are slower per round).
    ewma = {"mtp": 0.0, "dflash": 0.0}
    rounds = {"mtp": 0, "dflash": 0}
    draft_cache = drafter.reset(model)
    last_choice = "dflash"
    dwell = 0

    t1 = time.perf_counter()
    while len(toks) < max_tokens:
        if cur in eos:
            break
        budget = max_tokens - len(toks)
        # --- choose drafter: probe each arm once (min_dwell rounds each), then
        #     exploit the higher-EWMA arm; stay min_dwell rounds per switch. ---
        if dwell > 0 and last_choice is not None:
            choice = last_choice
        elif rounds["mtp"] == 0:
            choice = "mtp"
        elif rounds["dflash"] == 0:
            choice = "dflash"               # forced exploration of DFlash
        else:
            choice = "mtp" if ewma["mtp"] >= ewma["dflash"] else "dflash"
        if choice != last_choice:
            if choice == "dflash":
                draft_cache = drafter.reset(model)  # resync DFlash on switch-in
            dwell = min_dwell
        last_choice = choice
        dwell -= 1
        r_t0 = time.perf_counter()

        # --- draft ---
        if choice == "mtp":
            mtp_cache = lm.make_mtp_cache()
            mlogits = lm.mtp_forward(h_mtp, mx.array([[cur]]), mtp_cache)
            d = int(mx.argmax(mlogits[:, -1, :], axis=-1).item())
            draft_tokens = mx.array([[d]])
            bs = 2
        else:
            draft_tokens = drafter.draft_block(
                cur, h_df, draft_cache, dflash_block, _argmax_sampler, mx.int32)
            bs = dflash_block

        # --- verify (shared) ---
        verify_input = mx.concatenate(
            [mx.array([[cur]], dtype=mx.int32), draft_tokens.astype(mx.int32)], axis=1)
        vout = lm(verify_input, cache=cache, capture_layer_ids=cap)
        target_tokens = mx.argmax(vout.logits, axis=-1)
        accepted, new = _speculative_walk(draft_tokens, target_tokens, budget)

        for tk in new:
            toks.append(int(tk))
        cur = int(toks[-1])
        h_mtp_full, h_df_full = split_hidden(vout.hidden_states)
        keep = accepted + 1
        h_mtp = h_mtp_full[:, accepted:keep, :]
        h_df = h_df_full[:, accepted:keep, :]
        lm.rollback_speculative_cache(cache, vout.gdn_states, accepted, bs)
        mx.eval(cur)
        round_dt = time.perf_counter() - r_t0

        rounds[choice] += 1
        if rounds[choice] > 1 and round_dt > 0:   # skip compile-inflated 1st round
            rtps = len(new) / round_dt            # tokens / SECOND this round
            ewma[choice] = rtps if ewma[choice] == 0 else 0.6 * ewma[choice] + 0.4 * rtps

    decode_t = time.perf_counter() - t1
    text_out = getattr(processor, "tokenizer", processor).decode(toks)
    n = len(toks)
    return {
        "mode": "hybrid", "text": text_out, "tokens": n,
        "ttft_s": ttft, "decode_s": decode_t,
        "tps": (n - 1) / decode_t if decode_t > 0 else 0.0,
        "rounds": rounds, "ewma": ewma,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--prompt", default="Write a short paragraph about the city of Tokyo.")
    ap.add_argument("--max-tokens", type=int, default=128)
    a = ap.parse_args()

    model, processor, config = load_model(a.model)
    drafter, kind = load_dflash_drafter(a.drafter)
    print(f">> drafter kind={kind} block={getattr(drafter.config,'block_size','?')}", flush=True)
    vanilla_generate(model, processor, config, "Hello.", max_tokens=8)  # warmup

    rv = vanilla_generate(model, processor, config, a.prompt, a.max_tokens)
    rm = mtp_generate(model, processor, config, a.prompt, a.max_tokens)
    rd = dflash_generate(model, processor, config, drafter, a.prompt, a.max_tokens)
    rh = hybrid_generate(model, processor, config, drafter, a.prompt, a.max_tokens)

    print("\n--- hybrid output ---", flush=True)
    print(rh["text"][:200], flush=True)
    print("\n================ DFlash + native MTP ================", flush=True)
    print(f"  vanilla : {rv['tps']:.2f} tok/s (1.00x)", flush=True)
    print(f"  MTP     : {rm['tps']:.2f} tok/s ({rm['tps']/rv['tps']:.2f}x)", flush=True)
    print(f"  DFlash  : {rd['tps']:.2f} tok/s ({rd['tps']/rv['tps']:.2f}x)", flush=True)
    print(f"  HYBRID  : {rh['tps']:.2f} tok/s ({rh['tps']/rv['tps']:.2f}x) "
          f"rounds={rh['rounds']}", flush=True)
    print("====================================================", flush=True)


if __name__ == "__main__":
    main()
