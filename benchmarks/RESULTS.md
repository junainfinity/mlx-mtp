# mlx-mtp — RESULTS

**A from-scratch replication of MTPLX (native MTP speculative decoding) + optiq/omlx (mixed quant), combined into one pipeline that runs a *vision* model with its *embedded* MTP head as a self-drafter — on Apple Silicon.**

Project: `./` · package `mlx_mtp/` · venv `.venv-omlx`
Model: `osmQwopus3.6-27B` → our 8-bit quant `…/AI Models/osmapi/osmQwopus3.6-27B-mlxmtp-oQ8`
Build story: [[mlx-mtp — Build Log]]

## What we built (all three, working)
1. **mlx-mtp quantizer** (`mlx_mtp/oq_quantize.py`) — fork of omlx's oQ: LM→8-bit, **vision→fp16 automatically**, **MTP head preserved**. 52 GB → **28 GB** (505 quantized layers; vision 333/0 quantized; MTP 15 tensors preserved).
2. **mlx-mtp engine** (`mlx_mtp/engine.py`) — **native embedded-MTP** speculative decode (the novel piece). Drives the model's own `language_model.mtp` head as drafter: draft → verify(1 forward, capture logits+hidden+SSM) → greedy accept → `rollback_speculative_cache` (restores KV **and** Mamba/GDN state). Neither mlx_vlm nor omlx wires the embedded head for a VLM — both use an external gemma4 drafter.
3. **mlx-mtp benchmark** (`mlx_mtp/bench.py`) — vanilla vs MTP + vision, on the 8-bit quant.

## Benchmark — Apple M4 Max, 8-bit quant, greedy, D1
Raw JSON: `mlx-mtp-benchmark.json`.

| Metric | Value |
|---|---|
| **Vanilla AR decode** | **15.53 tok/s** (mean of 4 prompts) |
| **Native-MTP decode** | **23.09 tok/s** |
| **Speedup** | **1.49× mean** (range 1.40–1.60×) |
| **Draft acceptance** | **65.5%** (range 59–79%) |
| **TTFT** (vanilla) | **159 ms** |
| **Vision on 8-bit** | **16.19 tok/s**, TTFT 1422 ms, caption correct |
| Load time | 1.5 s |

Per-prompt:
| Prompt | vanilla | mtp | speedup | accept | identical |
|---|---|---|---|---|---|
| Tokyo paragraph | 15.74 | 25.13 | 1.60× | 79% | ✓ |
| MTP explanation | 15.53 | 22.99 | 1.48× | 62% | ✓ |
| Apple Silicon benefits | 15.50 | 21.71 | 1.40× | 59% | ✗* |
| Lighthouse story | 15.36 | 22.55 | 1.47× | 63% | ✓ |

\* 3/4 byte-identical to vanilla. The 1 divergence is a **near-tie argmax flip**: the 2-token *verify* forward is not bit-identical to single-step AR on a quantized model. Speculative decoding is exact w.r.t. the **verify-pass** distribution (and distribution-exact with residual sampling at temp>0), not bit-identical to AR. Not a correctness bug.

**Vision caption (8-bit quant):** correctly identified the red circle (upper-left), blue square (lower-right), white background — i.e. the fp16 vision tower is intact under 8-bit LM quantization.

## How to reproduce
```bash
cd .
# quantize (8-bit, vision fp16, MTP preserved)
PYTHONPATH=$PWD .venv-omlx/bin/python -m mlx_mtp.oq_quantize \
  --src "<models>/osmQwopus3.6-27B-v2-heretic" \
  --out "<models>/osmQwopus3.6-27B-mlxmtp-oQ8" --level 8
# benchmark
PYTHONPATH=$PWD .venv-omlx/bin/python -m mlx_mtp.bench \
  --model "<models>/osmQwopus3.6-27B-mlxmtp-oQ8" \
  --image test_image.png --max-tokens 128 --out mlx-mtp-benchmark.json
```

## DFlash, and the DFlash + native-MTP hybrid (`mlx_mtp/dflash.py`, `mlx_mtp/hybrid.py`)
Added the external block-diffusion drafter `z-lab/Qwen3.6-27B-DFlash` (gated; HF-token download, not stored) and a hybrid that runs **both** drafters in one loop.

3-way + hybrid (8-bit quant, 128 tok, greedy):
| Path | tok/s | speedup |
|---|---|---|
| Vanilla AR | 15.7 | 1.00× |
| **Native MTP** (embedded head) | **25.5** | **1.62×** |
| DFlash (external, block 16) | 19.9 | 1.27× |
| **Hybrid (adaptive)** | **23.8** | **1.51×** (`{mtp:64, dflash:4}`) |

**Can they be used together? Yes — built & working.** `_dflash_rounds` and `_mtp_rounds` are the same draft→verify→accept→`rollback_speculative_cache` shape, so one unified loop holds both and picks per round by a **throughput (tok/s) bandit**. Findings:
- They're **not additive** (both shorten decode steps) — the hybrid's role is to **pick the faster drafter per context**.
- A subtle but important lesson: a tokens-per-**round** bandit wrongly prefers DFlash's big blocks and *slows down* (0.81×); the bandit must optimize tokens-per-**second**. Fixed → 1.51×.
- On *this* 8-bit model **native MTP wins**, so the hybrid converges to it (gap to 1.62× = DFlash exploration cost). DFlash carries a cross-round draft-cache (MTP is stateless), which is why fine interleaving needs `reset()`+min-dwell and why the libs gate them mutually exclusive.
- **Practical recommendation:** on this model, just use native MTP. The hybrid pays off when DFlash's long blocks land better (different model/context).

## Honest scope notes
- **D1** (single draft/round) is implemented and verified. **D2/D3** (MTPLX's higher blocks, ~2.24× on a tuned model) need the MTP head chained on its own hidden across K drafts — a clean extension, not done here.
- We **forked** omlx (its mlx-vlm MTP runtime patch loads/attaches the head; its oQ does the quant) and **wrote our own** the draft/verify/accept loop driving the embedded head — the part that didn't exist for VLMs.
- Greedy decode used for clean A/B; the loop also supports temp>0 (probability-ratio accept + residual).

## Lineage / credit
- MTPLX (native MTP, no external drafter) — youssofal/MTPLX · mtplx.com
- optiq / omlx (oQ mixed quant + MTP runtime + vision) — jundot/omlx, mlx-optiq
- mlx / mlx-lm / mlx-vlm — ml-explore, Blaizzy
