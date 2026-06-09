# mlx-mtp

**Native MTP speculative decoding + vision-preserving 8-bit quantization for Qwen3.5 / 3.6 vision-language models on Apple Silicon.**

`mlx-mtp` runs a **VLM** with its **embedded multi-token-prediction (MTP) head** as a self-drafter — no external drafter — and ships a quantizer that keeps the vision tower in fp16 while compressing the language model to 8-bit and **preserving the MTP head**. It also includes an experimental **DFlash + MTP hybrid**.

> ⚠️ **Research artifact, not a polished package.** It builds on and adapts [oMLX](https://github.com/jundot/omlx) (the embedded-MTP runtime patch + `oQ` quantizer) and the ideas of [MTPLX](https://github.com/youssofal/MTPLX), and calls some **private** `mlx-vlm` internals. It pins to a specific MLX stack and will need updates as those move. See [Status & caveats](#status--caveats).

## Results (Apple M4 Max · osmQwopus-3.6-27B · 8-bit · greedy · D1)

| Path | tok/s | speedup |
|---|---|---|
| Vanilla AR | 15.7 | 1.00× |
| **Native MTP** (embedded head) | **25.5** | **1.62×** |
| DFlash (external diffusion drafter, block 16) | 19.9 | 1.27× |
| **DFlash + MTP hybrid** (adaptive) | 23.8 | 1.51× |

- Quantizer: 52 GB bf16 → **28 GB** 8-bit; **vision tower kept fp16**; **MTP head preserved**.
- Native MTP is **lossless** (output matches vanilla AR; speculative decoding is exact w.r.t. the verify-pass distribution).
- Vision works on the 8-bit quant (correct image captioning). Full numbers: [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

## How it works

- **Quantizer** (`mlx_mtp/oq_quantize.py`) — thin driver over oMLX's streaming `oQ` quantizer: language model → 8-bit affine, vision/audio + Qwen3.5 hybrid-SSM params (`a_log`, `dt_bias`, `conv1d`) kept fp16 automatically, `preserve_mtp=True`.
- **Engine** (`mlx_mtp/engine.py`) — drives the model's own `language_model.mtp` head: draft a token via `mtp_forward(pre_norm_hidden, tok)`, verify the target on `[tok, draft]` in one forward (capturing logits + hidden + Mamba/GDN state), greedy-accept iff `draft == argmax(target)`, and `rollback_speculative_cache(...)` to restore **both** KV and SSM/conv state on rejection. *(Neither mlx-vlm nor oMLX wires the embedded head as the drafter for a VLM — both use an external gemma4 drafter; this engine is the missing piece.)*
- **DFlash + hybrid** (`mlx_mtp/dflash.py`, `mlx_mtp/hybrid.py`) — adds the external DFlash drafter and a unified loop that picks MTP or DFlash per round by a **tokens-per-second** bandit. (They are *not* additive — both shorten decode steps — so the hybrid's job is to pick the faster drafter per context.)

## Requirements

Apple Silicon, macOS, Python 3.11+. This depends on the **oMLX** runtime stack (which is **not on PyPI**):

```bash
# 1. clone + install oMLX (pulls its pinned mlx / mlx-lm / mlx-vlm commits)
git clone https://github.com/jundot/omlx && pip install -e ./omlx
# 2. then use mlx-mtp from this repo (PYTHONPATH or pip install -e .)
pip install -e .
```

(Standalone `pip install mlx-mtp` is not yet possible because oMLX isn't a PyPI dependency — see caveats.)

## Usage

```bash
# Quantize a Qwen3.5/3.6 VLM to 8-bit (vision fp16, MTP preserved)
python -m mlx_mtp.oq_quantize --src <bf16-model-dir> --out <out-dir> --level 8

# Vanilla vs native-MTP (+ acceptance, lossless check)
python -m mlx_mtp.engine --model <out-dir> --max-tokens 128

# Full benchmark + vision test → JSON
python -m mlx_mtp.bench --model <out-dir> --image test_image.png --out benchmark.json

# DFlash alone, and the DFlash+MTP hybrid (needs a DFlash drafter)
python -m mlx_mtp.dflash --model <out-dir> --drafter <dflash-drafter-dir>
python -m mlx_mtp.hybrid --model <out-dir> --drafter <dflash-drafter-dir>
```

## Status & caveats

- **Derivative work.** The MTP-head attach and the quantizer come from oMLX (Apache-2.0); the engine also calls private mlx-vlm functions (`_speculative_walk`, `rollback_speculative_cache`, `capture_layer_ids`). Pinned to that stack; expect breakage across versions. See [`NOTICE`](NOTICE).
- **D1 only.** Single-token drafting is implemented; D2/D3 multi-draft (chain the head on its own hidden) is a natural extension.
- **Hybrid is experimental** and, on the tested model, converges to native MTP (which wins). DFlash carries a cross-round draft-cache while MTP is stateless, which is why interleaving needs `reset()` + a min-dwell.
- Tested on one model (osmQwopus-3.6-27B); numbers are illustrative, not a tuned benchmark.

## Credits

- **[MTPLX](https://github.com/youssofal/MTPLX)** (Youssof Altoukhi) — native MTP speculative decoding, the inspiration.
- **[oMLX](https://github.com/jundot/omlx)** — the `oQ` quantizer + embedded-MTP runtime patch + vision support this builds on.
- **[mlx / mlx-lm / mlx-vlm](https://github.com/ml-explore)** (Apple ml-explore, Prince Canuma) — the engine.
- **[mlx-optiq](https://mlx-optiq.com)** — mixed-precision quantization (the "optiq" reference).
- **DFlash** — `z-lab/Qwen3.6-27B-DFlash` block-diffusion drafter.

## License

[Apache-2.0](LICENSE). See [`NOTICE`](NOTICE) for attribution of incorporated/adapted work.
