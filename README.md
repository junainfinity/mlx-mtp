# mlx-mtp

**A new MLX quantization and inference stack for Apple Silicon — purpose-built for vision-language models with speculative decoding.**

`mlx-mtp` is a modular quantization + inference library for running large VLMs fast and locally on Apple Silicon. It ships:

- **Multiple quantization formats** — 8-bit affine (oQ), MXFP4 (OCP MX, 4-bit E2M1), with vision tower and SSM-sensitive params preserved in fp16 across all formats
- **Native MTP speculative decoding** — drives a model's own embedded multi-token-prediction head as a self-drafter; no external model needed
- **DFlash block-diffusion** — external block-diffusion drafter path with tunable block size
- **MTP + DFlash hybrid** — adaptive per-round bandit that picks whichever drafter is faster in context
- **A full benchmark harness** — vanilla vs MTP vs DFlash vs hybrid, with TTFT, acceptance rate, vision, and output-match checks

> **Best recommended model: [osmQwopus-3.6-27B](https://huggingface.co/junainfinity/osmQwopus)** — a 27B Qwen3.6-based uncensored VLM. All benchmarks in this repo are on osmQwopus; it is the most tested and optimized target. The stack will work with any Qwen3.5/3.6-family VLM that carries a MTP head and runs in `mlx-vlm`.

> **mlx-mtp is the inference engine behind [VibeStudio](https://github.com/junainfinity/VibeStudio)**, a local coding-agent desktop app (Tauri + React). VibeStudio's Decoding tab — Standard / MTP ×N / DFlash / Speculative — selects and parameterizes this engine through `omlx serve`.

---

## Benchmarks (Apple M4 Max · osmQwopus-3.6-27B · greedy · D1)

### 8-bit affine (oQ) — 28 GB · vision fp16 · MTP head preserved

| Path | tok/s | speedup |
|---|---|---|
| Vanilla AR | 15.7 | 1.00× |
| **Native MTP** (embedded head, no drafter) | **25.5** | **1.62×** |
| DFlash (z-lab drafter, block 16) | 24.1 | 1.52× |
| DFlash + MTP hybrid (adaptive) | 23.8 | 1.51× |

### MXFP4 (4-bit E2M1, group 32) — 14 GB · vision fp16 · DFlash-ready

| Path | tok/s | speedup |
|---|---|---|
| Vanilla AR | 27.4 | 1.00× |
| **DFlash** (z-lab drafter, block 16) | **36.5** | **1.33×** |

Half the size of oQ8 — Apple Silicon is memory-bandwidth-bound, so the smaller model decodes faster in absolute tok/s. Use oQ8 when output quality is the priority; MXFP4 when throughput or RAM headroom matters.

Full numbers: [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

### MXFP4 quality (KL divergence vs BF16 reference)

Measured via teacher-forced prefill over 10 diverse prompts (code, math, reasoning, instruction) — 244 token positions total, full vocabulary distributions compared at every position.

| Metric | Value | Notes |
|---|---|---|
| **KL(bf16 ‖ mxfp4) mean** | **0.034 nats** | Forward divergence from reference |
| JSD mean | 0.009 nats | Symmetric; JSD ≤ 0.01 = "essentially identical" |
| **Top-1 agreement** | **92.4%** | Both models pick the same argmax token |
| PPL increase | +0.87% | Perplexity under the reference token sequence |
| Code prompts (KL) | 0.010–0.012 | Lowest divergence — code is most deterministic |

JSD of 0.009 = 1.3% of the [0, ln(2)] bound. For a 4-bit model that is 3.7× smaller than bf16, this is excellent distribution fidelity. The 92.4% top-1 match matches what we observe in generation: most greedy outputs are byte-identical; divergences are near-tie argmax flips, not quality regressions.

---

## Quantization formats

### `mlx_mtp.oq_quantize` — 8-bit affine (oQ)

Thin driver over oMLX's streaming `oQ` quantizer. Language model → 8-bit affine (group 64); vision tower + projector + Qwen3.5 hybrid-SSM params (`a_log`, `dt_bias`, `conv1d`) kept fp16; MTP head preserved if present.

```bash
python -m mlx_mtp.oq_quantize \
  --src <bf16-model-dir> --out <out-dir> --level 8
```

52 GB bf16 → **28 GB** 8-bit. Vision captions correctly. MTP head loads and drafts.

### `mlx_mtp.mxfp4_quantize` — MXFP4 (OCP MX 4-bit)

OCP Microscaling FP4: 4-bit E2M1 mantissa + shared E8M0 scale per group of 32. Language model linears → MXFP4; vision tower + projector (all multimodal modules) + Qwen3.5 SSM-sensitive params → fp16. MTP head preserved if the source checkpoint contains it.

```bash
python -m mlx_mtp.mxfp4_quantize \
  --src <bf16-model-dir> --out <out-dir>
```

52 GB bf16 → **14 GB** 4-bit. Vision captions correctly. DFlash works. If the source has no MTP head the quantizer disables `mtp_num_hidden_layers` in the output config so it loads cleanly.

### Why vision and SSM params stay fp16

Vision encoders and projectors operate on continuous pixel/patch features — low-bit quantization visibly degrades captioning. Qwen3.5's hybrid SSM block has recurrent parameters (`a_log`, `dt_bias`, `conv1d`) that are both tiny and numerically sensitive; keeping them fp16 is essentially free. Everything else — attention + MLP linears, `lm_head` — is safe to quantize.

---

## Inference modes

### Native MTP (`mlx_mtp.engine`)

Drives the model's own `language_model.mtp` head as a self-drafter — no external drafter model required:

1. Draft: `mtp_forward(pre_norm_hidden, tok)` → candidate token
2. Verify: target forward on `[tok, draft]` in one pass, capturing logits + hidden + GDN/SSM state
3. Greedy-accept iff `draft == argmax(target)`; on reject, `rollback_speculative_cache(...)` restores both KV **and** SSM/conv state

Neither `mlx-vlm` nor `oMLX` wires the embedded MTP head as a drafter for a VLM — they both use an external gemma4-class drafter. This engine is the missing piece.

```bash
python -m mlx_mtp.bench \
  --model <out-dir> --image test_image.png --out benchmark.json
```

### DFlash (`mlx_mtp.dflash`)

Block-diffusion external drafter (e.g. `z-lab/Qwen3.6-27B-DFlash`). Block size is a tunable parameter — smaller blocks draft fewer tokens per round (less wasted work on rejects):

```bash
python -m mlx_mtp.dflash \
  --model <out-dir> --drafter z-lab/Qwen3.6-27B-DFlash \
  --block-size 8   # 8 | 16 | 32
```

DFlash is quant-agnostic on the target — it works on both oQ8 and MXFP4 builds.

### Block-size sweep

```bash
python -m mlx_mtp.dflash --model <out-dir> --drafter <drafter> --block-sweep
```

| block_size | tok/s | speedup |
|---|---|---|
| 8 | 24.0 | 1.58× |
| 16 | 23.9 | 1.57× |
| 32 | 22.7 | 1.49× |

### MTP + DFlash hybrid (`mlx_mtp.hybrid`)

Unified loop that maintains both the MTP head and the DFlash drafter, and picks whichever produces more **tokens per second** (not tokens per round — that metric incorrectly over-selects DFlash) in a per-context bandit. On osmQwopus, native MTP tends to win; the hybrid is useful on models where DFlash's acceptance rate is higher.

---

## Requirements

Apple Silicon · macOS · Python 3.11+

```bash
# 1. Install oMLX (not on PyPI; pins its mlx / mlx-lm / mlx-vlm commits)
git clone https://github.com/jundot/omlx && pip install -e ./omlx

# 2. Install mlx-mtp
git clone https://github.com/junainfinity/mlx-mtp && pip install -e ./mlx-mtp
```

Tested on: MLX 0.31.2 · mlx-vlm 0.6.2 · Python 3.12 · macOS Sequoia · M4 Max.

---

## Roadmap

- [ ] D2 / D3 multi-draft (chain the MTP head on its own hidden state for higher acceptance)
- [ ] MXFP8 and nvFP4 quantization formats (same vision/SSM protection)
- [ ] MTP-aware beam search
- [ ] Standalone PyPI release (currently gated on oMLX not being on PyPI)

---

## Status & caveats

**Derivative work.** The MTP-head attach and the oQ quantizer come from oMLX (Apache-2.0); the engine calls private `mlx-vlm` functions (`_speculative_walk`, `rollback_speculative_cache`, `capture_layer_ids`). Pinned to that stack; expect breakage across upstream versions. See [`NOTICE`](NOTICE).

**Not strictly lossless, but output is on par with vanilla.** MTP output matches vanilla greedy AR on most prompts (3/4 byte-identical in benchmarks); the occasional divergence is a near-tie argmax flip from the quantized 2-token verify forward — not a quality regression. Speculative decoding is exact w.r.t. the verify-pass distribution.

**D1 only.** Single-token drafting is implemented; D2/D3 multi-draft is a planned extension.

**Tested primarily on osmQwopus-3.6-27B.** Numbers are from real end-to-end runs, not a curated benchmark suite.

---

## Credits

- **[MTPLX](https://github.com/youssofal/MTPLX)** (Youssof Altoukhi) — native MTP speculative decoding for LLMs; the inspiration.
- **[oMLX](https://github.com/jundot/omlx)** — the `oQ` quantizer, embedded-MTP runtime patch, and vision support this builds on.
- **[mlx / mlx-lm / mlx-vlm](https://github.com/ml-explore)** (Apple ml-explore, Prince Canuma) — the underlying engine.
- **[mlx-optiq](https://mlx-optiq.com)** — mixed-precision quantization reference.
- **DFlash** — `z-lab/Qwen3.6-27B-DFlash` block-diffusion drafter.

---

## License

[Apache-2.0](LICENSE). See [`NOTICE`](NOTICE) for attribution of incorporated and adapted work.
