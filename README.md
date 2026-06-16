# mlx-mtp

**A pure-MLX quantization + inference stack for Apple Silicon — MXFP4/MXFP8 with the vision tower and MTP head preserved, plus native multi-token-prediction speculative decoding.**

`mlx-mtp` runs large Qwen3.5/3.6-family vision-language models fast and locally on Apple Silicon using **only Apple's MLX** (`mlx.core` / `mlx.nn`) for all compute — no third-party ML-inference frameworks. HuggingFace `transformers` is used *only* at the tokenizer + image-preprocessing I/O boundary (MLX ships no tokenizer).

It ships:

- **MXFP4 + MXFP8 quantization** (`mlx_mtp.quantize`) — OCP Microscaling (E2M1/E4M3 elements + a shared E8M0 scale per group of 32) via Apple's native `mx.quantize`. The **vision tower, the MTP head, and Qwen3.5's hybrid-SSM params stay fp16** automatically.
- **Native MTP speculative decoding** (`mlx_mtp.engine`) — drives the model's own embedded multi-token-prediction head as a self-drafter; no external draft model.
- **DFlash block-diffusion + MTP×DFlash hybrid** — an external block-diffusion drafter path and an adaptive per-context bandit (where a DFlash drafter is available).
- **A benchmark harness** — vanilla vs MTP vs DFlash vs hybrid (TTFT, acceptance rate, vision, output-match).

> mlx-mtp is the inference engine behind **[VibeStudio](https://github.com/junainfinity/VibeStudio)**, a local coding-agent desktop app (Tauri + React). VibeStudio's Decoding tab — Standard / MTP ×N / DFlash / Speculative — selects and parameterizes this engine.

---

## Purity (the design goal)

Every bit of model compute is Apple MLX. The quantizer imports **only `mlx.core` + the Python stdlib** — not even `transformers`. The runtime guard `mlx_mtp._import_guard.assert_no_forbidden_runtime()` raises if any third-party ML-inference framework ever lands in `sys.modules`.

The Qwen3.5 VLM model code under `mlx_mtp/models/` is implemented directly on `mlx.core` / `mlx.nn`; the MTP head is built natively and the GDN/SSM speculative-rollback is implemented in-repo. Verified statically, by a runtime `sys.modules` scan after a real quantize (only `mlx` is pulled in), and by an adversarial lazy/transitive-import sweep.

---

## Quantize

```bash
python -m mlx_mtp.quantize --src <bf16-model-dir> --out <out-dir> --mode mxfp4 --verify
python -m mlx_mtp.quantize --src <bf16-model-dir> --out <out-dir> --mode mxfp8 --verify
# console script equivalent:
mlx-mtp-quantize --src <bf16-dir> --out <out-dir> --mode mxfp8 --verify
```

Operates at the safetensors **tensor level** — no model instantiation. For each 2D `.weight` whose last dim is a multiple of 32 it emits Apple-MLX packed `uint32` weights + `uint8` E8M0 `.scales` (mxfp carries no biases). Everything matched by the skip predicates stays fp16:

- **Vision / multimodal** — `model.visual.*`, projectors, mergers (low-bit visibly degrades captioning).
- **MTP head** — `mtp.*` (kept so the self-drafter still works; the loader rebuilds it).
- **SSM-sensitive** — Qwen3.5 hybrid recurrence (`a_log`/`A_log`, `dt_bias`, `conv1d`, `ssm_*`): tiny and numerically sensitive, so fp16 is essentially free.

If the source has no MTP head, the output config sets `mtp_num_hidden_layers = 0` so it still loads cleanly. For a 27B Qwen3.6 VLM: ≈55 GB bf16 → **MXFP4 ≈16 GB** / **MXFP8 ≈30 GB**.

---

## Inference

### Native MTP (`mlx_mtp.engine`)

Drives `language_model.mtp` as a self-drafter — no external drafter required:

1. **Draft:** `mtp_forward(pre_norm_hidden, tok)` → candidate token
2. **Verify:** target forward on `[tok, draft]` in one pass (logits + hidden + GDN/SSM state)
3. **Accept** iff `draft == argmax(target)`; on reject, `rollback_speculative_cache(...)` restores both the KV cache **and** the SSM/conv recurrent state

### DFlash + hybrid (`mlx_mtp.dflash`, `mlx_mtp.hybrid`)

Block-diffusion external drafter (e.g. `z-lab/Qwen3.6-27B-DFlash`) with a tunable block size, plus a unified loop that picks MTP vs DFlash per context by tokens-per-second. Requires DFlash drafter weights — optional, "where available."

```bash
python -m mlx_mtp.bench --model <out-dir> --image test.png --out benchmark.json
```

---

## Serve (OpenAI-compatible)

Run a model behind a streaming OpenAI Chat Completions API — **pure stdlib HTTP + MLX**, no web framework:

```bash
mlx-mtp-serve --model-dir <out-dir> --port 8400 --host 127.0.0.1 [--api-key KEY]
# equivalently: python -m mlx_mtp.serve serve --model-dir <out-dir> --port 8400
```

- `GET /v1/models` — health/readiness probe
- `POST /v1/chat/completions` — streaming (SSE) or blocking; OpenAI-compatible
- Decoding per request via `"decoder": "mtp" | "vanilla"` (default `mtp`); the vanilla path honors `temperature` / `top_p` / `top_k`
- Multi-turn + system + **tool calling** (OpenAI `tools` → the model's tool syntax → `delta.tool_calls`)
- **Vision** via OpenAI `image_url` data-URI content blocks
- `<think>…</think>` reasoning is streamed as `delta.reasoning_content`
- Bearer auth when `--api-key` is set

Any OpenAI client (incl. agent frameworks) can drive it at `http://127.0.0.1:8400/v1`.

---

## Validation (osmQwopus-3.6-27B-Coder, this release)

Both quants pass the full real-weights gate on Apple Silicon:

| | MXFP4 | MXFP8 |
|---|---|---|
| Strict load (0 missing/unexpected keys), MTP head bound | ✓ | ✓ |
| Vanilla + native MTP decode (MTP output == vanilla greedy) | ✓ · accept ~0.77 | ✓ · accept ~0.77 |
| Vision caption | ✓ | ✓ |
| Purity guard (`assert_no_forbidden_runtime`) | ✓ | ✓ |
| Dequant fidelity vs bf16 (mean cosine) | 0.992 | 0.998 |

MXFP8 packed bytes are bit-identical to MLX's own `mx.quantize` round-trip, and quantization is deterministic (regenerated builds are byte-identical). Acceptance tests: `python -m pytest tests/ -v` (4/4; the name-parity/structure test runs when `QWOPUS_CODER` points at a bf16 checkpoint, otherwise skips — CI-safe).

A full throughput benchmark (tok/s, TTFT) for the Coder MXFP4/MXFP8 builds is forthcoming.

---

## Requirements & install

Apple Silicon · macOS · Python 3.11+

```bash
git clone https://github.com/junainfinity/mlx-mtp && pip install -e ./mlx-mtp
```

Runtime deps: `mlx`, `transformers`, `numpy`, `pillow`. Tested on MLX 0.31.2 · transformers 5.10.x · Python 3.14 · macOS · Apple M-series.

---

## Credits

- **[MLX](https://github.com/ml-explore/mlx)** (Apple) — `mlx.core` / `mlx.nn`, the compute engine this is built on.
- **DFlash** — `z-lab/Qwen3.6-27B-DFlash` block-diffusion drafter (optional, for the DFlash/hybrid paths).

---

## License

[Apache-2.0](LICENSE).
