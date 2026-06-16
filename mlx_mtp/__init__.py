"""mlx-mtp — native MTP speculative decoding + vision-preserving quantization for
Qwen3.5/3.6 VLMs on Apple Silicon, built on Apple's MLX ONLY.

Pure mlx.core / mlx.nn (+ a tokenizer I/O boundary). No third-party ML-inference
frameworks at runtime — the Qwen3.5 architecture (hybrid Gated-DeltaNet + full
attention), the vision tower, the embedded MTP head, and the DFlash drafter are
all implemented here.

  - mlx_mtp.quantize : tensor-level MXFP4 / MXFP8 quantizer that keeps the vision
                       tower, MTP head, and SSM-sensitive params in fp16.
  - mlx_mtp.loader   : pure-mlx checkpoint loader (rebuilds quantized layers from config).
  - mlx_mtp.engine   : vanilla AR + native embedded-MTP speculative decoding.
  - mlx_mtp.hybrid   : MTP + DFlash hybrid (where DFlash drafter weights are available).
"""
__version__ = "0.2.0"

from mlx_mtp.quantize import quantize  # noqa: E402,F401


def load_model(path):
    """Convenience re-export of mlx_mtp.loader.load (lazy import to keep mlx the only
    hard dependency of `import mlx_mtp`)."""
    from mlx_mtp.loader import load

    return load(path)
