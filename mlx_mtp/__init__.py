"""mlx-mtp — native MTP speculative decoding + vision-preserving quantization for
Qwen3.5/3.6 VLMs on Apple Silicon.

Two components:
  - mlx_mtp.quantize : 8-bit quantizer that keeps the vision tower fp16 AND
                       preserves the MTP head (which stock mlx_vlm convert drops).
  - mlx_mtp.engine   : loads the quant, attaches the MTP head, and runs native
                       MTP speculative decoding (draft K, verify K+1, p/q accept
                       + residual correction) — plus a vanilla AR baseline.

Built on mlx / mlx-lm / mlx-vlm. Blueprint distilled from youssofal/MTPLX and
jundot/omlx; this is an independent implementation.
"""
__version__ = "0.1.0"
