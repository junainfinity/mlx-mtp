"""mlx-mtp pure-mlx quantizer — zero mlx_vlm/omlx dependency, safetensors I/O at tensor level.

COMPLETE pure-MLX quantization recipe for mlx-mtp (uses only mlx.core + stdlib):
  - Quantizes the language model to mxfp4 (4-bit) or mxfp8 (8-bit), OCP microscaling
    (E2M1/E4M3 elements + shared E8M0 scale per group of 32).
  - Preserves the vision tower, the MTP head, and SSM-sensitive params in fp16.
  - Saves sharded safetensors + index + config.json for mlx-native reload by mlx_mtp.loader.
  - NO model-class instantiation; operates on tensor dicts only.

Usage:
    python -m mlx_mtp.quantize --src <bf16_model_dir> --out <out_dir> --mode mxfp4
    python -m mlx_mtp.quantize --src <bf16_model_dir> --out <out_dir> --mode mxfp8 --verify
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx

# ============================================================================
# SKIP PREDICATES: which modules/tensors to keep in fp16
# ============================================================================

MULTIMODAL_MODULES = (
    "visual",
    "vision_model",
    "vision_tower",
    "vl_connector",
    "sam_model",
    "audio_model",
    "audio_tower",
    "code_predictor",
    "img_projector",
    "multi_modal_projector",
    "merger",
)

SSM_SENSITIVE = (
    "a_log",
    "A_log",
    "dt_bias",
    "conv1d",
    "ssm_alpha",
    "ssm_beta",
    "time_decay",
    "time_faaaa",
)

COPY_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


def skip_multimodal_module(path: str) -> bool:
    """True if a tensor path belongs to a multimodal (vision/audio) module."""
    return any(module in path for module in MULTIMODAL_MODULES)


def _skip_reason(weight_name: str) -> Optional[str]:
    """Return a reason to keep this weight fp16, or None to quantize it."""
    if skip_multimodal_module(weight_name):
        return "vision"
    if "mtp" in weight_name.lower():
        return "mtp"
    if any(s in weight_name for s in SSM_SENSITIVE):
        return "ssm"
    return None


def _can_quantize(weight_name: str, shape: Tuple[int, ...]) -> bool:
    """Quantizable iff 2D and last dim divisible by the mx group size (32)."""
    if not weight_name.endswith(".weight"):
        return False
    if len(shape) != 2:
        return False
    if shape[-1] % 32 != 0:
        return False
    return True


# ============================================================================
# TENSOR-LEVEL I/O
# ============================================================================

def _save_safetensors(path: str, weights: Dict[str, mx.array]) -> None:
    mx.save_safetensors(path, weights, metadata={"format": "mlx"})


def _make_shards(
    weights: Dict[str, mx.array], max_size_gb: float = 5.0
) -> list[Dict[str, mx.array]]:
    max_bytes = int(max_size_gb * (1 << 30))
    shards: list[Dict[str, mx.array]] = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        v_bytes = v.nbytes
        if shard and shard_size + v_bytes > max_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v_bytes
    if shard:
        shards.append(shard)
    return shards


# ============================================================================
# QUANTIZATION LOGIC
# ============================================================================

def quantize_weights(
    weights: Dict[str, mx.array],
    mode: str = "mxfp4",
    group_size: int = 32,
    bits: Optional[int] = None,
) -> Tuple[Dict[str, mx.array], Dict[str, str]]:
    """Quantize a weights dict; return (quantized_weights, per-tensor stats).

    For each quantizable 2D `.weight` (last dim % 32 == 0) emits `{name}` (uint32
    packed) + `{name w/o .weight}.scales` (uint8 E8M0); mxfp4/mxfp8 carry NO biases.
    Vision, MTP, and SSM-sensitive tensors are passed through fp16.
    """
    if bits is None:
        bits = {"mxfp4": 4, "mxfp8": 8}.get(mode, 4)

    quantized: Dict[str, mx.array] = {}
    stats: Dict[str, str] = {}

    for name, w in weights.items():
        reason = _skip_reason(name)
        if reason:
            quantized[name] = w
            stats[name] = reason
            continue
        if not _can_quantize(name, w.shape):
            quantized[name] = w
            stats[name] = "non-2d-or-unaligned"
            continue
        try:
            w_q, scales = mx.quantize(w, group_size=group_size, bits=bits, mode=mode)
            quantized[name] = w_q
            quantized[f"{name[:-len('.weight')]}.scales"] = scales
            stats[name] = "quantized"
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: failed to quantize {name}: {e}")
            quantized[name] = w
            stats[name] = f"quantize_error:{type(e).__name__}"

    return quantized, stats


def _count_stats(stats: Dict[str, str]) -> Dict[str, int]:
    counts = {"quantized": 0, "fp16_vision": 0, "fp16_mtp": 0, "fp16_ssm": 0, "fp16_other": 0}
    for reason in stats.values():
        if reason == "quantized":
            counts["quantized"] += 1
        elif reason == "vision":
            counts["fp16_vision"] += 1
        elif reason == "mtp":
            counts["fp16_mtp"] += 1
        elif reason == "ssm":
            counts["fp16_ssm"] += 1
        else:
            counts["fp16_other"] += 1
    return counts


# ============================================================================
# CONFIG HANDLING
# ============================================================================

def update_config(config: Dict[str, Any], mode: str, bits: int, group_size: int,
                  mtp_count: int) -> Dict[str, Any]:
    quant_info = {"group_size": group_size, "bits": bits, "mode": mode}
    config["quantization"] = quant_info
    config["quantization_config"] = quant_info
    config.setdefault("vision_config", {})["skip_vision"] = True
    config["mlx_mtp"] = {
        "version": 1,
        "backend": mode,
        "vision_fp16": True,
        "mtp_preserved": mtp_count > 0,
    }
    if mtp_count == 0:
        # source has no MTP head -> don't let the arch declare phantom MTP params
        config["mtp_num_hidden_layers"] = 0
        config.setdefault("text_config", {})["mtp_num_hidden_layers"] = 0
        config["mlx_mtp"]["note"] = "source had no MTP head"
    return config


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def quantize(src: str, out: str, mode: str = "mxfp4") -> Dict[str, int]:
    """Quantize a model dir (safetensors + config.json) to mxfp4/mxfp8."""
    t0 = time.time()
    mode = mode.lower()
    bits = {"mxfp4": 4, "mxfp8": 8}[mode]
    group_size = 32

    print(f">> mlx-mtp quantize: LM->{mode} (g{group_size}), vision+MTP+SSM->fp16")
    print(f"   source: {src}")
    print(f"   output: {out}")

    src_p = Path(src)
    out_p = Path(out)
    out_p.mkdir(parents=True, exist_ok=True)

    weight_files = sorted(glob.glob(str(src_p / "*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {src_p}")

    print(">> loading safetensors...", flush=True)
    weights: Dict[str, mx.array] = {}
    for wf in weight_files:
        print(f"   reading {Path(wf).name}...", flush=True)
        weights.update(mx.load(wf))
    print(f"   loaded {len(weights)} tensors")

    print(f">> quantizing {len(weights)} tensors...", flush=True)
    q_weights, stats = quantize_weights(weights, mode=mode, group_size=group_size, bits=bits)
    weight_stats = _count_stats(stats)
    print(f">> stats: {json.dumps(weight_stats)}")
    mtp_count = weight_stats["fp16_mtp"]

    print(f">> saving {len(q_weights)} tensors...", flush=True)
    shards = _make_shards(q_weights)
    shard_count = len(shards)
    shard_fmt = ("model-{:05d}-of-{:05d}.safetensors" if shard_count > 1 else "model.safetensors")
    index = {"metadata": {"total_size": sum(w.nbytes for w in q_weights.values())},
             "weight_map": {}}
    for i, shard in enumerate(shards):
        shard_name = shard_fmt.format(i + 1, shard_count)
        gb = sum(v.nbytes for v in shard.values()) / 1e9
        print(f"   writing {shard_name} ({gb:.2f}GB)...", flush=True)
        _save_safetensors(str(out_p / shard_name), shard)
        for k in shard:
            index["weight_map"][k] = shard_name
    index["weight_map"] = {k: index["weight_map"][k] for k in sorted(index["weight_map"])}
    (out_p / "model.safetensors.index.json").write_text(json.dumps(index, indent=4))

    print(">> updating config...", flush=True)
    cfg = json.loads((src_p / "config.json").read_text())
    cfg = update_config(cfg, mode=mode, bits=bits, group_size=group_size, mtp_count=mtp_count)
    (out_p / "config.json").write_text(json.dumps(dict(sorted(cfg.items())), indent=4))

    print(">> copying auxiliary files...", flush=True)
    for fname in COPY_FILES:
        sf = src_p / fname
        if sf.exists():
            shutil.copy2(sf, out_p / fname)
            print(f"   copied {fname}")

    print(f">> done in {time.time() - t0:.1f}s", flush=True)
    return weight_stats


def verify(out: str) -> None:
    out_p = Path(out)
    print(f"\nVerification: {out}")
    cfg = json.loads((out_p / "config.json").read_text())
    print(f"  config.quantization: {cfg.get('quantization', 'MISSING')}")
    print(f"  config.mlx_mtp: {cfg.get('mlx_mtp', 'MISSING')}")
    idx = json.loads((out_p / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    print(f"  total_size: {idx['metadata']['total_size'] / 1e9:.2f}GB | weight_map: {len(wm)}")
    scale_keys = [k for k in wm if k.endswith(".scales")]
    bias_keys = [k for k in wm if k.endswith(".biases")]
    vision_scaled = [k for k in scale_keys if skip_multimodal_module(k)]
    mtp_scaled = [k for k in scale_keys if "mtp" in k.lower()]
    print(f"  quantized (.scales): {len(scale_keys)}")
    print(f"  vision quantized (want 0): {len(vision_scaled)}")
    print(f"  MTP quantized (want 0): {len(mtp_scaled)}")
    print(f"  .biases (want 0 for mx): {len(bias_keys)} {'OK' if not bias_keys else 'WARN'}")


def main():
    ap = argparse.ArgumentParser(description="Pure-MLX quantizer for mlx-mtp (mxfp4/mxfp8)")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["mxfp4", "mxfp8"], default="mxfp4")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    stats = quantize(a.src, a.out, mode=a.mode)
    if a.verify:
        verify(a.out)
    print(f"\nDone. Stats: {json.dumps(stats)}")


if __name__ == "__main__":
    main()
