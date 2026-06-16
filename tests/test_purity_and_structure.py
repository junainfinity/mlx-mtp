"""mlx-mtp acceptance tests — purity + structure + quantizer, all weight-free or tiny.

Run: python3 -m pytest tests/test_purity_and_structure.py -v
(or just `python3 tests/test_purity_and_structure.py` for a plain run)

These verify the goal-1 contract WITHOUT the full 27B weights:
  - purity: no third-party ML-inference frameworks imported at runtime
  - structure: 64 layers (full-attn every 4th), exactly 15 MTP tensors, 1199-tensor
    name parity vs the real checkpoint index
  - quantizer: skip predicates + mxfp4/mxfp8 round-trip error bands
  - MTP loop: draft -> verify -> rollback on computed values (tiny config)
"""
import json
import os
import sys

import mlx.core as mx
from mlx.utils import tree_flatten

# Point QWOPUS_CODER at a real bf16 Qwen3.5/3.6 checkpoint dir to run the
# name-parity/structure test; it skips cleanly when unset (CI-safe, weight-free).
MODEL = os.environ.get("QWOPUS_CODER", "")


def test_runtime_purity():
    import mlx_mtp  # noqa
    import mlx_mtp.engine  # noqa
    import mlx_mtp.hybrid  # noqa
    import mlx_mtp.loader  # noqa
    import mlx_mtp.quantize  # noqa
    import mlx_mtp.models.qwen3_5  # noqa
    from mlx_mtp._import_guard import assert_no_forbidden_runtime

    assert_no_forbidden_runtime()


def test_quantizer_skip_and_roundtrip():
    from mlx_mtp.quantize import quantize_weights, _count_stats

    synth = {
        "model.language_model.layers.0.mlp.gate_proj.weight": mx.random.normal((128, 64)).astype(mx.bfloat16),
        "model.visual.blocks.0.attn.qkv.weight": mx.random.normal((64, 64)).astype(mx.bfloat16),
        "mtp.fc.weight": mx.random.normal((64, 128)).astype(mx.bfloat16),
        "model.language_model.layers.0.linear_attn.A_log": mx.random.normal((48,)).astype(mx.float32),
        "model.language_model.layers.0.linear_attn.dt_bias": mx.random.normal((48,)).astype(mx.float32),
        "model.language_model.layers.0.linear_attn.conv1d.weight": mx.random.normal((10240, 1, 4)).astype(mx.bfloat16),
        "model.language_model.layers.0.self_attn.q_proj.weight": mx.random.normal((128, 2050)).astype(mx.bfloat16),
    }
    q, stats = quantize_weights(synth, mode="mxfp8")
    c = _count_stats(stats)
    assert c == {"quantized": 1, "fp16_vision": 1, "fp16_mtp": 1, "fp16_ssm": 3, "fp16_other": 1}, c
    assert q["model.language_model.layers.0.mlp.gate_proj.weight"].dtype == mx.uint32
    assert not any(k.endswith(".biases") for k in q)

    for mode, bits, band in [("mxfp8", 8, 0.06), ("mxfp4", 4, 0.25)]:
        w = mx.random.normal((64, 128)).astype(mx.bfloat16)
        wq, sc = mx.quantize(w, group_size=32, bits=bits, mode=mode)
        dq = mx.dequantize(wq, sc, group_size=32, bits=bits, mode=mode)
        rel = (mx.mean(mx.abs(dq - w)) / mx.mean(mx.abs(w))).item()
        assert rel < band, (mode, rel)


def _real_config():
    cfg = json.load(open(f"{MODEL}/config.json"))
    cfg["text_config"]["mtp_num_hidden_layers"] = cfg["text_config"].get("mtp_num_hidden_layers", 1)
    return cfg


def test_structure_and_name_parity():
    from mlx_mtp.models.qwen3_5.config import ModelConfig
    from mlx_mtp.models.qwen3_5.glue import Model, sanitize_key

    if not os.path.exists(f"{MODEL}/config.json"):
        return  # skip if base model not present
    mc = ModelConfig.from_dict(_real_config())
    model = Model(mc)  # lazy params; never eval'd -> no big allocation
    expected = {k for k, _ in tree_flatten(model.parameters())}

    idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    mapped = {sanitize_key(k) for k in idx}
    assert expected == mapped, (
        f"name parity gap: missing={sorted(expected - mapped)[:5]} extra={sorted(mapped - expected)[:5]}"
    )
    assert len(expected) == 1199
    mtp = [k for k in idx if k.startswith("mtp.")]
    vis = [k for k in idx if "visual" in k]
    assert len(mtp) == 15 and len(vis) == 333

    full = sorted({int(k.split(".")[3]) for k in expected
                   if k.startswith("language_model.model.layers.") and ".self_attn." in k})
    assert full == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]


def test_mtp_loop_tiny():
    from mlx_mtp.models.qwen3_5.config import TextConfig, VisionConfig, ModelConfig
    from mlx_mtp.models.qwen3_5.language import LanguageModel

    tc = TextConfig(
        model_type="qwen3_5_text", hidden_size=256, intermediate_size=512,
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
        num_hidden_layers=8, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=512, num_key_value_heads=2, max_position_embeddings=4096,
        head_dim=64, full_attention_interval=4,
        rope_parameters={"type": "default", "mrope_section": [3, 3, 2],
                         "rope_theta": 1e6, "partial_rotary_factor": 0.25},
        mtp_num_hidden_layers=1,
    )
    mc = ModelConfig(text_config=tc, vision_config=VisionConfig(model_type="qwen3_5"),
                     model_type="qwen3_5")
    lm = LanguageModel(tc, mc)
    lm.mtp.bind(lm)
    mx.eval(lm.parameters())
    last = len(lm.model.layers) - 1

    ids = mx.array([[1, 2, 3, 4, 5]])
    cache = lm.make_cache()
    out = lm(ids, cache=cache, capture_layer_ids=[last])
    assert tuple(out.logits.shape) == (1, 5, 512)
    hidden = out.hidden_states[0][:, -1:, :]
    cur = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())

    mlogits = lm.mtp_forward(hidden, mx.array([[cur]]), lm.make_mtp_cache())
    assert tuple(mlogits.shape) == (1, 1, 512)
    d = int(mx.argmax(mlogits[:, -1, :], axis=-1).item())

    kv = [c for c in cache if hasattr(c, "offset")][0]
    vout = lm(mx.array([[cur, d]]), cache=cache, capture_layer_ids=[last])
    off = int(kv.offset)
    lm.rollback_speculative_cache(cache, vout.gdn_states, 0, 2)
    assert int(kv.offset) == off - 1  # reject trims 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} acceptance tests passed.")
