"""Tokenizer / image-processor boundary — the ONLY sanctioned non-mlx runtime dependency.

mlx has no tokenizer and no image preprocessor; Qwopus3.6 ships a 20MB BPE tokenizer.json
(vocab 248320 + vision specials) and a Qwen3-VL image processor. We use HuggingFace
transformers/AutoProcessor strictly for text<->id and pixel I/O — never for model compute.
Everything downstream (model, quant, speculative decoding) is pure mlx.core/mlx.nn.
"""
from __future__ import annotations

from typing import List, Optional

import mlx.core as mx


def load_processor(path: str):
    """AutoProcessor (fast tokenizer + Qwen3-VL image processor)."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(path)


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def _tok(processor):
    return getattr(processor, "tokenizer", processor)


def encode(processor, text: str) -> List[int]:
    return _tok(processor).encode(text)


def decode(processor, ids) -> str:
    return _tok(processor).decode(ids)


def eos_ids(processor, config) -> set:
    """EOS id set mirroring the original pipeline: eos_token_id (248044) + <|im_end|>."""
    ids = set()
    tok = _tok(processor)
    cfg_eos = config.get("eos_token_id") if isinstance(config, dict) else getattr(config, "eos_token_id", None)
    for v in (getattr(tok, "eos_token_id", None), cfg_eos):
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, (list, tuple)):
            ids.update(int(x) for x in v)
    try:
        ie = tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(ie, int) and ie >= 0:
            ids.add(ie)
    except Exception:
        pass
    return ids


def apply_chat_template(processor, config, text: str, num_images: int = 0) -> str:
    """Render the model's chat template (chat_template.jinja in the model dir)."""
    tok = _tok(processor)
    content = text
    if num_images > 0:
        content = [{"type": "image"} for _ in range(num_images)] + [{"type": "text", "text": text}]
    messages = [{"role": "user", "content": content}]
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def prompt_ids(processor, config, text: str, num_images: int = 0):
    prompt = apply_chat_template(processor, config, text, num_images=num_images)
    return mx.array([_tok(processor).encode(prompt)]), prompt


def preprocess_images(processor, images, text: str = ""):
    """Return (pixel_values: mx.array, image_grid_thw: mx.array) at the I/O boundary."""
    proc_out = processor(text=text or " ", images=images, return_tensors="np")
    pv = mx.array(proc_out["pixel_values"])
    grid = mx.array(proc_out["image_grid_thw"]) if "image_grid_thw" in proc_out else None
    return pv, grid
