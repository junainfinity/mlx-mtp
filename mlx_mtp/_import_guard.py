"""Runtime purity guard for mlx-mtp.

Goal 1 of the project: mlx-mtp must run on Apple's mlx (mlx.core + mlx.nn) ONLY —
no mlx_vlm, no mlx_lm, no omlx at runtime. The single sanctioned non-mlx boundary
is the tokenizer/image-processor (transformers / tokenizers / PIL), used purely for
text<->id and pixel I/O in mlx_mtp.tokenizer.

`assert_no_forbidden_runtime()` is the load-bearing acceptance check: after importing
the package, no forbidden framework should be present in sys.modules.
"""
from __future__ import annotations

import sys

FORBIDDEN = ("mlx_vlm", "mlx_lm", "omlx")
# transformers/tokenizers/PIL are the allowed tokenizer/I-O boundary; numpy is stdlib-grade.
ALLOWED_BOUNDARY = ("transformers", "tokenizers", "PIL", "numpy")


def forbidden_loaded() -> list[str]:
    """Return any forbidden top-level modules currently imported."""
    return sorted({
        m.split(".")[0]
        for m in sys.modules
        if m.split(".")[0] in FORBIDDEN
    })


def assert_no_forbidden_runtime() -> None:
    """Raise if any forbidden ML framework leaked into sys.modules."""
    bad = forbidden_loaded()
    if bad:
        raise AssertionError(
            f"mlx-mtp purity violated: forbidden runtime modules imported: {bad}. "
            f"mlx-mtp must use only mlx.core/mlx.nn (+ tokenizer boundary {ALLOWED_BOUNDARY})."
        )
