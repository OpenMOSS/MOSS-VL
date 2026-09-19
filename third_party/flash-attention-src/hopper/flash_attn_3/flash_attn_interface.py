"""Thin submodule that re-exports symbols from the top-level ``flash_attn_interface``.

Exists so callers using the official FA3 package layout
(``from flash_attn_3.flash_attn_interface import ...``) resolve to the same
symbols as ``from flash_attn_interface import ...``. Transformer-Engine's
attention backend imports both public and private (underscore-prefixed)
names from this module, so we re-export both.
"""

import flash_attn_interface as _fai

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_with_kvcache",
    "flash_attn_combine",
    "get_scheduler_metadata",
    "_flash_attn_forward",
    "_flash_attn_backward",
]

# Public API.
from flash_attn_interface import (  # noqa: F401
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_qkvpacked_func,
    flash_attn_with_kvcache,
    flash_attn_combine,
    get_scheduler_metadata,
)

# Private helpers used by third-party integrations (e.g. Transformer Engine).
_flash_attn_forward = _fai._flash_attn_forward
_flash_attn_backward = _fai._flash_attn_backward


# Mirror anything else that might be referenced via attribute access.
def __getattr__(name):  # pragma: no cover - simple passthrough
    return getattr(_fai, name)
