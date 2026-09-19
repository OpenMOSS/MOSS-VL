"""FlashAttention-3 package compatibility exports."""

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_with_kvcache",
    "flash_attn_combine",
    "get_scheduler_metadata",
]


def __getattr__(name):
    if name in __all__:
        from . import flash_attn_interface

        return getattr(flash_attn_interface, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
