"""Small compatibility helpers around SGLang ``ServerArgs``."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_global_server_args():
    """Return SGLang's process-global server args through a lazy import."""
    from sglang.srt.server_args import get_global_server_args as _get_global_server_args

    return _get_global_server_args()


def override_server_args(server_args: Any, source: str, **fields: Any) -> None:
    """Apply an audited ServerArgs mutation across SGLang config APIs.

    SGLang 0.5.16 exposes ``ServerArgs.override``. Newer SGLang releases split
    mutation by lifecycle: unpublished configuration uses
    ``declare_late_resolution`` and published configuration uses the runtime
    context. Keep that version boundary in one vendor shim so Omni call sites
    retain source-labelled mutation provenance.
    """
    legacy_override = getattr(server_args, "override", None)
    if callable(legacy_override):
        legacy_override(source, **fields)
        return

    from sglang.srt.runtime_context import get_context

    context = get_context()
    try:
        published_server_args = context.server_args
    except (AttributeError, ValueError):
        published_server_args = None

    if published_server_args is server_args and callable(
        getattr(context, "override", None)
    ):
        context.override(source, **fields)
        return

    try:
        from sglang.srt.arg_groups.overrides import declare_late_resolution

        declare_late_resolution(server_args, source, **fields)
        return
    except ImportError:
        pass

    # 0.5.14 has no override machinery; apply the audited fields directly.
    logger.debug(
        "override_server_args(%s): applying fields directly (pre-0.5.15 sglang)",
        source,
    )
    for name, value in fields.items():
        setattr(server_args, name, value)


__all__ = ["get_global_server_args", "override_server_args"]
