"""Finish owned cleanup before propagating cancellation to its caller."""

import asyncio

import anyio


async def await_cleanup(task: asyncio.Task):
    cancelled = None
    # AnyIO cancellation is level-triggered; asyncio.shield alone would make
    # a cancelled scope interrupt every subsequent attempt to await the task.
    with anyio.CancelScope(shield=True):
        while True:
            try:
                result = await asyncio.shield(task)
                break
            except asyncio.CancelledError as exc:
                if task.cancelled():
                    raise
                cancelled = exc
    if cancelled is not None:
        raise cancelled
    return result
