"""Entry point: ``python -m adapter`` from the vl_legacy_adapter directory."""

from __future__ import annotations

import asyncio
import logging

from .server import AdapterServer


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(AdapterServer().run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
