"""Dependency access for the gateway plane: everything lives on app.state so
tests can inject a pool/registry/token issuer without touching global config."""
from __future__ import annotations

from typing import Union

from fastapi import Request, WebSocket

from .pool import GatewayPool
from .session import GatewayRegistry
from .tokens import TokenIssuer

Conn = Union[Request, WebSocket]


def get_gateway_pool(conn: Conn) -> GatewayPool:
    return conn.app.state.gateway_pool


def get_gateway_registry(conn: Conn) -> GatewayRegistry:
    return conn.app.state.gateway_registry


def get_gateway_tokens(conn: Conn) -> TokenIssuer:
    return conn.app.state.gateway_tokens
