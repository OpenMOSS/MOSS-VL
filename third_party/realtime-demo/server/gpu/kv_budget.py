"""Per-session KV-cache budgeting.

Policy (confirmed): video quality is fixed, spare VRAM buys session LENGTH.
Every session gets a hard floor sized for `KV_SESSION_MIN_MINUTES` of streaming
video; when memory is abundant the budget grows to `KV_MEMORY_RATIO` of the
post-weights free VRAM. On boxes where even the floor doesn't fit (4090 after
~18 GB of weights) we proceed with `floor_met=False` and a loud warning — the
small box must keep working exactly as before, just without the guarantee.

Analytic bytes/token comes from the checkpoint config; cross-check it
empirically for the served checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..logging_conf import get_logger

log = get_logger(__name__)

MIB = 2**20

# below this VRAM, KV exhaustion means a real OOM → default to hard enforcement
KV_ENFORCE_HARD_BELOW_MIB = 32 * 1024


@dataclass(frozen=True)
class KvBudget:
    bytes_per_token: int
    budget_tokens: int
    floor_tokens: int
    cap_tokens: int
    tokens_per_second: float
    floor_met: bool
    enforce_hard: bool

    @property
    def est_max_seconds(self) -> float:
        if self.tokens_per_second <= 0:
            return 0.0
        return self.budget_tokens / self.tokens_per_second

    def as_dict(self) -> dict:
        return {
            "bytes_per_token": self.bytes_per_token,
            "budget_tokens": self.budget_tokens,
            "floor_tokens": self.floor_tokens,
            "cap_tokens": self.cap_tokens,
            "tokens_per_second": self.tokens_per_second,
            "floor_met": self.floor_met,
            "enforce_hard": self.enforce_hard,
            "est_max_seconds": round(self.est_max_seconds, 1),
        }


def _cfg(model_config: dict, key: str, default: Any = None) -> Any:
    """Read a text-model key, tolerating both flat and nested (text_config) layouts."""
    if key in model_config:
        return model_config[key]
    text = model_config.get("text_config")
    if isinstance(text, dict) and key in text:
        return text[key]
    return default


def kv_bytes_per_token(model_config: dict) -> int:
    """Analytic KV bytes per cached token: 2 (K+V) × layers × kv_heads × head_dim × 2 (bf16)."""
    layers = int(_cfg(model_config, "num_hidden_layers", 0))
    heads = int(_cfg(model_config, "num_attention_heads", 0))
    kv_heads = int(_cfg(model_config, "num_key_value_heads", heads or 0))
    head_dim = _cfg(model_config, "head_dim")
    if head_dim is None:
        hidden = int(_cfg(model_config, "hidden_size", 0))
        head_dim = hidden // heads if heads else 0
    head_dim = int(head_dim)
    if not (layers and kv_heads and head_dim):
        raise ValueError(
            f"cannot derive KV bytes/token from config "
            f"(layers={layers}, kv_heads={kv_heads}, head_dim={head_dim})")
    return 2 * layers * kv_heads * head_dim * 2


def compute_kv_budget(
    free_bytes: int,
    bytes_per_token: int,
    settings: Settings,
    *,
    gpu_total_mib: int = 0,
    tokens_per_second: float = 0.0,
) -> KvBudget:
    """Size a session's KV budget from post-load free VRAM.

    floor  = KV_SESSION_MIN_MINUTES of streaming at tokens_per_second
    cap    = (free − KV_SAFETY_MARGIN_MIB) / bytes_per_token
    budget = clamp(max(floor, KV_MEMORY_RATIO × cap), 0, cap)
    """
    tps = tokens_per_second or settings.kv_tokens_per_second_est
    floor_tokens = int(settings.kv_session_min_minutes * 60.0 * tps)
    usable = max(0, free_bytes - settings.kv_safety_margin_mib * MIB)
    cap_tokens = usable // max(1, bytes_per_token)
    budget_tokens = int(min(cap_tokens, max(floor_tokens, settings.kv_memory_ratio * cap_tokens)))
    floor_met = budget_tokens >= floor_tokens

    enforce = (settings.kv_enforce or "auto").strip().lower()
    if enforce == "auto":
        enforce_hard = bool(gpu_total_mib) and gpu_total_mib < KV_ENFORCE_HARD_BELOW_MIB
    else:
        enforce_hard = enforce == "hard"

    budget = KvBudget(
        bytes_per_token=bytes_per_token,
        budget_tokens=budget_tokens,
        floor_tokens=floor_tokens,
        cap_tokens=int(cap_tokens),
        tokens_per_second=tps,
        floor_met=floor_met,
        enforce_hard=enforce_hard,
    )
    if not floor_met:
        log.warning(
            "KV budget below the %.0f-minute floor: budget=%d tokens (%.1f min at %.0f tok/s), "
            "floor=%d — proceeding, sessions may be cut short",
            settings.kv_session_min_minutes, budget_tokens,
            budget.est_max_seconds / 60.0, tps, floor_tokens)
    return budget
