"""sglang-omni remote realtime backend (VLM_DEPLOY=sglang_omni).

The realtime VLM lives on remote sglang-omni servers (one WS session per
server) instead of local HF worker subprocesses. `SglangOmniPool` is the
gateway-facing facade (VlmAdapter protocol); `SglangOmniSession` translates
the sglang-omni event stream back into the demo orchestrator's control-token
text stream so server/session/orchestrator.py runs unchanged.
"""
from .pool import SglangOmniPool

__all__ = ["SglangOmniPool"]
