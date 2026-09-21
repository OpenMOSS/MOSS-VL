"""Gateway plane (GATEWAY_PLAN.md §2-P1): a thin passthrough in front of the
sglang-omni realtime instances.

External clients speak the sglang-omni `/v1/video/realtime` data-plane event
protocol verbatim; this plane only adds session management (REST control plane
+ one-shot ws_tokens), replica routing/capacity, frame-size policing, and
transport-dead teardown. It never rewrites event semantics. The demo plane
(routers/session_ws.py) is untouched and coexists.
"""
