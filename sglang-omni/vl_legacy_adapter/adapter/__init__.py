"""Legacy VL realtime protocol adapter for sglang-omni.

Translates the legacy VL WebSocket contract (start/ready/frame/output/stop)
onto the native sglang-omni video realtime protocol (WS /v1/video/realtime).
Purely additive: nothing in sglang-omni is modified.
"""
