"""GPU topology detection, worker placement and KV budgeting (multi-GPU serving).

`topology.probe_topology()` detects the box at startup, `placement.plan_placement()`
turns it into a per-process plan (VLM workers / ASR device / TTS sidecars), and
`kv_budget` sizes each session's KV headroom. `supervisor` owns the worker /
sidecar process lifecycles. Everything here is torch-free at import time — the
gateway must never create CUDA contexts.
"""
