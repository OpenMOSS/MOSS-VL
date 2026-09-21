"""VLM worker process — one GPU, one model replica, at most one live session.

The gateway (server/app.py) spawns one of these per placement-plan GPU via
`server/gpu/supervisor.py` and talks to it through `WorkerVlmProxy`
(server/adapters/vlm/moss_vl_hf/worker_proxy.py). The worker owns everything CUDA: model
load, the `real_time_generate` daemon thread, JPEG decode, offline-chat decode
and KV accounting — the gateway process stays free of model GIL/CUDA work so
WebSocket handling never stalls.

Run standalone (debug):  python -m server.vlm_worker --port 9000 --worker-id 0
"""
