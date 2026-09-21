# 部署与 Memory 运维

推荐安装与日常启停见[安装指南](../deployment/repro/README.md)。本页中的 `scripts/gpu/*` 用于既有集群环境，不是新机器的通用入口。

### Memory 后端资源与启动

`deploy.conf` 中 4B 默认使用 `DECIDE_LLM_MEM_FRAC=0.2`、`DECIDE_LLM_CONTEXT_LENGTH=16384`、
`DECIDE_LLM_MAX_RUNNING_REQUESTS=4`、`DECIDE_LLM_MAX_TOTAL_TOKENS=65536`，默认 GPU 1。
这些参数仅作用于 memory decide/compact，不改变主 VLM 的资源配置。

在 GPU 节点执行 `START_4B=1 _PI_ON_GPU=1 bash scripts/gpu/start_pi_agent.sh`；CPU 节点
省略 `_PI_ON_GPU=1` 即通过现有 SSH 通道执行。环境覆盖会完整转发，优先于 `deploy.conf`。
脚本核对实际模型、GPU 和资源参数，仅在完全匹配时复用；不匹配时只重启具有本部署归属的进程。
`FORCE_4B=1` 与 `FORCE_PI=1` 分别强制重启对应组件，旧的无标记实例仍需人工确认后迁移。

`MEMORY_BACKEND_WAIT_S` 默认 1200 秒，用于冷启动等待。pi-agent `/health` 与 `/ready`
验证模型生成、模型 ID 和分词预算接口；依赖未就绪返回 503；`/live` 仅报告进程存活。
配置及代码指纹不匹配时不会复用旧 pi-agent。鉴权配置变更需显式设置 `FORCE_PI=1`。

`MEMORY_PI_DECIDE_TIMEOUT_S` / `MEMORY_PI_COMPACT_TIMEOUT_S` 是包含连接、响应读取、
重试及退避的总期限；剩余期限通过请求头传给 pi-agent。4xx 不重试，暂时性 5xx/连接故障
最多尝试三次且不能突破总期限；pi-agent 超时返回 504 并取消下游请求。

### 进程与记忆保护

- 部署脚本通过部署目录与组件环境标记清理进程，并用 psutil 校验进程身份。不会仅按端口、进程名称或旧 PID 文件结束服务。共享/复用的无标记后端不属于自动关停范围。
- 首次从旧脚本迁移时，需人工确认并停止无标记旧实例；端口仍被占用时新启动会报错，不会抢占。tmux session 也必须带当前目录的归属标记；名称冲突可使用独立的 `DEMO_SESSION`。
- 长期记忆压缩读取完整原始 journal，按完整 QA 分段合并，并纳入中断内容；过期预取结果重新生成。原始记录不因压缩删除。
- pi-agent 使用实际后端 tokenizer 检查最终请求。默认应用上下文 16384 token，压缩单段输入 4096 token；这是请求预算，不是 GPU 显存配置。具体参数见[pi-agent 文档](../services/pi_agent/README.md)。
- 写入队列满时仅淘汰视频帧，不淘汰已接收的文本。队列全为文本时，新的 `note_utterance()` 返回 `False`，增加 `utterances_rejected` 并记录错误；该文本未被接收，可重试。此保护不等同于无限队列或保证新文本永不被拒绝。
