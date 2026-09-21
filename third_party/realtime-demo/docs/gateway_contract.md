# 薄网关接口契约

范围为 `/v1/realtime`，不包含 Demo `/api/sessions` 的语音和 memory 编排。平台负责 Bearer、客户额度和限流。本层负责一次性建连凭证、粘性路由、计量与保活，断连销毁会话。

## 生命周期与错误

- 创建在 `GATEWAY_CREATE_TIMEOUT_S`（默认 30 秒）内尝试，每个副本最多一次。失败/取消/超时均回收尚未移交的 slot 和迟到握手连接；必要的连接清理完成后才归还资源。
- REST 容量满：503 `session_capacity_exceeded`；没有可达副本或创建 deadline 到期：503 `no_available_replica`。不可达与容量占满不混为同一状态。
- token 绑定 session_id 与内部 epoch，一次性消费，默认 60 秒过期；reset/销毁撤销旧凭证。旧凭证即使在 reset 前被取出，也不能 attach 新 epoch。
- reset 保留网关 session_id，但后端会话和凭证均重建；不存在或被并发 DELETE 销毁时返回 404 `session_not_found`。不重放旧视频和模型 KV。
- 实例起始状态为 DOWN，首次探活成功才可路由。READY/BUSY/DOWN 均周期探测；探活和 metadata 请求失败不会抹掉活跃 slot 计数，原会话依其实际连接生命周期释放。

## 帧大小

- `GATEWAY_MAX_FRAME_BYTES` 默认 33554432（32 MiB）。
- `scripts/deploy/run_backend.sh` 默认传输上限为应用上限的两倍，即 64 MiB，可用 `WS_MAX_SIZE` 指定，但必须大于应用上限。
- `session.configured.max_frame_bytes` 表示 `min(网关上限, 后端通告上限)`；其余未知字段保留。只有需要修正此有效限制时才重编码 configured，普通数据事件保持原文透传。
- 在传输可接收范围内，超过有效上限的帧收到 `invalid_request`，随后 close 1009 并销毁会话。超过传输硬上限时，可能直接 close 1009，不保证额外 JSON 错误事件。
- 不使用部署脚本而直接启动 Uvicorn 时，必须同步配置 `--ws-max-size`。否则其默认 16 MiB 仍可能先于应用层拒绝。

## 模型版本

握手 `session.created`、REST 创建/查询响应和 JSONL 账单均包含 `model_version`、`model_version_source`。

优先使用可信后端握手中的非空 model_version；缺失时查部署配置：

```text
GATEWAY_MODEL_VERSION=release-identifier
GATEWAY_MODEL_VERSIONS={"http://replica-a:18500":"revision-a","http://replica-b:18500":"revision-b"}
```

按 URL 的配置优先于公共配置；来源分别为 backend 或 deployment。全部缺失时值为 null、来源为 unknown，不用 model 名称伪造版本。上线前应按实际发布清单配置稳定版本，混合版本部署使用 URL 映射。

## 验证与部署

`server/tests/test_gateway_lifecycle_fixes.py` 覆盖创建取消、迟到握手、有限重试、reset token、取消 reset、健康变化、模型版本、REST 竞争和启动脚本参数。

部署前应核对版本标识、应用与传输帧上限，并确保后端实例上限与 `SGLANG_OMNI_SESSIONS_PER_REPLICA` 一致。部署后重启相关服务，检查健康状态、版本信息和会话创建流程。
