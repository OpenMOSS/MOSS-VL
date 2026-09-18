# 示例

[English](./README.md) | **简体中文**

在已安装的后端环境中，从仓库根目录执行命令。

## MOSS-VL Realtime

使用 [MOSS-VL-Realtime-SGLANG](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG) 模型，先完成[安装指南](../docs/get_started/installation_zh.md)。

```bash
bash deployment/moss_vl_realtime/start.sh "$MODEL_PATH"
python examples/moss_vl_realtime_client.py \
  --url ws://127.0.0.1:18500/v1/video/realtime \
  --prompt "Describe the visible scene." \
  --frame deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122/frame_0000.png --timestamp 0.0 \
  --frame deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122/frame_0001.png --timestamp 1.0
```

服务端与客户端分别在两个终端运行。默认服务为单卡四会话；客户端按时间戳回放，最后一帧关闭输入。参考语义测试使用 1 FPS。

显式单卡或 TP 参数用 `examples/run_moss_vl_realtime_server.py --help` 查看；协议见 [Cookbook](../docs/cookbook/moss_vl_realtime.md)，性能见[测试指南](../deployment/moss_vl_realtime/README_zh.md)。

## 其他模型启动器

`run_omni.py` 将 Qwen3-Omni 和 Ming-Omni 的启动配置组织为预设：

| 预设 | 用途 |
| --- | --- |
| `qwen3-text-server` | Qwen3-Omni 文本输出服务 |
| `qwen3-speech-server` | Qwen3-Omni 文本与音频服务 |
| `qwen3-speech` | 离线 Qwen3-Omni 语音生成 |
| `ming-text-server` | Ming-Omni 文本输出服务 |
| `ming-speech-server` | Ming-Omni 文本与音频服务 |
| `ming-speech` / `ming-text` | 离线 Ming-Omni 语音或文本生成 |

```bash
python examples/run_omni.py --help
python examples/run_omni.py qwen3-text-server --help
python examples/run_omni.py qwen3-text-server \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8000 --model-name qwen3-omni
python examples/run_omni.py ming-text-server \
  --model-path inclusionAI/Ming-flash-omni-2.0 --port 8001 --model-name ming-omni
```

只启动所需服务。各模型 GPU 放置与配置见[文档](../docs/README_zh.md)。旧版 `run_qwen3_omni_*.py` 和 `run_ming_omni_*.py` 保留为兼容入口。

## 新增启动器

在 [launchers](./launchers/) 中新增模型预设模块，并在 [_omni_launcher.py](./_omni_launcher.py) 注册。模型默认值、stage 修改和请求结构放在模型模块内；共享启动器只负责注册和 CLI 分发。
