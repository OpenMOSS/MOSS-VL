# 测试

[English](./README.md) | **简体中文**

## MOSS-VL Realtime

完成[后端安装](../docs/get_started/installation_zh.md)后，在仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/unit_test/moss_vl_realtime \
  tests/unit_test/serve/test_video_realtime*.py
```

这些测试不加载模型权重，覆盖增量请求、视觉 KV、batch、速率控制、WebSocket 生命周期、进程清理与交付脚本。

精度、单路时延、实时多路时延三项 GPU 测试见[交付测试指南](../deployment/moss_vl_realtime/README_zh.md)。参考性能结果与单元测试结果分别记录。

## 目录

| 路径 | 范围 |
| --- | --- |
| [unit_test/moss_vl_realtime](./unit_test/moss_vl_realtime/) | 实时模型与交付接口 |
| [unit_test/serve](./unit_test/serve/) | API 校验、流式输出与会话生命周期 |
| [unit_test/pipeline](./unit_test/pipeline/) | Stage 放置、IPC、路由与资源归属 |
| [unit_test/scheduling](./unit_test/scheduling/) | 请求准入、队列、缓存与调度 |
| [unit_test/router](./unit_test/router/) | 路由控制面与数据面 |
| [unit_test/client](./unit_test/client/) | 客户端事件与完成处理 |
| [test_model](./test_model/) | 模型集成与 GPU 基准 |
| [test_ci](./test_ci/) | 部署相关 CI 场景 |
| [data](./data/) | 小型共享媒体样例 |
| [utils](./utils/) | 共享测试工具 |

## 扩展检查

```bash
python -m pytest tests/unit_test -q
python -m pytest tests/test_model -m benchmark -v -s
```

部分单元测试需要加速器或其他模型的可选依赖。集成测试可能启动真实服务、下载模型并占用较多显存；在共享机器上运行前应阅读对应 fixture。跳过测试不等于该硬件已通过验证。

Marker 在 [pyproject.toml](../pyproject.toml) 注册。模型 CI 参数见 [test_model/conftest.py](./test_model/conftest.py)，可用 `python -m pytest tests/test_model --help` 查看。

## 新增测试

- 验证用户可见行为与资源归属，不依赖无关实现细节。
- 按行为所属组件放置测试，不新增根层级 `tests/test_*.py`。
- 复用已有 fixture 与 fake；GPU 测试明确硬件和模型要求。
- 有状态改动覆盖取消、异常与清理，只清理测试自己创建的资源。
- 样例保持小型、确定性；权重、下载数据集和生成结果不放入单元测试目录。
