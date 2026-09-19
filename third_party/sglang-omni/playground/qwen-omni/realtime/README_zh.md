# Wire Service：/v1/realtime 网页示例

[English](./README.md) | **简体中文**

![预览](preview.png)

Qwen3-Omni 的麦克风 WebSocket 客户端，可选择纯文本或文本加流式 PCM16 音频输出。使用原生 HTML/CSS/JS，无需前端构建。MOSS-VL 视频接口是独立的 `/v1/video/realtime`，对应应用见 [Demo](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo)。

## 启动

先选择一种后端配置。

纯文本，单卡：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --text-only --port 8765 --enable-realtime
```

文本与音频，单张 H200：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config examples/configs/qwen3_omni_colocated_h200.yaml \
  --colocate --port 8765 --enable-realtime
```

文本与音频，两张 GPU：

```bash
python examples/run_omni.py qwen3-speech-server \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 --gpu-talker 1 \
  --gpu-code-predictor 1 --gpu-code2wav 1 \
  --port 8765 --enable-realtime
```

另开终端启动网页：

```bash
cd playground/qwen-omni/realtime
python -m http.server 8080
```

访问 <http://127.0.0.1:8080>，选择输出模式，点击 **Open Wire**、**Begin Transmission** 后说话。连接期间不可改变输出模式；**Text + audio** 需要语音后端。远程访问麦克风页面需要 HTTPS，或通过 SSH 转发到 localhost。

## 测试

在仓库根目录执行浏览器播放状态回归：

```bash
node --test playground/qwen-omni/realtime/playback.test.js
```

## 界面

| 面板 | 含义 |
| --- | --- |
| Endpoint | WebSocket 地址 |
| Output | 默认纯文本，请求 `["text"]`，显示回答和用户原始转录；音频模式请求 `["text", "audio"]`，播放语音并显示回答 |
| Instructions | 通过 `session.update` 设置系统提示，只影响回答，不改变逐字转录 |
| Responses | 每个 VAD 轮次的回答；使用 `response.text.delta`，纯文本模式另显示 `conversation.item.input_audio_transcription.delta` |

## 音频与打断

- 服务端始终启用 VAD，无需手动 commit；说话后停顿即可触发。
- 页面内联创建 AudioWorklet，无需 package.json。
- 录音为 16 kHz、PCM16 小端，通过 base64 的 `input_audio_buffer.append` 发送。
- 音频输出为单声道 24 kHz PCM16 小端，由 Web Audio 排队播放。
- 文本加音频模式默认允许新语音打断。服务端取消回答，浏览器停止已排队播放；用户转录保留在历史中，被取消的助手输出不保留。纯文本模式继续完成当前回答。
- 自定义音频客户端在 `input_audio_buffer.speech_started` 时停止播放，并忽略被打断响应后续的 `response.audio.delta`，直到 `response.done`。若此时还未收到 `response.created`，保留待处理打断标记，拿到 ID 后再丢弃对应响应。
- 播放已开始时，发送 `conversation.item.truncate`，包含音频事件中的助手 `item_id`、`content_index: 0` 与已播放时长 `audio_end_ms`。服务端回复 `conversation.item.truncated` 并移除该助手历史项；因文本与音频未对齐，移除的是整段转录。
- 自动打断以 `response.done.status="cancelled"`、原因 `turn_detected` 结束；显式 `response.cancel` 的原因为 `client_cancelled`。

关闭某个音频会话的自动打断：

```json
{
  "type": "session.update",
  "session": {
    "modalities": ["text", "audio"],
    "turn_detection": {
      "type": "server_vad",
      "interrupt_response": false
    }
  }
}
```

只有 `turn_detection.interrupt_response` 可动态修改。VAD 的 `threshold`、`prefix_padding_ms` 和 `silence_duration_ms` 不会通过该接口重新配置。独立 `/v1/audio/speech` TTS 不受麦克风打断逻辑影响。连接断开时页面显示状态，需要重新连接。
