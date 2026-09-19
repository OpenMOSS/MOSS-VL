# Benchmarks

**English** | [简体中文](./README_zh.md)

General audio and multimodal benchmarks. For MOSS-VL realtime accuracy and latency, use the [three delivery tests](../deployment/moss_vl_realtime/README.md).

## Layout

| Directory | Contents |
| --- | --- |
| [tasks](./tasks/) | Task execution |
| [metrics](./metrics/) | Performance and accuracy metrics |
| [dataset](./dataset/) | Loaders and download helpers |
| [benchmarker](./benchmarker/) | Runner, data structures, and utilities |
| [eval](./eval/) | CLI entry points |
| [tts_serving](./tts_serving/README.md) | TTS serving contract and load tests |

Caches and generated results are local artifacts, not source files.

## Entry Points

| Entry | Purpose |
| --- | --- |
| [benchmark_tts_seedtts.py](./eval/benchmark_tts_seedtts.py) | TTS speed, WER, and voice cloning |
| [benchmark_tts_serving.py](./eval/benchmark_tts_serving.py) | TTS HTTP, streaming, WebSocket, voice, and batch contracts |
| [benchmark_omni_seedtts.py](./eval/benchmark_omni_seedtts.py) | Omni speech output speed and WER |
| [benchmark_omni_mmsu.py](./eval/benchmark_omni_mmsu.py) | MMSU audio comprehension |
| [benchmark_omni_mmau.py](./eval/benchmark_omni_mmau.py) | MMAU audio comprehension |
| [benchmark_omni_mmar.py](./eval/benchmark_omni_mmar.py) | MMAR audio reasoning |
| [benchmark_omni_mmmu.py](./eval/benchmark_omni_mmmu.py) | MMMU image understanding |
| [benchmark_omni_videomme.py](./eval/benchmark_omni_videomme.py) | Video-MME video understanding |
| [benchmark_omni_videoamme.py](./eval/benchmark_omni_videoamme.py) | Video-AMME video/audio understanding |
| [benchmark_asr_seedtts.py](./eval/benchmark_asr_seedtts.py) | ASR concurrency, WER, coverage, and RTFx |

Use `python -m benchmarks.eval.<entry> --help` for dataset, server, concurrency, and output options. Configure the server for the requested modality; text-only, speech, and transcription are separate workloads.

## TTS Example

Start a compatible S2-Pro server on port 8000, then run from the repository root:

```bash
python -m benchmarks.dataset.prepare --dataset seedtts
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model fishaudio/s2-pro --port 8000 \
  --output-dir results/s2pro_en --lang en --max-samples 50 --concurrency 8
```

Model launch commands are in the [TTS guide](../docs/basic_usage/tts.md). Avoid running several model servers on the same port or overlapping their GPU allocations.

## Generation and Evaluation

Split generation from transcription to avoid GPU contention. The second command reuses saved audio; stop the generation server before allocating the ASR device.

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --stream --model fishaudio/s2-pro --port 8000 \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --output-dir results/s2pro_en --lang en --max-samples 50 --concurrency 8

python -m benchmarks.eval.benchmark_tts_seedtts \
  --transcribe-only --model fishaudio/s2-pro \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --output-dir results/s2pro_en --lang en --device cuda:0

python -m benchmarks.eval.benchmark_tts_seedtts \
  --utmos-only --output-dir results/s2pro_en --device cuda:0
python -m benchmarks.eval.benchmark_tts_seedtts \
  --similarity-only --output-dir results/s2pro_en --device cuda:0
```

`--concurrency` and `--max-concurrency` are aliases for TTS. Flat `ref_audio`/`ref_text` is the default reference format; Higgs TTS and MOSS-TTS use `--ref-format references`. MOSS-TTS supports `--token-count auto`; plain TTS without cloning can use `--no-ref-audio` and a model-supported `--voice`.

The Omni SeedTTS entry also supports generation/transcription phases; transcription uses the ASR server selected by `--port`. Read its module help before reusing a TTS command.

## ASR Concurrency

Start the matching ASR server before running:

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000 \
  --max-samples 20 --concurrencies 2 --repeats 1 --stream
```

The report includes WER, evaluation coverage, RTF, RTFx (successful input-audio seconds per wall-clock second), and routing balance. Streaming adds text TTFT and inter-chunk latency.

## Offline Audio Quality

UTMOS estimates naturalness with the `balacoon/utmos` model; speaker similarity compares generated and reference voices using WavLM. Both operate on saved audio without a TTS server.

| Metric | Output | Default cache / override |
| --- | --- | --- |
| UTMOS | `utmos_results.json`: mean, median, P5/P95, evaluated/skipped counts, per-sample scores | `~/.cache/sglang-omni/utmos` / `UTMOS_CACHE_DIR` |
| Speaker similarity | `similarity_results.json`: mean cosine similarity multiplied by 100, counts, per-sample scores | `~/.cache/sglang-omni/speaker_sim` / `SEEDTTS_SIM_CACHE_DIR` |

```bash
python -m benchmarks.metrics.utmos --warm-cache
python -m benchmarks.metrics.speaker_similarity_assets --warm-cache
```

Report quality and latency alongside the model, dataset, hardware, and generation settings. A serving-contract pass is not a semantic or audio-quality score.
