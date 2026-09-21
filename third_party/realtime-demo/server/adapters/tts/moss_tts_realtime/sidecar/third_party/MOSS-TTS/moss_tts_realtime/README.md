# MOSS-TTS-Realtime Architecture

This document details the **MOSS-TTS-Realtime** architecture.

## 1. Overview
MOSS-TTS-RealTime is built on a scalable architecture consisting of a 1.7B-parameter backbone and a 200M-parameter local Transformer, designed to deliver high-quality, low-latency speech synthesis. The backbone encodes linguistic and contextual information and exposes its hidden states to the local Transformer, which autoregressively generates RVQ-based audio tokens. Audio reconstruction is performed using the MOSS-Audio-Tokenizer, enabling efficient and high-fidelity waveform synthesis.

The model adopts a hierarchical input design in which text tokens and speech tokens are processed at different levels, allowing the system to naturally support streaming text input and streaming audio output. This design enables incremental generation while preserving long-range context, making it well suited for real-time conversational scenarios.

---

## 2. Technical Specifications

| Feature | Specification |
| :--- | :--- |
| **Backbone Model** | Initialized from **Qwen3-1.7B** |
| **Depth Transformer** | 4 Transformer blocks, architecturally aligned with the Qwen3-1.7B backbone |
| **Audio Tokenizer** | **Cat** (Causal Audio Tokenizer) |
| **Sampling Rate** | 24,000 Hz |
| **Frame Rate** | 12.5 Hz (1s ≈ 12.5 tokens/blocks) |
| **Codebooks** | 16 RVQ layers |
| **Generation Mode** | Purely Autoregressive (AR) |

------

## 3. Usage
[MOSS-TTS-Realtime Usage](../docs/moss_tts_realtime_model_card.md)
