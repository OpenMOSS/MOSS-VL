# MOSS-VL: Scaling Video Understanding with Absolute Timestamps and Cross-attention RoPE

> This is the deep-dive companion to the [MOSS-VL GitHub repository](../README.md).
> Looking to **run the model**? Head to the [Quick Start](../README.md#-quick-start). This page is about *how it works and why*.

---

## 🎬 See it first

<div align="center">
  <video src="https://gist.github.com/user-attachments/assets/66406aaa-f09f-412c-87b1-97753895ef1f" width="70%" controls></video>
  <video src="https://gist.github.com/user-attachments/assets/d1ccae33-472f-4d92-96c4-fb6253b07189" width="70%" controls></video>
  <p>More on the <a href="https://OpenMOSS.github.io/MOSS-VL-Demo/#/">Interactive Demo Page</a> · <a href="https://huggingface.co/spaces/OpenMOSS-Team/MOSS-VL">HuggingFace Space</a></p>
</div>

MOSS-VL doesn't just describe a video — it reasons about *when* things happen and *how fast* they move. The rest of this post explains the three ideas that make that possible.

---

## Why video understanding is hard

Most vision-language models treat a video as a bag of frames. That loses two things that matter:

1. **Time.** Without a precise sense of *when* each frame occurs, a model can't reason about pacing, duration, or motion — velocity, acceleration, and trajectory all become guesswork.
2. **Spatio-temporal location.** Grounding a small object or a transient action means pinning it to a specific *(time, height, width)* coordinate, not just "somewhere in the clip".

MOSS-VL attacks both, along a systematic scaling strategy on three axes — **data**, **parameters**, and **context** — with two architectural ideas at its core: **absolute timestamps** and **Cross-attention RoPE (XRoPE)**.

---

## 🏗 Model Architecture

**MOSS-VL** adopts a cross-attention-based architecture that decouples visual encoding from cognitive reasoning. This design significantly reduces latency, enabling instantaneous responses to dynamic video streams. Natively supporting **interleaved modalities**, it processes complex sequences of images and videos within a unified pipeline — eliminating the need for heavy pre-processing.

<p align="center">
    <img src="../assets/structure.png" alt="MOSS-VL Architecture" width="90%"/>
    <br>
    <em>Figure 1: Overall architecture of MOSS-VL.</em>
</p>

The released checkpoints are 11B-parameter models (`model_type: moss_vl`, `MossVLForConditionalGeneration`) with a 256K-token context window (`max_position_embeddings = 262144`), pairing a self-developed vision encoder (`moss_vl_vision`) with a Qwen-based language backbone.

---

## 🧩 Absolute Timestamps

To ensure the model accurately perceives the pacing and duration of events, **MOSS-VL** injects **absolute timestamps** alongside each sampled frame, grounding the reasoning process in a **precise temporal reference**.

### Input Representation

<p align="center">
    <img src="../assets/timestamp_input.gif" alt="Timestamped Sequence Input Illustration" width="90%"/>
    <br>
    <em>Figure 2: Illustration of timestamped video sequence input.</em>
</p>

Each video is interleaved with **precise time markers**, where each timestamp is wrapped by **dedicated special tokens** (`<|time_start|>` … `<|time_end|>`) that explicitly anchor the **temporal location** of every visual frame:

```text
<|im_start|><|vision_start|>
<|time_start|>0.0 seconds<|time_end|><|image_pad|>
<|time_start|>1.2 seconds<|time_end|><|image_pad|>
<|time_start|>2.3 seconds<|time_end|><|image_pad|>
...
<|vision_end|>The video shows a dynamic scene with continuous actions...<|im_end|>
```

**🌟 Why this matters:**
- **Adaptability to Variable FPS:** The use of explicit timestamps allows the model to handle non-uniform sampling rates without loss of temporal context.
- **Precise Temporal Analysis:** Absolute time unlocks fine-grained action localization, grounding every response in exact temporal coordinates.
- **Motion Dynamics:** By exposing time intervals ($dt$), the model can reason about movement physics, enabling accurate estimation of velocity, acceleration, and trajectory.

---

## 🧬 Cross-attention RoPE (XRoPE)

MOSS-VL utilizes Cross-attention Rotary Position Embedding (XRoPE), tailored to its cross-attention based vision–language architecture. This mechanism maps text tokens and video patches into a unified 3D coordinate space defined by Time (t), Height (h), and Width (w).

<p align="center">
    <img src="../assets/3d-rope.png" alt="MOSS-VL XRoPE Architecture Illustration" width="80%"/>
    <br>
    <em>Figure 3: MOSS-VL with Cross-attention RoPE (XRoPE).</em>
</p>

To optimize cross-modal alignment, **XRoPE** is injected into the vision **Key (K)** for position-awareness while leaving the **Value (V)** untouched to preserve feature fidelity. In parallel, it is applied to the text **Query (Q)**, allowing the model to probe arbitrary spatio-temporal regions through direct coordinate alignment.

**🌟 Why this matters**

- **Unified Modality Modeling** — By expressing time as a shared dimension across both language and video, **XRoPE** enables seamless, cohesive video-text reasoning within a single coordinate system.
- **Precise Grounding** — Aligned ($t, h, w$) coordinates empower the model to localize small objects and transient actions anywhere in the 3D video volume — down to the patch and the moment.
- **Dynamic Input Support** — The 3D grid natively accommodates arbitrary aspect ratios and resolutions, eliminating the need for fixed-length padding or rigid input constraints.

---

## 📊 Training Strategy

MOSS-VL is trained using a multi-stage approach to progressively build multimodal capabilities.

<p align="center">
    <img src="../assets/total_data_distribution.png" alt="MOSS-VL Training Data Distribution" width="80%"/>
    <br>
    <em>Figure 4: Overall training data distribution of MOSS-VL.</em>
</p>

### Pre-training (PT)
MOSS-VL is pre-trained via a systematic four-stage curriculum that progressively builds up multimodal capabilities from the ground up:

* **Stage 1 — Vision-Language Alignment** — Establishes the initial bridge between visual features and the language space. Training on large-scale image-text pairs, the model learns to associate visual concepts with their textual counterparts while developing foundational OCR skills for text-in-image understanding.

* **Stage 2 — Large-Scale Multimodal Pre-training** — Scales up exposure to massive, diverse multimodal corpora, broadening the model's grasp of world knowledge and complex scenes — laying a robust foundation for general-purpose intelligence and high-resolution perception. In addition, short video clips are introduced at this stage to seed preliminary video understanding.

* **Stage 3 — High-Quality Multimodal Pre-training** — Elevates overall model quality by training on large volumes of high-quality perception, understanding, and reasoning data. This phase combines fine-grained image perception, complex multi-image comprehension, and high-fidelity video reasoning to sharpen the model's ability to capture intricate visual details and master temporal relationships across rich multimodal inputs.

* **Stage 4 — Annealing & Long-Context Extrapolation** — Stretches the model's horizon toward long-form video understanding, while a carefully designed annealing schedule trains on curated, top-tier multimodal data to push final performance to its peak.

| Stage | Strategy | Data Composition |
| :--- | :--- | :--- |
| **1** | **Vision-Language Alignment** | <img src="../assets/pt-stage1.png" width="400"/> |
| **2** | **Large-Scale Multimodal Pre-training** | <img src="../assets/pt-stage2.png" width="400"/> |
| **3** | **High-Quality Multimodal Pre-training** | <img src="../assets/pt-stage3.png" width="400"/> |
| **4** | **Annealing & Long-Context Extrapolation** | <img src="../assets/pt-stage4.png" width="400"/> |

### Supervised Fine-Tuning (SFT)
Building on the pre-trained foundation, **MOSS-VL** is further refined through **Supervised Fine-Tuning (SFT)** to align with human intent and unlock its full interactive and instruction-following capabilities.

<p align="center">
    <img src="../assets/SFT.png" alt="MOSS-VL SFT Data Composition" width="50%"/>
    <br>
    <em>Figure 5: Data composition of MOSS-VL SFT.</em>
</p>

### Reinforcement Learning from Human Feedback (RLHF)

> [!NOTE]
> MOSS-VL is currently undergoing RLHF training. Stay tuned for updates.

---

## 📊 Benchmark Analysis

We evaluated MOSS-VL across four key dimensions: Multimodal Perception, Multimodal Reasoning, Document/OCR, and Video Understanding. Across the board, MOSS-VL consistently ranks first or second against industry-leading baselines such as Qwen2.5-VL and Qwen3-VL.

<p align="center">
    <img src="../assets/MOSS-VL-Benchmark.png" alt="MOSS-VL Benchmark Comparison" width="90%"/>
    <br>
    <em>Figure 6: Detailed benchmark comparison between MOSS-VL and the Qwen series.</em>
</p>

The chart below visualizes MOSS-VL's balanced and well-rounded capability profile across 30+ specialized benchmarks. Represented by the solid blue region, MOSS-VL achieves the broadest overall coverage, with particularly strong showings in the Video Understanding and Multimodal Perception quadrants.

<p align="center">
  <img src="../assets/radar.png" width="600px" alt="MOSS-VL Evaluation Radar">
  <br>
  <em>Figure 7: Benchmark analysis of MOSS-VL.</em>
</p>

Headline numbers:
- **Video Understanding: 65.8** — +2 pts over Qwen3-VL; +8.3 pts over Qwen3-VL-8B-Instruct on `VSI-bench`.
- **Document / OCR: 83.9**.
- First or second across `VideoMME`, `MLVU`, `EgoSchema`, `BLINK`, `MMBench`, `CVBench`, and `VisuLogic`.

---

## 🔬 Ablations

> **TODO** — Add the studies that justify each design choice. Suggested figures:
> - XRoPE on/off vs. baseline RoPE — grounding accuracy on `VSI-bench` / temporal-localization tasks.
> - Absolute vs. relative timestamps — effect on motion-dynamics and duration questions.
> - Context-length extrapolation (Stage 4) — accuracy vs. video length.
>
> _Placeholder: fill in with real curves before publishing. Do not ship fabricated numbers._

---

## 🧭 Limitations & failure cases

> **TODO** — Honest failure gallery (this is what earns trust). Candidate categories:
> - Fine-grained counting under heavy occlusion.
> - Very long videos beyond the trained context horizon.
> - Fast motion / high-FPS scenes where sampling misses key frames.
>
> _Placeholder: add 2–3 real examples with model output and commentary._

---

## Get started

The model is open and Apache-2.0 licensed.

- **Run it:** [Quick Start](../README.md#-quick-start)
- **Download:** [MOSS-VL-Base-0408 / MOSS-VL-Instruct-0408](../README.md#-model-download)
- **Deploy:** [SGLang guide](../third_party/sglang/README.md)
- **Fine-tune:** [`finetune/README.md`](../finetune/README.md)

If MOSS-VL is useful in your work, please ⭐ the [repo](https://github.com/OpenMOSS/MOSS-VL) and cite the [technical report](../README.md#-citation).
