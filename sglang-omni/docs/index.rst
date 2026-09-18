MOSS-VL Realtime Backend
==============================

Realtime video understanding with persistent WebSocket sessions, incremental
visual KV, and dynamic multi-session scheduling. This repository is built on
`SGLang-Omni <https://github.com/sgl-project/sglang-omni>`_ and uses the
`MOSS-VL-Realtime-SGLANG checkpoint <https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG>`_.

For browser interaction, voice, and memory, see the companion
`Demo <https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo>`_.
The `delivery guide <https://github.com/fnlp-vision/sglang-omni-realtime/tree/main/deployment/moss_vl_realtime>`_
contains launch commands, three public tests, and reference results.

.. toctree::
   :maxdepth: 1
   :caption: MOSS-VL Realtime

   README.md
   README_zh.md
   get_started/installation.md
   get_started/installation_zh.md
   cookbook/moss_vl_realtime.md
   cookbook/moss_vl_realtime_capacity.md

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/installation_xpu.md
   get_started/release_notes.md
   get_started/apiserver_quickstart.md


.. toctree::
   :maxdepth: 1
   :caption: Cookbook

   cookbook/higgs_tts.md
   cookbook/voxtral_tts.md
   cookbook/fishaudio_s2_pro.md
   cookbook/qwen3_tts.md
   cookbook/ming_tts.md
   cookbook/moss_tts.md
   cookbook/moss_tts_local.md
   cookbook/dots_tts.md
   cookbook/minimax_music3.md
   cookbook/zonos2.md
   cookbook/qwen3_asr.md
   cookbook/fun_asr.md
   cookbook/arkasr.md
   cookbook/moss_transcribe_diarize.md
   cookbook/whisper_asr.md
   cookbook/qwen3_omni.md
   cookbook/ming_omni.md
   cookbook/llada2_uni.md

.. toctree::
   :maxdepth: 1
   :caption: General Usage

   basic_usage/qwen3_omni.md
   basic_usage/audio_translations.md
   basic_usage/tts.md
   basic_usage/tts_process_topology.md
   basic_usage/omni_router.md
   basic_usage/mps_dp.md


.. toctree::
   :maxdepth: 1
   :caption: Benchmarks

   benchmarks/relay.md


.. toctree::
   :maxdepth: 1
   :caption: Developer Reference

   developer_reference/main.md
   developer_reference/apiserver_design.md
   developer_reference/pipeline.md
   developer_reference/config.md
   developer_reference/communication.md
   developer_reference/reference_encode_service.md
   developer_reference/profiler.md
   developer_reference/qwen3_asr_concurrency_profile.md
   developer_reference/rl_admin_control.md
   developer_reference/tts_model_integration.md

.. toctree::
   :maxdepth: 1
   :caption: Design

   design/gpu_radix_hash.md
   design/refactor_rfc.md
