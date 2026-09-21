from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModel

from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
from mossttsrealtime.processing_mossttsrealtime import MossTTSRealtimeProcessor
from mossttsrealtime.streaming_mossttsrealtime import (
    AudioStreamDecoder,
    MossTTSRealtimeInference,
    MossTTSRealtimeStreamingSession,
)

SAMPLE_RATE = 24000


def _extract_codes(encode_result):
    return encode_result["audio_codes"]


def _load_audio(path: Path, target_sample_rate: int = 24000) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def _load_codec(codec_path: str, device: torch.device):
    codec = AutoModel.from_pretrained(codec_path, trust_remote_code=True).eval()
    return codec.to(device)


def _sanitize_tokens(
    tokens: torch.Tensor,
    codebook_size: int,
    audio_eos_token: int,
) -> tuple[torch.Tensor, bool]:
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.numel() == 0:
        return tokens, False
    eos_rows = (tokens[:, 0] == audio_eos_token).nonzero(as_tuple=False)
    invalid_rows = ((tokens < 0) | (tokens >= codebook_size)).any(dim=1)
    stop_idx = None
    if eos_rows.numel() > 0:
        stop_idx = int(eos_rows[0].item())
    if invalid_rows.any():
        invalid_idx = int(invalid_rows.nonzero(as_tuple=False)[0].item())
        stop_idx = invalid_idx if stop_idx is None else min(stop_idx, invalid_idx)
    if stop_idx is not None:
        tokens = tokens[:stop_idx]
        return tokens, True
    return tokens, False


def decode_audio_frames(
    audio_frames: list[torch.Tensor],
    decoder: AudioStreamDecoder,
    codebook_size: int,
    audio_eos_token: int,
) -> Iterator[np.ndarray]:
    for frame in audio_frames:
        tokens = frame
        if tokens.dim() == 3:
            tokens = tokens[0]
        if tokens.dim() != 2:
            raise ValueError(f"Expected [T, C] audio tokens, got {tuple(tokens.shape)}")
        tokens, _ = _sanitize_tokens(tokens, codebook_size, audio_eos_token)
        if tokens.numel() == 0:
            continue
        decoder.push_tokens(tokens.detach())
        for wav in decoder.audio_chunks():
            if wav.numel() == 0:
                continue
            yield wav.detach().cpu().numpy().reshape(-1)


def flush_decoder(decoder: AudioStreamDecoder) -> Iterator[np.ndarray]:
    final_chunk = decoder.flush()
    if final_chunk is not None and final_chunk.numel() > 0:
        yield final_chunk.detach().cpu().numpy().reshape(-1)



def fake_llm_text_stream(
    text: str,
    chunk_chars: int = 1,
    delay_s: float = 0.0,
) -> Iterator[str]:
    """
    Simulation LLM stream output text delta.
    The streaming response of OpenAI/vLLM is replaced in the real scenario.
    """
    if not text:
        return
    step = max(1, chunk_chars)
    for idx in range(0, len(text), step):
        if delay_s > 0 and idx > 0:
            time.sleep(delay_s)
        yield text[idx : idx + step]



# ── Core streaming Generation ──────────────────────────────────────────────────
def run_streaming_tts(
    session: MossTTSRealtimeStreamingSession,
    codec,
    decoder: AudioStreamDecoder,
    text_deltas: Iterator[str],
) -> Iterator[np.ndarray]:
    """
    Receive stream text delta → push_text → output wav chunk in real time.
    """
    codebook_size = int(getattr(codec.config, "codebook_size", 1024))
    audio_eos_token = int(getattr(session.inferencer, "audio_eos_token", 1026))

    with codec.streaming(batch_size=1):
        for delta in text_deltas:
            print(delta, end="", flush=True)
            audio_frames = session.push_text(delta)
            yield from decode_audio_frames(audio_frames, decoder, codebook_size, audio_eos_token)

        audio_frames = session.end_text()
        yield from decode_audio_frames(audio_frames, decoder, codebook_size, audio_eos_token)

        while True:
            audio_frames = session.drain(max_steps=1)
            if not audio_frames:
                break
            yield from decode_audio_frames(audio_frames, decoder, codebook_size, audio_eos_token)
            if session.inferencer.is_finished:
                break

        yield from flush_decoder(decoder)



def write_wav(out_path: Path, sample_rate: int, chunks: Iterator[np.ndarray]) -> None:
    all_chunks: list[np.ndarray] = []
    for chunk in chunks:
        all_chunks.append(chunk.astype(np.float32).reshape(-1))
    if not all_chunks:
        raise RuntimeError("No audio chunks produced.")
    audio = np.concatenate(all_chunks)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())



def run_one_turn(
    session: MossTTSRealtimeStreamingSession,
    codec,
    assistant_text: str,
    *,
    decode_chunk_frames: int = 3,
    decode_overlap_frames: int = 0,
    device: torch.device,
    delta_chunk_chars: int = 1,
    delta_delay_s: float = 0.0,
    out_path: Path | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    print(f"\n[TTS] assistant len({len(assistant_text)}): {assistant_text[:60]}...")

    decoder = AudioStreamDecoder(
        codec,
        chunk_frames=decode_chunk_frames,
        overlap_frames=decode_overlap_frames,
        decode_kwargs={"chunk_duration": -1},
        device=device,
    )

    text_deltas = fake_llm_text_stream(
        assistant_text,
        chunk_chars=delta_chunk_chars,
        delay_s=delta_delay_s,
    )
    wav_chunks = run_streaming_tts(
        session=session,
        codec=codec,
        decoder=decoder,
        text_deltas=text_deltas,
    )
    if out_path is not None:
        write_wav(out_path, sample_rate, wav_chunks)
        print(f"\n[OK] 写入: {out_path}")
    else:
        for _ in wav_chunks:
            pass
        print()



def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Simulation multiple rounds LLM streaming text → TTS streaming audio。"
    )
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--codec_path", type=str, required=True)
    p.add_argument(
        "--prompt_wav",
        type=str,
        required=True,
        help="Timbre prompt audio",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="out_turns",
        help="Output directory, generate turn_N.wav each round",
    )

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--attn_implementation", type=str, default="sdpa")
    p.add_argument("--sample_rate", type=int, default=SAMPLE_RATE)

    p.add_argument("--decode_chunk_frames", type=int, default=3)
    p.add_argument("--decode_overlap_frames", type=int, default=0)

    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.6)
    p.add_argument("--top_k", type=int, default=30)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--repetition_window", type=int, default=50)
    p.add_argument("--no_sample", action="store_true")
    p.add_argument("--max_length", type=int, default=3000)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--delta_chunk_chars", type=int, default=1)
    p.add_argument("--delta_delay_s", type=float, default=0.0)
    return p



def main():
    args = build_arg_parser().parse_args()
    if args.repetition_window is not None and int(args.repetition_window) <= 0:
        args.repetition_window = None

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    processor = MossTTSRealtimeProcessor(tokenizer)

    if args.attn_implementation and args.attn_implementation.lower() not in {"none", ""}:
        model = MossTTSRealtime.from_pretrained(
            str(args.model_path),
            attn_implementation=args.attn_implementation,
            torch_dtype=dtype,
        ).to(device)
    else:
        model = MossTTSRealtime.from_pretrained(str(args.model_path), torch_dtype=dtype).to(device)
    model.eval()

    codec = _load_codec(args.codec_path, device)

    with torch.inference_mode():
        prompt_audio = _load_audio(Path(args.prompt_wav), target_sample_rate=args.sample_rate)
        prompt_result = codec.encode(prompt_audio.unsqueeze(0).to(device))
        prompt_tokens = _extract_codes(prompt_result).squeeze(1).cpu().numpy()

    # -- Initialize session (cross-round reuse, KV cache remains above) --
    inferencer = MossTTSRealtimeInference(model, tokenizer, max_length=args.max_length)
    inferencer.reset_generation_state(keep_cache=False)

    session = MossTTSRealtimeStreamingSession(
        inferencer,
        processor,
        codec=codec,
        codec_sample_rate=args.sample_rate,
        codec_encode_kwargs={},
        prefill_text_len=processor.delay_tokens_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=not args.no_sample,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
    )
    session.set_voice_prompt_tokens(prompt_tokens)

    dialogue = [
        {
            "user_text": "Hey, I just landed in Paris. I have about six hours before my next flight. Any ideas?",
            "user_wav": "./audio/user1.wav",
            "assistant_text": (
                "Nice, welcome to Paris! Six hours is actually perfect for a short city walk. Are you traveling light, or do you have luggage with you?"
            ),
        },
        {
            "user_text": "Just a backpack. I don’t want anything too rushed.",
            "user_wav": "./audio/user2.wav",
            "assistant_text": (
                "Got it. In that case, I’d suggest starting near the Seine. You could walk from Notre-Dame to the Louvre, grab a coffee."
            ),
        },
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for turn_idx, turn in enumerate(dialogue):
        user_text = turn["user_text"]
        user_wav_path = Path(turn["user_wav"])
        assistant_text = turn["assistant_text"]

        print(f"\n{'='*60}")
        print(f"[Turn {turn_idx}] user: {user_text}")

        # ── 编码本轮 user 音频 ──
        with torch.inference_mode():
            user_audio = _load_audio(user_wav_path, target_sample_rate=args.sample_rate)
            user_result = codec.encode(user_audio.unsqueeze(0).to(device))
            user_tokens = _extract_codes(user_result).squeeze(1).cpu().numpy()

        session.reset_turn(
            user_text=user_text,
            user_audio_tokens=user_tokens,
            include_system_prompt=(turn_idx == 0),
            reset_cache=(turn_idx == 0),
        )

        run_one_turn(
            session=session,
            codec=codec,
            assistant_text=assistant_text,
            decode_chunk_frames=args.decode_chunk_frames,
            decode_overlap_frames=args.decode_overlap_frames,
            device=device,
            delta_chunk_chars=args.delta_chunk_chars,
            delta_delay_s=args.delta_delay_s,
            out_path=out_dir / f"turn_{turn_idx}.wav",
            sample_rate=args.sample_rate,
        )

    print(f"\n{'='*60}")
    print(f"[DONE] All {len(dialogue)} rounds of audio have been written to {out_dir}/")


if __name__ == "__main__":
    main()


#   python3 example_multiturn_stream_to_tts.py \
#     --model_path OpenMOSS-Team/MOSS-TTS-Realtime \
#     --codec_path OpenMOSS-Team/MOSS-Audio-Tokenizer \
#     --prompt_wav ./audio/prompt_audio1.mp3 