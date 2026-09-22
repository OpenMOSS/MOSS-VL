// The realtime session hook (backend_overhaul.md §F5): REST lifecycle +
// session socket + mic worklet + frame sampler + PCM player, surfaced as a
// small state machine the live page binds to. The server owns orchestration —
// this hook only streams media up and turns events into UI state.
import { useCallback, useEffect, useRef, useState } from 'react';

import { closedSegments, windowedCaption, type CapKind } from '../lib/captionStream';
import { startFrameSampler, type FrameSampler, type SamplerClock } from '../lib/frameSampler';
import { startMicCapture, type MicCapture } from '../lib/micWorklet';
import { PcmPlayer } from '../lib/pcmPlayer';
import {
  resolveApiUrl,
  SessionSocket,
  toWsUrl,
  type ServerEvent,
  type SocketState,
} from '../lib/sessionSocket';

export interface SessionUiConfig {
  asrLanguage: string;
  vadSensitivity: number; // UI slider 0..100 → wire 0..1
  captureMode: 'auto' | 'ptt';
  ttsVoice: string;
  /** Creation-time TTS engine pick: 'local' (boot provider pool) |
   *  'elevenlabs' | 'minimax' (cloud; needs the API key at boot). Not
   *  live-updatable — the engine is leased when the session is built. */
  ttsEngine?: 'local' | 'elevenlabs' | 'minimax';
  speakingRate: number;
  /** KV-prefilled at session CREATION only ('' → server default); the live
   *  session.update path ignores it by design (orchestrator _UPDATABLE_CONFIG). */
  systemPrompt?: string;
  /** Recorded with the session's history entry so the sidebar can tell a
   *  camera chat from a screen share / file stream / image / source-less
   *  session. Creation-time INITIAL source — mid-session switches ride the
   *  `input.video.source` event, not config. */
  videoSource?: 'camera' | 'screen' | 'file' | 'image' | 'none';
  /** Frames/second the browser samples and streams up. Creation-time only —
   *  the frame sampler cadence is fixed for the session's lifetime. */
  frameFps?: number;
  /** Sampling params baked into the generation engine at session CREATION
   *  (server GenerationParams). Not in _UPDATABLE_CONFIG → ignored on a live
   *  session.update, applied only when the next session is created. */
  temperature?: number;
  topP?: number;
  topK?: number;
  /** tokens-per-SECOND generation rate cap, enforced by the backend engine
   *  (sent on the wire as params.max_tokens_per_turn at session creation).
   *  The caption/transcript reveal is immediate, so this knob IS the visible
   *  text speed; audio follows at the same pace. */
  maxTokensPerTurn?: number;
}

export interface SessionMetrics {
  asrMs: number | null;
  ttftMs: number | null;
  ttfaMs: number | null;
  audioBufferS: number;
}

export interface MemoryRecallItem {
  id: number;
  text: string;
  session_ts: number | null;
  kind: string;
  media: string | null;
  score: number;
}

export interface UseSessionCallbacks {
  onCaption: (speaker: 'user' | 'ai', text: string) => void;
  onUserTurn: (text: string) => void;
  /** ONE closed sentence of the streaming reply — fired the moment its
   *  terminator arrives (and once more for an unterminated tail when the
   *  turn ends), in step with the subtitle advancing. Each call is one
   *  transcript bubble: the turn renders as a run of sentence bubbles.
   *  `emittedAt` is the SERVER-side generation time of the delta that closed
   *  the sentence (epoch seconds) — display arrival time lags and bunches
   *  under round churn, so bubbles should stamp with this when present.
   *  `capped` marks a chunk force-closed by the subtitle's 3-line cap (not
   *  model punctuation); `cont` marks the chunk that CONTINUES a capped one —
   *  the bubble renders as a follow-up (bead glyph instead of the sender
   *  label, shallow accent spine). */
  onAiSentence: (text: string, emittedAt?: number, capped?: CapKind, cont?: boolean) => void;
  /** Rolling ASR hypothesis while the user is speaking ('' = cleared — the
   *  final text arrives via onCaption/onUserTurn). Engines without streaming
   *  partials never call this. */
  onAsrPartial?: (text: string) => void;
  /** The server acked a source change (`input.video.source.changed`, also
   *  replayed on reconnect) — render it as the transcript's dedicated system
   *  bubble. */
  onSourceChanged?: (info: { kind: string; name?: string }) => void;
  /** The server recalled memory items relevant to the current turn
   *  ('memory.recalled', also replayed on reconnect — dedupe by item id) —
   *  render them as a transcript memory card. */
  onMemoryRecalled?: (items: MemoryRecallItem[]) => void;
  onMemoryNotice?: (message: string) => void;
  onLog: (text: string, type?: string) => void;
}

export interface ConnectOptions {
  stream: MediaStream | null;
  videoEl: HTMLVideoElement | null;
  config: SessionUiConfig;
  /** Timestamp basis of the session's FIRST segment: 'media' when videoEl
   *  plays a local file (frames advance with currentTime; pause/ended stop the
   *  flow), 'live' (default) for camera/screen/no source. Mid-session source
   *  swaps change it via setSamplerClock — see frameSampler's segment model. */
  initialClock?: SamplerClock;
}

const FRAME_FPS = 1;
const MAX_WS_BUFFER = 512 * 1024; // skip frames when the uplink backs up
const PLAYBACK_REPORT_MS = 1000;
// Listening-state failsafe: after a PTT release, if no transcription.done ever
// arrives (commit lost to a reconnect gap, ASR hiccup), stop showing the green
// listening state. Generous vs the ~1-2s a normal ASR finalize takes.
const LISTEN_CLEAR_MS = 8_000;
// Orb energy is driven by TEXT-generation throughput, not audio: a VLM streams
// tokens live while the single downstream TTS is serialized/late. This is the
// chars/sec that reads as a "full" swell (~fast bilingual streaming). Tunable.
const GEN_RATE_FULL = 22;
// Display pacing infrastructure (paceBufRef/drip) is kept but DISABLED:
// connect() always leaves paceCharsPerSRef at 0 so text reveals at the real
// model rate. (Knob-paced reveal was tried and caused captions to drift far
// behind the TTS audio; keep the plumbing in case pacing ever returns.)
const PACE_TICK_MS = 100;

function toWireConfig(c: Partial<SessionUiConfig>): Record<string, unknown> {
  const wire: Record<string, unknown> = {};
  if (c.asrLanguage !== undefined) wire.asr_language = c.asrLanguage;
  if (c.vadSensitivity !== undefined) wire.vad_sensitivity = Math.min(1, Math.max(0, c.vadSensitivity / 100));
  if (c.captureMode !== undefined) wire.capture_mode = c.captureMode;
  if (c.ttsVoice !== undefined) wire.tts_voice = c.ttsVoice;
  if (c.ttsEngine !== undefined) wire.tts_engine = c.ttsEngine;
  if (c.speakingRate !== undefined) wire.speaking_rate = c.speakingRate;
  if (c.videoSource !== undefined) wire.video_source = c.videoSource;
  if (c.systemPrompt !== undefined && c.systemPrompt.trim() !== '') {
    wire.system_prompt = c.systemPrompt.trim(); // empty draft → omit → server default
  }
  // sampling params nest under `params` (server SessionConfig.params); honored
  // at creation, ignored on a live update (not in _UPDATABLE_CONFIG). fps is
  // client-only (frame sampler) and never goes on the wire.
  const params: Record<string, unknown> = {};
  if (c.temperature !== undefined) params.temperature = c.temperature;
  if (c.topP !== undefined) params.top_p = c.topP;
  if (c.topK !== undefined) params.top_k = c.topK;
  // tokens-per-SECOND generation rate cap, paced by the backend engine
  // (server/schemas.py GenerationParams.max_tokens_per_turn → real_time_generate)
  if (c.maxTokensPerTurn !== undefined && c.maxTokensPerTurn > 0) {
    params.max_tokens_per_turn = Math.round(c.maxTokensPerTurn);
  }
  if (Object.keys(params).length > 0) wire.params = params;
  return wire;
}

export function useSession(callbacks: UseSessionCallbacks) {
  const [connected, setConnected] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [listening, setListening] = useState(false);
  const [responding, setResponding] = useState(false);
  // The context command currently executing server-side ('/compact' |
  // '/clear' | null). The server busy-locks other events while one runs, so
  // the UI gates its context-command buttons on this.
  const [contextCommand, setContextCommand] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [audioOut, setAudioOut] = useState<{ sample_rate: number; channels: number } | null>(null);
  const [metrics, setMetrics] = useState<SessionMetrics>({
    asrMs: null,
    ttftMs: null,
    ttfaMs: null,
    audioBufferS: 0,
  });

  const cb = useRef(callbacks);
  cb.current = callbacks;

  const socketRef = useRef<SessionSocket | null>(null);
  const playerRef = useRef<PcmPlayer | null>(null);
  const micRef = useRef<MicCapture | null>(null);
  const samplerRef = useRef<FrameSampler | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const reportTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const captureModeRef = useRef<'auto' | 'ptt'>('ptt');
  const pttActiveRef = useRef(false);
  const listenClearTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const micMutedRef = useRef(false);
  const videoOnRef = useRef(true);
  const responseIdRef = useRef<string | null>(null);
  const captionBufRef = useRef('');
  const aiSentencesRef = useRef(0); // sentences already committed for the open response
  // --- display pacing state (generation is uncapped; pacing is reveal-only) ---
  // Continuous drip: deltas queue in paceBufRef and reveal at the configured
  // chars/s REGARDLESS of round boundaries — a new model round never resets
  // the stream. The ONLY truncation points are user interrupt (barge-in /
  // cancel / context reseat → pending text discarded) and teardown (reveal
  // everything still pending).
  const paceBufRef = useRef('');
  const paceCharsPerSRef = useRef(0); // 0 = immediate (no pacing)
  const paceTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastEmittedAtRef = useRef<number | null>(null); // server gen time of the latest delta
  const activeRef = useRef(false);
  const connectGenerationRef = useRef(0);
  const micSetupAbortRef = useRef<AbortController | null>(null);
  // --- text-generation throughput (drives the orb) ---
  // genRate: smoothed chars/sec, estimated from inter-delta timing.
  // genLevel: the final buttery 0..1 the orb reads via getGenLevel() — swells
  // on a token burst, eases down (exp decay) through the gaps and at turn end.
  const genRateRef = useRef(0);
  const genDeltaTsRef = useRef(0); // performance.now() of the last text delta
  const genLevelRef = useRef(0);

  const updateMicGate = () => {
    const forward =
      !micMutedRef.current && (captureModeRef.current === 'auto' || pttActiveRef.current);
    micRef.current?.setForwarding(forward);
  };

  const teardown = useCallback((deleteSession: boolean) => {
    activeRef.current = false;
    connectGenerationRef.current += 1;
    micSetupAbortRef.current?.abort();
    micSetupAbortRef.current = null;
    // A session cut mid-response reveals + commits everything still pacing;
    // endPaceTurn runs unconditionally (paced mode moves text through a
    // buffer even with responseIdRef already cleared by response.done).
    endPaceTurn(false);
    if (reportTimerRef.current) clearInterval(reportTimerRef.current);
    reportTimerRef.current = null;
    if (listenClearTimerRef.current) clearTimeout(listenClearTimerRef.current);
    listenClearTimerRef.current = null;
    samplerRef.current?.stop();
    samplerRef.current = null;
    void micRef.current?.stop();
    micRef.current = null;
    socketRef.current?.close();
    socketRef.current = null;
    void playerRef.current?.close();
    playerRef.current = null;
    const sid = sessionIdRef.current;
    sessionIdRef.current = null;
    if (deleteSession && sid) {
      void fetch(resolveApiUrl(`/api/sessions/${sid}`), { method: 'DELETE', keepalive: true }).catch(
        () => undefined,
      );
    }
    pttActiveRef.current = false;
    responseIdRef.current = null;
    captionBufRef.current = '';
    aiSentencesRef.current = 0;
    paceBufRef.current = '';
    lastEmittedAtRef.current = null;
    genRateRef.current = 0; // orb back to rest
    genDeltaTsRef.current = 0;
    genLevelRef.current = 0;
    setConnected(false);
    setConnecting(false);
    setListening(false);
    setResponding(false);
    setContextCommand(null);
  }, []);

  // Commit each not-yet-committed closed sentence of the reply buffer as its
  // own transcript bubble (one per onAiSentence call). A bare '.' at the very
  // end of the buffer is ambiguous mid-stream (the next delta may turn it into
  // a decimal — "3." + "14"), so that sentence holds until more text arrives;
  // `final` (turn over) commits it plus any unterminated tail.
  const flushAiSentences = (final: boolean) => {
    const buf = captionBufRef.current;
    const closed = closedSegments(buf);
    const closedLen = closed.reduce((n, s) => n + s.text.length, 0);
    let commitCount = closed.length;
    if (!final && commitCount > 0 && closedLen === buf.length && buf.endsWith('.')) commitCount--;
    const emittedAt = lastEmittedAtRef.current ?? undefined;
    for (; aiSentencesRef.current < commitCount; aiSentencesRef.current++) {
      const seg = closed[aiSentencesRef.current];
      const prev = aiSentencesRef.current > 0 ? closed[aiSentencesRef.current - 1] : undefined;
      const s = seg.text.trim();
      if (s) cb.current.onAiSentence(s, emittedAt, seg.capped, !!prev?.capped);
    }
    if (final) {
      const tail = buf.slice(closedLen).trim();
      if (tail) cb.current.onAiSentence(tail, emittedAt, undefined, !!closed[closed.length - 1]?.capped);
    }
  };

  const revealFromCaptionBuf = () => {
    // subtitle shows ONE sentence at a time; reveal keeps model pace
    cb.current.onCaption('ai', windowedCaption(captionBufRef.current));
    // transcript: each sentence becomes its OWN bubble the moment it
    // closes — the log fills in step with the subtitle advancing
    flushAiSentences(false);
  };

  const commitTurnTail = () => {
    // turn over: commit any held-back sentence + the unterminated tail; the
    // empty caption lets the orb rest instead of freezing on the last sentence
    flushAiSentences(true);
    cb.current.onCaption('ai', '');
  };

  const dripPaceBuffer = () => {
    const budget = Math.max(1, Math.round((paceCharsPerSRef.current * PACE_TICK_MS) / 1000));
    if (paceBufRef.current) {
      const moved = paceBufRef.current.slice(0, budget);
      captionBufRef.current += moved;
      paceBufRef.current = paceBufRef.current.slice(moved.length);
      revealFromCaptionBuf();
    }
    if (!paceBufRef.current && paceTimerRef.current) {
      clearInterval(paceTimerRef.current);
      paceTimerRef.current = null;
    }
  };

  const startPaceDrip = () => {
    if (!paceTimerRef.current) paceTimerRef.current = setInterval(dripPaceBuffer, PACE_TICK_MS);
  };

  // Only real truncation point: user interrupt (barge-in / cancel / context
  // reseat) discards still-hidden text and starts fresh; teardown (discard=false)
  // instead reveals everything generated before committing. Either way the
  // visible tail commits and per-turn sentence bookkeeping restarts.
  const endPaceTurn = (discardPending: boolean) => {
    if (paceTimerRef.current) {
      clearInterval(paceTimerRef.current);
      paceTimerRef.current = null;
    }
    if (paceBufRef.current) {
      if (!discardPending) captionBufRef.current += paceBufRef.current;
      paceBufRef.current = '';
    }
    commitTurnTail();
    captionBufRef.current = '';
    aiSentencesRef.current = 0;
  };

  const handleEvent = (ev: ServerEvent) => {
    switch (ev.type) {
      case 'session.created': {
        setConnected(true);
        setConnecting(false);
        const out = ev.audio_out as { sample_rate: number; channels: number } | undefined;
        if (out) setAudioOut(out);
        if ((ev.replayed as number) > 0) {
          cb.current.onLog(`Session resumed (${ev.replayed} events replayed).`, 'live');
        }
        break;
      }
      case 'input.transcription.delta': {
        // rolling hypothesis while the user speaks — final text still arrives
        // via input.transcription.done (which clears this)
        cb.current.onAsrPartial?.(String(ev.text ?? ''));
        break;
      }
      case 'input.transcription.done': {
        const text = String(ev.text ?? '');
        if (listenClearTimerRef.current) {
          clearTimeout(listenClearTimerRef.current); // the real clear arrived
          listenClearTimerRef.current = null;
        }
        setListening(false);
        cb.current.onAsrPartial?.(''); // the turn is over either way
        if (text) {
          cb.current.onCaption('user', text);
          cb.current.onUserTurn(text);
        }
        break;
      }
      case 'input.text.done': {
        // typed turn accepted — the SERVER echo drives the caption/transcript
        // (same shape as the ASR path: no local echo, replay-safe)
        const text = String(ev.text ?? '');
        if (text) {
          cb.current.onCaption('user', text);
          cb.current.onUserTurn(text);
        }
        break;
      }
      case 'input.video.source.changed': {
        cb.current.onSourceChanged?.({
          kind: String(ev.kind ?? ''),
          ...(typeof ev.name === 'string' && ev.name ? { name: ev.name } : {}),
        });
        break;
      }
      case 'context.command': {
        const command = String(ev.command ?? '');
        const status = String(ev.status ?? '');
        if (status === 'running') {
          setContextCommand(command);
          playerRef.current?.stopAll();
          endPaceTurn(true);
          setListening(false);
          cb.current.onAsrPartial?.('');
        } else if (status === 'completed' || status === 'failed') {
          setContextCommand(null);
        }
        const message = status === 'completed'
          ? (command === '/clear' ? '上下文已清空' : '上下文已压缩')
          : status === 'failed' ? `${command} 失败：${String(ev.message ?? '')}`
          : `${command} 执行中`;
        cb.current.onMemoryNotice?.(message);
        break;
      }
      case 'memory.notice': {
        if (typeof ev.message === 'string') cb.current.onMemoryNotice?.(ev.message);
        break;
      }
      case 'memory.recalled': {
        if (Array.isArray(ev.items) && ev.items.length > 0) {
          const items = (ev.items as MemoryRecallItem[]).filter(
            (it) => it && typeof it.id === 'number',
          );
          if (items.length > 0) cb.current.onMemoryRecalled?.(items);
        }
        break;
      }
      case 'turn.speech_started':
        playerRef.current?.stopAll(); // duck: the user is talking over the reply
        endPaceTurn(true); // user interrupt: drop still-hidden text, start fresh
        setListening(true);
        break;
      case 'response.created':
        // rounds continue the same reveal stream: NO text-state resets here
        responseIdRef.current = String(ev.response_id);
        genRateRef.current = 0; // each turn's orb energy builds fresh from its tokens
        genDeltaTsRef.current = 0;
        setResponding(true);
        break;
      case 'response.text.delta':
        if (ev.response_id === responseIdRef.current) {
          if (typeof ev.emitted_at === 'number') lastEmittedAtRef.current = ev.emitted_at;
          const delta = String(ev.delta ?? '');
          // throughput: chars in this delta / the gap since the last one → an
          // EMA of chars/sec. Timing gaps ARE the signal (bursty = energetic).
          const nowT = performance.now();
          const prevTs = genDeltaTsRef.current;
          genDeltaTsRef.current = nowT;
          const gapS = prevTs ? Math.min(1, Math.max(0.02, (nowT - prevTs) / 1000)) : 0.1;
          genRateRef.current = genRateRef.current * 0.6 + (delta.length / gapS) * 0.4;
          // display pacing: knob>0 buffers text and drips at rate×1.6 chars/s;
          // knob 0/absent reveals immediately at model pace
          if (paceCharsPerSRef.current > 0) {
            paceBufRef.current += delta;
            startPaceDrip();
          } else {
            captionBufRef.current += delta;
            revealFromCaptionBuf();
          }
        }
        break;
      case 'response.done': {
        const reason = String(ev.stop_reason ?? 'end_turn');
        if (reason !== 'end_turn') {
          playerRef.current?.stopResponse(String(ev.response_id));
        }
        if (ev.response_id === responseIdRef.current) {
          setResponding(false);
          responseIdRef.current = null;
          if (typeof ev.gen_ended_at === 'number') lastEmittedAtRef.current = ev.gen_ended_at;
          // text: nothing to do — whatever already dripped stays displayed;
          // the reveal keeps going if the next round is still feeding
        }
        break;
      }
      case 'status': {
        const m = (ev.metrics ?? {}) as Record<string, number | null>;
        const q = (ev.queues ?? {}) as Record<string, number>;
        setMetrics({
          asrMs: m.asr_ms ?? null,
          ttftMs: m.vlm_ttft_ms ?? null,
          ttfaMs: m.tts_ttfa_ms ?? null,
          audioBufferS: q.audio_buffer_s ?? 0,
        });
        break;
      }
      case 'error': {
        const message = `[${ev.code}] ${ev.message}`;
        cb.current.onLog(message, 'red');
        if (ev.code === 'vlm_stopped' || ev.code === 'vlm_unavailable') setError(message);
        break;
      }
      default:
        break; // pong / session.updated / audio.done — no UI state
    }
  };

  const handleSocketState = (state: SocketState) => {
    if (state === 'reconnecting') {
      cb.current.onLog('Connection lost — reconnecting…', 'red');
    } else if (state === 'gone') {
      cb.current.onLog('Session expired on the server.', 'red');
      setError('session expired');
      teardown(false);
    } else if (state === 'busy') {
      cb.current.onLog('Server is at capacity — not reconnecting.', 'red');
      setError('server at capacity — please retry shortly');
      teardown(false);
    } else if (state === 'superseded') {
      cb.current.onLog('Session was taken over by another tab.', 'red');
      teardown(false);
    }
  };

  const connect = useCallback(
    async ({ stream, videoEl, config, initialClock }: ConnectOptions) => {
      if (activeRef.current) return false;
      activeRef.current = true;
      const generation = ++connectGenerationRef.current;
      const isCurrent = () => activeRef.current && connectGenerationRef.current === generation;
      setConnecting(true);
      setError(null);
      captureModeRef.current = config.captureMode;
      // reveal is immediate — the rate knob caps BACKEND generation
      // (params.max_tokens_per_turn), so the visible text speed equals the
      // real model rate; no client-side reveal pacing.
      paceCharsPerSRef.current = 0;
      try {
        const res = await fetch(resolveApiUrl('/api/sessions'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ config: toWireConfig(config) }),
        });
        if (!res.ok) {
          const body = await res.json().catch(() => ({}) as { detail?: string });
          if (res.status === 409) {
            // all GPU replicas host a live session (server sends Retry-After ≈ grace)
            throw new Error('所有模型通道都在使用中，请稍后重试 (all model slots busy — retry shortly)');
          }
          throw new Error(body.detail || `session create failed (HTTP ${res.status})`);
        }
        const body = (await res.json()) as { session_id: string; ws_url: string };
        if (!isCurrent()) {
          void fetch(resolveApiUrl(`/api/sessions/${body.session_id}`), { method: 'DELETE', keepalive: true }).catch(() => undefined);
          return false;
        }
        sessionIdRef.current = body.session_id;

        playerRef.current = new PcmPlayer();
        const socket = new SessionSocket(toWsUrl(resolveApiUrl(body.ws_url)), {
          onEvent: (event) => { if (isCurrent()) handleEvent(event); },
          onAudio: (pcm, descriptor) => {
            if (!isCurrent()) return;
            playerRef.current?.playChunk(
              String(descriptor.response_id),
              pcm,
              Number(descriptor.sample_rate) || 48000,
              Number(descriptor.channels) || 1,
            );
          },
          onStateChange: (state) => { if (isCurrent()) handleSocketState(state); },
        });
        socketRef.current = socket;
        socket.connect();

        if (videoEl) {
          // operator-set frames/sec (streaming model param) → falls back to the
          // 1 fps default; guarded to a sane floor so the sampler never stalls.
          // The sampler always starts (even with no source: readyState<2 keeps
          // frames from flowing) — its segmented clock IS the session clock
          // that source-change events and still-image frames stamp against.
          const frameFps = config.frameFps && config.frameFps > 0 ? config.frameFps : FRAME_FPS;
          const uplinkOk = () =>
            videoOnRef.current && (socketRef.current?.bufferedAmount ?? Infinity) < MAX_WS_BUFFER;
          samplerRef.current = startFrameSampler(
            videoEl,
            (jpeg, ts) => socketRef.current?.sendFrame(jpeg, ts),
            { fps: frameFps, shouldSend: uplinkOk, clock: initialClock ?? 'live' },
          );
        }
        if (stream && stream.getAudioTracks().length > 0) {
          const abort = new AbortController();
          micSetupAbortRef.current = abort;
          try {
            const mic = await startMicCapture(stream, (pcm) => {
              if (isCurrent()) socket.sendMic(pcm);
            }, { signal: abort.signal });
            if (!isCurrent()) {
              await mic.stop();
              return false;
            }
            micRef.current = mic;
            updateMicGate();
          } catch (error) {
            if (!isCurrent()) return false;
            cb.current.onLog(`Microphone unavailable; video/text remain connected: ${String(error)}`, 'red');
          } finally {
            if (micSetupAbortRef.current === abort) micSetupAbortRef.current = null;
          }
        }
        if (!isCurrent()) return false;
        reportTimerRef.current = setInterval(() => {
          const player = playerRef.current;
          if (player) {
            socketRef.current?.sendJSON('playback.status', {
              buffered_s: Number(player.bufferedSeconds().toFixed(2)),
            });
          }
        }, PLAYBACK_REPORT_MS);
        return true;
      } catch (exc) {
        if (!isCurrent()) return false;
        teardown(true);
        const message = exc instanceof Error ? exc.message : String(exc);
        setError(message);
        throw exc;
      }
    },
    [teardown],
  );

  const disconnect = useCallback(() => teardown(true), [teardown]);

  const pttDown = useCallback(() => {
    if (!socketRef.current || captureModeRef.current !== 'ptt') return;
    if (listenClearTimerRef.current) {
      clearTimeout(listenClearTimerRef.current); // a fresh turn owns the state
      listenClearTimerRef.current = null;
    }
    pttActiveRef.current = true;
    setListening(true);
    socketRef.current.sendJSON('input.audio.start');
    playerRef.current?.stopAll(); // duck immediately, don't wait for the server echo
    updateMicGate();
  }, []);

  const pttUp = useCallback(() => {
    if (!socketRef.current || !pttActiveRef.current) return;
    pttActiveRef.current = false;
    socketRef.current.sendJSON('input.audio.commit');
    // failsafe: if the commit's transcription.done never lands (lost to a
    // reconnect gap), don't wear the green listening state forever
    if (listenClearTimerRef.current) clearTimeout(listenClearTimerRef.current);
    listenClearTimerRef.current = setTimeout(() => {
      listenClearTimerRef.current = null;
      setListening(false);
    }, LISTEN_CLEAR_MS);
    updateMicGate();
  }, []);

  const sendText = useCallback((text: string) => {
    socketRef.current?.sendJSON('text.input', { text });
  }, []);

  /** Link the uploaded source video (CAS handle) to the session's history, so
   *  its replay can play the file back against the turns' media_ts anchors. */
  const attachVideo = useCallback((desc: { media: string; name?: string; durationS?: number }) => {
    socketRef.current?.sendJSON('video.attach', {
      media: desc.media,
      ...(desc.name ? { name: desc.name } : {}),
      ...(desc.durationS && Number.isFinite(desc.durationS) ? { duration_s: desc.durationS } : {}),
    });
  }, []);

  /** Flip the frame sampler's timestamp basis for a mid-session source swap
   *  ('media' = video file, 'live' = camera/screen/none). Call BEFORE
   *  sendSourceChange so the event's session_ts_start equals the new
   *  segment's base. Camera↔screen swaps keep one live segment — skip it. */
  const setSamplerClock = useCallback((clock: SamplerClock) => {
    samplerRef.current?.setClock(clock);
  }, []);

  /** Current session-clock ms (the frame sampler's segmented monotone clock). */
  const getSessionTsMs = useCallback(() => samplerRef.current?.currentTs() ?? 0, []);

  /** Single-shot frame for a still-image source: one 0x02 frame stamped with
   *  the current session clock; no sampling follows (the stage <video> has no
   *  source while an image is staged). */
  const sendImageFrame = useCallback((jpeg: ArrayBuffer) => {
    samplerRef.current?.sendStill(jpeg);
  }, []);

  /** Announce a source change to the orchestrator (`input.video.source`): it
   *  records the segment (file positions derive from session_ts_start), emits
   *  the journaled `input.video.source.changed` for the transcript's dedicated
   *  bubble, and buffers the model-facing timeline note. */
  const sendSourceChange = useCallback(
    (desc: {
      kind: 'camera' | 'screen' | 'file' | 'image' | 'none';
      name?: string;
      media?: string;
      durationS?: number;
      mediaOffsetS?: number;
    }) => {
      const tsS = (samplerRef.current?.currentTs() ?? 0) / 1000;
      socketRef.current?.sendJSON('input.video.source', {
        kind: desc.kind,
        session_ts_start: Math.round(tsS * 1000) / 1000,
        ...(desc.name ? { name: desc.name } : {}),
        ...(desc.media ? { media: desc.media } : {}),
        ...(desc.durationS && Number.isFinite(desc.durationS) ? { duration_s: desc.durationS } : {}),
        ...(desc.mediaOffsetS ? { media_offset: desc.mediaOffsetS } : {}),
      });
    },
    [],
  );

  const cancelResponse = useCallback(() => {
    socketRef.current?.sendJSON('response.cancel');
    playerRef.current?.stopAll();
    endPaceTurn(true);
  }, []);

  const updateConfig = useCallback((patch: Partial<SessionUiConfig>) => {
    if (patch.captureMode !== undefined) {
      captureModeRef.current = patch.captureMode;
      pttActiveRef.current = false;
      updateMicGate();
    }
    socketRef.current?.sendJSON('session.update', { config: toWireConfig(patch) });
  }, []);

  const setMicMuted = useCallback((muted: boolean) => {
    micMutedRef.current = muted;
    updateMicGate();
  }, []);

  const setVideoForwarding = useCallback((on: boolean) => {
    videoOnRef.current = on;
  }, []);

  /** Instantaneous TTS playback level (kept for callers that still want audio). */
  const getOutputLevel = useCallback(() => playerRef.current?.level() ?? 0, []);

  /** Text-generation throughput as a smoothed 0..1 the orb reads each RAF frame.
   *  Normalizes the chars/sec estimate, decays it by the time since the last
   *  delta (so it eases toward rest through token gaps / at turn end), then
   *  applies an asymmetric final smooth (quick swell, gentle settle) for a
   *  fluid, non-jittery Liquid-Glass motion. Stable identity, RAF-safe. */
  const getGenLevel = useCallback(() => {
    const nowT = performance.now();
    const since = genDeltaTsRef.current ? (nowT - genDeltaTsRef.current) / 1000 : 999;
    const decayed = genRateRef.current * Math.exp(-since / 0.5); // 0.5s ease-down
    const inst = Math.min(1, decayed / GEN_RATE_FULL);
    genLevelRef.current += (inst - genLevelRef.current) * (inst > genLevelRef.current ? 0.3 : 0.07);
    return genLevelRef.current;
  }, []);

  useEffect(() => () => teardown(true), [teardown]);

  return {
    connected,
    connecting,
    listening,
    responding,
    contextCommand,
    error,
    metrics,
    audioOut,
    connect,
    disconnect,
    pttDown,
    pttUp,
    sendText,
    attachVideo,
    setSamplerClock,
    getSessionTsMs,
    sendImageFrame,
    sendSourceChange,
    cancelResponse,
    updateConfig,
    setMicMuted,
    setVideoForwarding,
    getOutputLevel,
    getGenLevel,
  };
}
