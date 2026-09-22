import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useSession, type MemoryRecallItem } from './hooks/useSession';
import { resolveApiUrl } from './lib/sessionSocket';
import { startMicCapture, type MicCapture } from './lib/micWorklet';
import { startAuroraFilament } from './lib/auroraFilament';
import {
  IconContext,
  CaretDown,
  ChatCircleDots,
  VideoCamera,
  NotePencil,
  ClockCounterClockwise,
  SquaresFour,
  Moon,
  Sun,
  Translate,
  MagnifyingGlass,
  ChartBar,
  Waveform,
  Code,
  Question,
  Paperclip,
  ListBullets,
  Microphone,
  PaperPlaneTilt,
  MicrophoneSlash,
  VideoCameraSlash,
  CameraRotate,
  Monitor,
  AppWindow,
  Browser,
  PhoneCall,
  PhoneDisconnect,
  Keyboard,
  FileVideo,
  FileImage,
  UploadSimple,
  Play,
  Sliders,
  User,
  Copy,
  Check,
  PencilSimple,
  ArrowsClockwise,
  SpeakerHigh,
  Stop,
  CircleNotch,
} from '@phosphor-icons/react';

// ==================== TYPES ====================
export interface ChatMessage {
  id: string;
  sender: 'user' | 'ai';
  text: string;
  generativeCard?: 'metrics' | 'audio' | 'code';
  cardId?: string;
  latency?: string;
  /** Model is thinking — render the Aurora Filament vessel instead of text. */
  isTyping?: boolean;
  /** Tokens still arriving — the trailing characters carry the trio shimmer. */
  isStreaming?: boolean;
  /** Generated reply eligible for the settled halo (set on real model turns
   *  and demo cards, never on welcome/history/error bubbles). Only the LAST
   *  message actually renders it — the halo marks the live end of the chat. */
  halo?: boolean;
  /** Preview URLs (thumb or data-URL) of images the user attached to this turn. */
  images?: string[];
  /** Blob URLs of uploaded videos attached to this turn (CAS-served, Range-seekable). */
  videos?: string[];
}

/** One pending paperclip attachment. `hash` is the CAS identity from
 *  POST /api/media; empty hash = legacy inline base64 fallback (backend
 *  media store unreachable) where `url` IS the data-URL. */
interface Attachment {
  id: string;
  kind: 'image' | 'video';
  hash: string;
  url: string;        // full media URL (or data-URL in fallback mode)
  previewUrl: string; // thumbnail/poster for the chip ('' = no poster)
  name: string;       // original filename
  /** in-flight upload: the chip presents the transfer on the thumbnail
   *  itself (frosted → filled sharp); sends are gated until it settles */
  uploading?: boolean;
  /** 0..1 while bytes move; null = length unknown / server-side processing
   *  (thumbnail extraction) → fully sharp under the processing shimmer */
  progress?: number | null;
  /** transfer outlived the 300ms grace window → engage the frost+fill
   *  presentation (quick uploads render the plain thumbnail directly) */
  slow?: boolean;
}

/** neutral film-frame glyph for a video chip with no poster */
const FILM_GLYPH = 'data:image/svg+xml;utf8,' + encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23888" stroke-width="1.6"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="M7 4v16M17 4v16M2 9h5M2 15h5M17 9h5M17 15h5"/></svg>');

/** Sidebar history entry (GET /api/history). */
interface HistoryEntry {
  conversation_id: string;
  title: string | null;
  kind: string;
  turn_count: number;
  created_at: number;
  /** realtime sessions only: the INITIAL source — 'camera' | 'screen' |
   *  'file' | 'image' | 'none' (from the session config) */
  video_source?: string | null;
}

/** One dialogue line of a recorded LIVE session (read-only replay). `mediaTs`
 *  is the video position (s) the turn happened at — the seek anchor in
 *  file-stream replays; null for turns without a frame reference. */
interface LiveReplayTurn {
  role: 'user' | 'assistant';
  text: string;
  ts: number;
  mediaTs: number | null;
  /** a journaled source change — renders as the dedicated system chip */
  system?: boolean;
}

interface LiveReplayData {
  cid: string;
  title: string | null;
  createdAt: number;
  /** the streamed source video (file-stream sessions), served from the CAS */
  video: { url: string; durationS: number | null } | null;
  turns: LiveReplayTurn[];
}

/** Wire-shaped conversation context for multi-turn chat. Image parts carry a
 *  CAS handle (`media: "sha256:…"`) or an inline data-URL (`image:`); video
 *  parts carry a CAS handle only (the backend samples frames from the blob) —
 *  all decoded in document order (vlm_hf._prepare_chat_messages). */
type WirePart =
  | { type: 'text'; text: string }
  | { type: 'image'; media?: string; image?: string }
  | { type: 'video'; media: string };
interface WireMessage {
  role: 'user' | 'assistant';
  content: string | WirePart[];
}
/** Cap the resent context — a marathon thread must not grow prefill unboundedly. */
const MAX_CONTEXT_MESSAGES = 24;

/** The server-side default realtime system prompt (board inference.py) — shown
 *  as EDITABLE starting content in the panel; sending it back verbatim is
 *  equivalent to omitting it. */
const DEFAULT_REALTIME_PROMPT =
  'You are a helpful AI assistant specializing in real-time video analysis. '
  + 'The video streams to you frame by frame. At every frame, you decide independently '
  + 'whether to respond or stay silent — output `<|silence|>` when nothing relevant has '
  + 'happened, and respond when the visual content warrants it.';

interface AudioPlayerState {
  isPlaying: boolean;
  currentIndex: number;
}

// ==================== HISTORY SCROLL RAIL ====================
// The dot-on-a-rail pager under the history search box (expanded sidebar and
// collapsed-sidebar popup each mount their own — position/index state is per
// instance). Drag runs on Pointer Events + setPointerCapture, so the gesture
// stays alive wherever the cursor goes with no document-level listeners; the
// outside-dismiss in App keys off pointerdown, so a drag that STARTS here can
// never dismiss the popup when it ENDS outside. (The old mouse version died to
// the UI-Events rule that a cross-element press+release synthesizes a click on
// their common ancestor — outside the popup — which the click-based dismiss
// then acted on.) Grab-offset, rect caching and the selection guard follow
// Radix ScrollArea; role="scrollbar" + arrow keys per the ARIA pattern.
function HistoryScrollRail({
  listRef,
  itemsCount,
  listId,
  label,
}: {
  listRef: { current: HTMLDivElement | null };
  itemsCount: number;
  listId: string;
  label: string;
}) {
  const [percent, setPercent] = useState<number>(0);
  const [index, setIndex] = useState<number>(1);
  const [dragging, setDragging] = useState<boolean>(false);
  const [hovered, setHovered] = useState<boolean>(false);
  const [focused, setFocused] = useState<boolean>(false);
  const trackRef = useRef<HTMLDivElement | null>(null);
  const draggingRef = useRef<boolean>(false);
  const rectRef = useRef<DOMRect | null>(null); // cached at press — no per-move reads
  const grabOffsetRef = useRef<number>(0);      // press-point → thumb-center distance
  const prevUserSelectRef = useRef<string>('');

  const indexFor = (pct: number) =>
    itemsCount < 1 ? 0 : Math.min(itemsCount, Math.max(1, Math.ceil(pct * (itemsCount - 1)) + 1));

  // Mirror the list's own scrolling (wheel / re-filter). During a drag the
  // pointer math is authoritative — scrollTop is quantized to device pixels
  // and following it back would stair-step the thumb.
  useEffect(() => {
    const list = listRef.current;
    if (!list) return;
    const sync = () => {
      if (draggingRef.current) return;
      const maxScroll = list.scrollHeight - list.clientHeight;
      const pct = maxScroll > 0 ? list.scrollTop / maxScroll : 0;
      setPercent(pct * 100);
      setIndex(indexFor(pct));
    };
    sync();
    list.addEventListener('scroll', sync, { passive: true });
    return () => list.removeEventListener('scroll', sync);
  }, [listRef, itemsCount]);

  const applyPct = (raw: number) => {
    const pct = Math.max(0, Math.min(1, raw));
    setPercent(pct * 100);
    setIndex(indexFor(pct));
    const list = listRef.current;
    if (list) {
      list.scrollTop = (list.scrollHeight - list.clientHeight) * pct;
    }
  };

  const pctAt = (clientX: number) => {
    const rect = rectRef.current;
    return rect && rect.width > 0 ? (clientX - grabOffsetRef.current - rect.left) / rect.width : 0;
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    const track = trackRef.current;
    if (!track) return;
    const rect = track.getBoundingClientRect();
    rectRef.current = rect;
    // Pressing the THUMB drags from the grabbed spot (no recentering jump);
    // pressing the bare rail jumps to the press point — native scrollbar feel.
    const onThumb = (e.target as HTMLElement).closest('.search-scrollbar-thumb');
    grabOffsetRef.current = onThumb ? e.clientX - (rect.left + (rect.width * percent) / 100) : 0;
    track.setPointerCapture(e.pointerId); // implicit release on pointerup/cancel
    draggingRef.current = true;
    setDragging(true);
    prevUserSelectRef.current = document.body.style.userSelect;
    document.body.style.userSelect = 'none'; // sweeping over the list must not select text
    applyPct(pctAt(e.clientX));
    e.preventDefault(); // keep focus in the search box; no selection start
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!draggingRef.current) return;
    applyPct(pctAt(e.clientX));
  };

  const endDrag = () => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    setDragging(false);
    rectRef.current = null;
    document.body.style.userSelect = prevUserSelectRef.current;
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (itemsCount < 1) return;
    const step = 1 / Math.max(1, itemsCount - 1);
    const pct = percent / 100;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') applyPct(pct + step);
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') applyPct(pct - step);
    else if (e.key === 'Home') applyPct(0);
    else if (e.key === 'End') applyPct(1);
    else return;
    e.preventDefault();
  };

  return (
    <div
      ref={trackRef}
      className="search-scrollbar-track"
      role="scrollbar"
      aria-controls={listId}
      aria-label={label}
      aria-orientation="horizontal"
      aria-valuemin={1}
      aria-valuemax={Math.max(1, itemsCount)}
      aria-valuenow={Math.max(1, index)}
      aria-valuetext={`${index} / ${itemsCount}`}
      tabIndex={0}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onLostPointerCapture={endDrag}
      onPointerEnter={() => setHovered(true)}
      onPointerLeave={() => setHovered(false)}
      onFocus={() => setFocused(true)}
      onBlur={() => setFocused(false)}
      onKeyDown={handleKeyDown}
    >
      <div className="search-scrollbar-fill" style={{ width: `${percent}%` }}></div>
      <div className="search-scrollbar-thumb" style={{ left: `${percent}%` }}>
        {(dragging || hovered || focused) && itemsCount > 0 && (
          <span className="scroll-index-tooltip">{index}</span>
        )}
      </div>
    </div>
  );
}

// ==================== AURORA FILAMENT (thinking + reply halo) ====================
// The model turn is ONE bubble for its whole life: born as an empty glass
// vessel whose edge carries waving trio-colored ribbons (painter: src/lib/auroraFilament.ts), type
// streams into the same element, and the ring stops when the reply settles.
// Colors follow the orb theme so the chat indicator and the live orb read as
// one organism.
//
// Reply-bubble halo. While the model THINKS and while tokens STREAM the bubble
// keeps the LIVE aurora ring
// (the thinking glow carries straight over — the box does not change identity
// mid-inference, only text fills in); once the reply settles it becomes the
// static, coreless remnant anchored at the top-left/bottom-right corners
// (Option B, research 2026-07-08). History bubbles stay quiet vessels, so
// exactly one glow lives in the view at a time.
function AuroraStaticHalo({ orbTheme, dark, animated = false }: {
  orbTheme: string; dark: boolean; animated?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const host = canvas?.parentElement;
    if (!canvas || !host) return;
    return startAuroraFilament(canvas, host, {
      orbTheme,
      dark,
      reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
      mode: animated ? 'animated' : 'static',
    });
  }, [orbTheme, dark, animated]);

  return <canvas ref={canvasRef} className="aurora-filament-canvas" aria-hidden="true" />;
}

// Liquid-Glass upload progress puck — the iOS media-send treatment (HIG:
// determinate whenever the length is known, keep it moving otherwise): the
// thumbnail renders immediately and a small dark-frosted glass disc floats
// centered on it carrying a thin white ring. Dark puck + white ring on BOTH
// page themes, because it always sits over media, never over the page.
// `progress` 0..1 fills the ring; null spins an indeterminate arc (server-side
// processing). Lives centered on the live PIP during the file's CAS transfer.
function GlassProgress({ progress, size = 40 }: {
  progress: number | null; size?: number;
}) {
  const C = 2 * Math.PI * 16; // r=16 in the 40-unit viewBox
  const frac = progress === null ? 0.3 : Math.max(0.04, Math.min(1, progress));
  return (
    <div
      className={`glass-progress${progress === null ? ' indeterminate' : ''}`}
      style={{ width: size, height: size }}
      role="progressbar"
      aria-valuenow={progress === null ? undefined : Math.round(frac * 100)}
    >
      <svg viewBox="0 0 40 40">
        <circle className="gp-track" cx="20" cy="20" r="16" />
        <circle className="gp-fill" cx="20" cy="20" r="16" strokeDasharray={`${(frac * C).toFixed(2)} ${C.toFixed(2)}`} />
      </svg>
    </div>
  );
}

// Memory recall entry in the transcript log: one centered tag (like the
// source-change chip) that expands on click to reveal the recalled rows in a
// block beneath it, and folds back on a second click — the bubbles after it
// reflow naturally. Module-level so the open/closed state survives re-renders.
function MemoryRecallLogEntry({ items, language, formatTs }: {
  items: MemoryRecallItem[]; language: string; formatTs: (seconds: number) => string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className={`log-entry memory${open ? ' open' : ''}`}>
      <button
        type="button"
        className="memory-recall-tag"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="memory-recall-glyph" aria-hidden="true" />
        {language === 'en'
          ? `Recalled ${items.length} ${items.length === 1 ? 'memory' : 'memories'}`
          : `回忆起 ${items.length} 条记忆`}
      </button>
      {open && (
        <div className="memory-recall-card">
          {items.map((item) => (
            <div key={item.id} className="memory-recall-row">
              <span className="memory-recall-text">
                {item.text || (language === 'en' ? 'camera view' : '画面')}
              </span>
              {typeof item.session_ts === 'number' && (
                <span className="memory-recall-ts">t={formatTs(item.session_ts)}</span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function App() {
  // ==================== STATE MANAGEMENT ====================
  // General UI
  const [isStreamingMode, setIsStreamingMode] = useState<boolean>(false);
  // Which swap-button (if any) is locked to its just-clicked glyph until the pointer leaves
  const [swapLocked, setSwapLocked] = useState<'mode' | 'theme' | 'history' | null>(null);
  const [theme, setTheme] = useState<'dark' | 'light'>('light');
  const [sidebarExpanded, setSidebarExpanded] = useState<boolean>(false);
  const [historyOpen, setHistoryOpen] = useState<boolean>(false);
  const [demosOpen, setDemosOpen] = useState<boolean>(false);
  const [language, setLanguage] = useState<'en' | 'zh'>('zh');
  const [dialogueStarted, setDialogueStarted] = useState<boolean>(false);

  // Typewriter effect state
  const [phraseIndex, setPhraseIndex] = useState<number>(0);
  const [displayedText, setDisplayedText] = useState<string>('');
  const [isDeleting, setIsDeleting] = useState<boolean>(false);
  const [showGreeting, setShowGreeting] = useState<boolean>(true);

  // Streaming Mode states
  const [focusedFeed, setFocusedFeed] = useState<'model' | 'user'>('model');
  const [showSessionPanel, setShowSessionPanel] = useState<boolean>(false);
  const [streamConversation, setStreamConversation] = useState<{ sender: 'user' | 'ai'; text: string; time: string; queued?: boolean; capped?: boolean; cont?: boolean; system?: boolean; memory?: MemoryRecallItem[] }[]>([]);
  // Queries typed/dictated BEFORE the session is live: each one gets its own
  // (dashed) bubble in the transcript log, and the whole batch collapses into
  // ONE opening request at connect (the server echo re-adds the merged bubble).
  const [pendingLiveQueries, setPendingLiveQueries] = useState<string[]>([]);
  // live mirror: the connect flush runs seconds after its closure was made —
  // it must see queries queued while the handshake was in flight
  const pendingLiveQueriesRef = useRef(pendingLiveQueries);
  pendingLiveQueriesRef.current = pendingLiveQueries;
  const [streamSpeaker, setStreamSpeaker] = useState<'user' | 'ai'>('ai');
  // rolling ASR hypothesis while the user speaks (input.transcription.delta);
  // '' = none — the final text arrives as a normal transcript turn
  const [liveAsrDraft, setLiveAsrDraft] = useState<string>('');
  // mirror + paired setter: pre-connect dictation writes the rolling
  // hypothesis STRAIGHT into the transcript's asr-pending bubble, and the
  // commit callback (fired from stale dictation closures) reads the latest
  const liveAsrDraftRef = useRef('');
  const setLiveAsrDraftBoth = (text: string) => {
    liveAsrDraftRef.current = text;
    setLiveAsrDraft(text);
  };
  const [asrLanguage, setAsrLanguage] = useState<string>('zh');
  const [vadSensitivity, setVadSensitivity] = useState<number>(50);
  // Streaming model params for the NEXT session (创建时生效, like the prompt):
  // fps = frames/sec the browser streams up; temperature/top_p/top_k and the
  // tokens/sec rate cap are baked into the generation engine at session
  // creation → the fields lock while a session is live. Defaults mirror the
  // server (GenerationParams in server/schemas.py: 0.7 / 0.8 / 20, rate cap
  // default 4 tok/s; frame sampler default 1 fps).
  // String-backed so decimals/empties type freely; parsed+clamped at connect
  // and normalized on blur (parseParam).
  const [streamFps, setStreamFps] = useState<string>('1');
  const [temperature, setTemperature] = useState<string>('0.7');
  const [topP, setTopP] = useState<string>('0.8');
  const [topK, setTopK] = useState<string>('20');
  const [maxTokensRate, setMaxTokensRate] = useState<string>('4');
  const parseParam = (s: string, lo: number, hi: number, fallback: number) => {
    const v = parseFloat(s);
    return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : fallback;
  };
  // System prompt for the NEXT realtime session. It is KV-prefilled at session
  // creation (board realtime patch) — it cannot change mid-session, so the
  // field locks while connected. The default prompt is EDITABLE starting
  // content (not a hint); the draft persists per browser.
  const [systemPrompt, setSystemPrompt] = useState<string>(() => {
    try {
      const saved = localStorage.getItem('moss_live_system_prompt');
      return saved && saved.trim() ? saved : DEFAULT_REALTIME_PROMPT;
    } catch { return DEFAULT_REALTIME_PROMPT; }
  });
  useEffect(() => {
    try { localStorage.setItem('moss_live_system_prompt', systemPrompt); } catch { /* private mode */ }
  }, [systemPrompt]);
  // Panel reorg: prompt + transcript log are the centerpiece; ASR/TTS/visual
  // config folds into a 更多设置 disclosure at the bottom. State persists per
  // browser and auto-collapses when a session connects (mid-call you care
  // about the transcript — most of the config is locked then anyway).
  const [moreSettingsOpen, setMoreSettingsOpen] = useState<boolean>(() => {
    try { return localStorage.getItem('moss_more_settings_open') === '1'; } catch { return false; }
  });
  useEffect(() => {
    try { localStorage.setItem('moss_more_settings_open', moreSettingsOpen ? '1' : '0'); } catch { /* private mode */ }
  }, [moreSettingsOpen]);

  // Typed input on the live page: the ⌨ bead at the LEFT end of the call pill
  // morphs the pill itself into a text field (iOS 26 morph pattern — one glass
  // object, two states; no second floating surface). Typed turns ride the same
  // session lane as voice: barge-in + freshest frame + captions + TTS.
  const [liveTextOpen, setLiveTextOpen] = useState<boolean>(false);
  const [liveTextDraft, setLiveTextDraft] = useState<string>('');
  const liveTextInputRef = useRef<HTMLInputElement | null>(null);
  // time tag beside the transcript title: a FILE session shows the position in
  // the streamed video (media clock), a camera session the elapsed call time
  const [liveMediaTime, setLiveMediaTime] = useState<{ kind: 'file' | 'live'; s: number } | null>(null);
  // PTT-first: button-up = end-of-turn, no server-side silence wait
  // (asr-tts_research.md §1B A3); `auto` (server VAD) stays behind the toggle.
  const [captureMode, setCaptureMode] = useState<'auto' | 'ptt'>('ptt');
  // default voice = Yuewen (zh_4, female B)
  const [voicePreset, setVoicePreset] = useState<string>('Yuewen');
  // TTS engine pick: local pool vs a cloud lane (ElevenLabs / MiniMax).
  // Creation-time (locked while a session is live); each cloud option only
  // appears when the backend reports that lane (API key present at boot). When
  // one is selected, the voice select lists its voices (voice_id on the wire).
  const [ttsEngine, setTtsEngine] = useState<'local' | 'elevenlabs' | 'minimax'>('minimax');
  const [cloudTts, setCloudTts] = useState<Partial<Record<'elevenlabs' | 'minimax', {
    ready: boolean; model?: string; voices: { voice_id: string; name: string }[];
  }>>>({});

  // Engine may DEFAULT to a cloud lane (no onChange fires) — once that lane's
  // voice registry lands, re-seat a stale voicePreset (e.g. the local 'Yuewen')
  // to the lane's first voice, or the cloud API 2054s and the session is mute.
  useEffect(() => {
    if (ttsEngine === 'local') return;
    const lane = cloudTts[ttsEngine];
    if (!lane?.voices?.length) return;
    if (!lane.voices.some((v) => v.voice_id === voicePreset)) {
      setVoicePreset(lane.voices[0].voice_id);
    }
  }, [ttsEngine, cloudTts, voicePreset]);

  // New sidebar modal and search states (the scroll-rail position/index/drag
  // state lives inside each HistoryScrollRail instance)
  const [activeModal, setActiveModal] = useState<'history' | 'demos' | null>(null);
  const [modalTop, setModalTop] = useState<number>(0);
  const [historyQuery, setHistoryQuery] = useState<string>('');

  // Real conversation history (GET /api/history) — refreshed whenever the
  // history UI opens; the search box filters client-side (same .includes()
  // semantics the rail was built around). Live and offline histories are
  // SEPARATE lists: the chat page shows recorded chat threads (kind=chat,
  // continuable), the live page shows recorded sessions (kind=realtime,
  // read-only replays).
  const [historyItems, setHistoryItems] = useState<HistoryEntry[]>([]);

  const refreshHistory = useCallback(async () => {
    try {
      const kind = isStreamingMode ? 'realtime' : 'chat';
      const res = await fetch(resolveApiUrl(`/api/history?limit=100&kind=${kind}`));
      if (!res.ok) return; // history disabled / backend down — keep the last list
      const body = (await res.json()) as { conversations: HistoryEntry[] };
      setHistoryItems(body.conversations.filter((c) => c.turn_count > 0 && c.kind === kind));
    } catch {
      /* backend unreachable — sidebar just shows the previous (or empty) list */
    }
  }, [isStreamingMode]);

  // page switch = different history list — drop the cached one so the other
  // mode's entries never flash while the refetch is in flight
  useEffect(() => {
    setHistoryItems([]);
  }, [isStreamingMode]);

  const filteredHistory = historyItems.filter((item) =>
    (item.title || '').toLowerCase().includes(historyQuery.toLowerCase())
  );

  // Chat Mode State
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      sender: 'ai',
      text: '欢迎来到 MOSS 实时体验系统（离线对话模式）。您可以尝试索取系统运行日志、语音波形或玻璃微态样式表设计。',
    }
  ]);
  const [chatInput, setChatInput] = useState<string>('');
  const [chatInputBlocked, setChatInputBlocked] = useState<boolean>(false);
  const chatInputRef = useRef<HTMLInputElement | null>(null);
  // Keep the composer focused after a send so the user can keep typing without
  // re-clicking. rAF defers past the commit that morphs/redisables the send
  // button (a synchronous focus() would race that re-render).
  const focusChatInput = () => requestAnimationFrame(() => chatInputRef.current?.focus());
  const [isDictating, setIsDictating] = useState<boolean>(false);
  const [audioPlayers, setAudioPlayers] = useState<Record<string, AudioPlayerState>>({});
  // Pending paperclip attachments (uploaded to the CAS via POST /api/media)
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  // In-flight attachment uploads by chip id — the chip's × aborts its XHR
  const attachmentXhrsRef = useRef<Map<string, XMLHttpRequest>>(new Map());

  // ---- media viewer (lightbox): click any thumbnail/bubble media → glass modal.
  // Dots (top-left, macOS-style beads): first closes, second reveals media info
  // on hover. Backdrop click and Escape also close.
  const [mediaViewer, setMediaViewer] = useState<{ kind: 'image' | 'video'; url: string } | null>(null);
  const [viewerInfo, setViewerInfo] = useState<Array<[string, string]> | null>(null);
  const [viewerInfoOpen, setViewerInfoOpen] = useState<boolean>(false);
  const viewerMediaRef = useRef<HTMLImageElement | HTMLVideoElement | null>(null);
  // grace timer so the card survives the pointer's trip from dot to card
  const viewerInfoHideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const viewerInfoZoneEnter = () => {
    if (viewerInfoHideTimer.current) clearTimeout(viewerInfoHideTimer.current);
    viewerInfoHideTimer.current = null;
    void loadViewerInfo();
  };

  const viewerInfoZoneLeave = () => {
    if (viewerInfoHideTimer.current) clearTimeout(viewerInfoHideTimer.current);
    viewerInfoHideTimer.current = setTimeout(() => setViewerInfoOpen(false), 200);
  };

  const openMediaViewer = (kind: 'image' | 'video', displayUrl: string) => {
    // bubbles/chips show the thumb — the viewer wants the full blob
    const url = displayUrl.endsWith('/thumb') ? displayUrl.slice(0, -'/thumb'.length) : displayUrl;
    setViewerInfo(null);
    setViewerInfoOpen(false);
    setMediaViewer({ kind, url });
  };

  const closeMediaViewer = () => {
    setMediaViewer(null);
    setViewerInfoOpen(false);
  };

  const fmtBytes = (n: number) =>
    n >= 1024 * 1024 ? `${(n / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`;

  /** Lazily assemble the info rows on first hover of the info dot. Size/mime
   *  come from a 1-byte Range probe (the CAS serves 206 + Content-Range);
   *  dimensions/duration come from the loaded media element itself. */
  const loadViewerInfo = async () => {
    setViewerInfoOpen(true);
    if (viewerInfo || !mediaViewer) return;
    const zh = language !== 'en';
    const rows: Array<[string, string]> = [];
    const el = viewerMediaRef.current;
    rows.push([zh ? '类型' : 'Type', mediaViewer.kind === 'video' ? (zh ? '视频' : 'Video') : (zh ? '图片' : 'Image')]);
    if (el instanceof HTMLImageElement && el.naturalWidth) {
      rows.push([zh ? '尺寸' : 'Size', `${el.naturalWidth} × ${el.naturalHeight}`]);
    } else if (el instanceof HTMLVideoElement && el.videoWidth) {
      rows.push([zh ? '尺寸' : 'Size', `${el.videoWidth} × ${el.videoHeight}`]);
      if (Number.isFinite(el.duration) && el.duration > 0) {
        const s = Math.round(el.duration);
        rows.push([zh ? '时长' : 'Duration', `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`]);
      }
    }
    const url = mediaViewer.url;
    if (url.startsWith('data:')) {
      const b64 = url.split(',', 2)[1] ?? '';
      rows.push([zh ? '大小' : 'Bytes', fmtBytes(Math.floor(b64.length * 0.75))]);
      rows.push([zh ? '来源' : 'Source', zh ? '内联（未入库）' : 'inline (not stored)']);
    } else {
      try {
        const res = await fetch(url, { headers: { Range: 'bytes=0-0' } });
        const mime = res.headers.get('content-type');
        const range = res.headers.get('content-range'); // "bytes 0-0/12345"
        if (mime) rows.push(['MIME', mime.split(';')[0]]);
        const total = range?.split('/')[1];
        if (total && !Number.isNaN(Number(total))) rows.push([zh ? '大小' : 'Bytes', fmtBytes(Number(total))]);
      } catch { /* backend unreachable — show what we have */ }
      const hex = url.match(/\/api\/media\/([0-9a-f]{64})/)?.[1];
      if (hex) rows.push(['SHA-256', `${hex.slice(0, 10)}…${hex.slice(-6)}`]);
    }
    setViewerInfo(rows);
  };

  // Escape closes the viewer (outside-click lives on the backdrop element)
  useEffect(() => {
    if (!mediaViewer) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeMediaViewer();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [mediaViewer]);

  // Streaming Mode State (`streamConnected` itself lives in useSession below)
  const [micMuted, setMicMuted] = useState<boolean>(false);
  const [videoEnabled, setVideoEnabled] = useState<boolean>(false);
  // Media-file streaming mode (board parity): a picked local VIDEO plays in
  // the pip <video> via an object URL — never uploaded for inference — and its
  // frames stream to the model through the same sampler as the camera; a
  // picked IMAGE is staged on the stage and sent as ONE frame. In file mode
  // the connect button becomes ▶ Start.
  const [mediaFile, setMediaFile] = useState<{ kind: 'video' | 'image'; name: string; url: string; size: number; mime: string } | null>(null);
  const mediaFileInputRef = useRef<HTMLInputElement | null>(null);
  const mediaFileRef = useRef(mediaFile);
  mediaFileRef.current = mediaFile;
  // Background CAS upload of the picked file (starts at pick time), so the
  // session's history replay can play the video back. `key` is the object URL
  // of the pick it belongs to; promise resolves to the CAS handle or null.
  const videoUploadRef = useRef<{ key: string; promise: Promise<string | null> } | null>(null);
  // Upload feedback for the picked file: 0..1 while bytes move, 'processing'
  // once they've landed and the server extracts the poster, null = settled.
  // Drives the PIP's centered glass puck + the file bead's bare ring.
  const [liveUploadPct, setLiveUploadPct] = useState<number | 'processing' | null>(null);
  const liveUploadXhrRef = useRef<XMLHttpRequest | null>(null);
  // pip info card (the viewer's two-bead pattern: × close · i info-on-hover)
  const [pipInfoOpen, setPipInfoOpen] = useState<boolean>(false);
  const pipInfoHideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Phone extras: front/rear camera facing (button shows only when the device
  // actually has >1 camera) and FaceTime-style tap-to-hide of the stage chrome.
  const [cameraFacing, setCameraFacing] = useState<'user' | 'environment'>('user');
  const [hasMultipleCameras, setHasMultipleCameras] = useState<boolean>(false);
  const [stageChromeHidden, setStageChromeHidden] = useState<boolean>(false);
  // Live video source: the camera bead opens a glass source menu (camera /
  // screen share) instead of toggling blindly — a picked screen rides the
  // EXACT same pipeline as the camera (track → localStream → <video> →
  // frame sampler); only the acquisition differs (getDisplayMedia).
  const [videoSourceKind, setVideoSourceKind] = useState<'camera' | 'screen'>('camera');
  const [videoMenuOpen, setVideoMenuOpen] = useState<boolean>(false);
  const videoMenuAnchorRef = useRef<HTMLDivElement | null>(null);
  // our own screen-surface pre-picker (centered glass sheet, Meet's
  // tab/window/screen pattern); the browser's native confirm still follows —
  // the web cannot enumerate windows, only hint the picker
  const [screenPickerOpen, setScreenPickerOpen] = useState<boolean>(false);
  // The ONE derived source the unified Media control reflects: a staged file/
  // image wins, else the live camera/screen when enabled, else none. The
  // underlying states stay (pip classes, analyser, flip bead all key off them);
  // mediaFile and videoEnabled are kept mutually exclusive by the source paths.
  const activeSource: 'none' | 'camera' | 'screen' | 'file' | 'image' = mediaFile
    ? (mediaFile.kind === 'video' ? 'file' : 'image')
    : videoEnabled
      ? videoSourceKind
      : 'none';
  // Simulated session for the live-page sidebar demo: drives the same
  // data-session drift/caption choreography as a real connection.
  const [subtitleDemoActive, setSubtitleDemoActive] = useState<boolean>(false);
  const subtitleDemoTimersRef = useRef<any[]>([]);
  const [orbTheme, setOrbTheme] = useState<string>('fluid');
  const [voiceRate, setVoiceRate] = useState<number>(1.0);
  const [captionsText, setCaptionsText] = useState<string>('已就绪，等待开启视频通话。点击绿色电话图标开始。');

  // Read-only replay of a recorded live session (glass modal): file-stream
  // sessions play the stored video with the dialogue anchored at each turn's
  // media_ts (click a timestamp chip → seek); camera sessions replay the
  // transcript log alone. Nothing here writes back into the conversation.
  const [liveReplay, setLiveReplay] = useState<LiveReplayData | null>(null);
  const [replayTime, setReplayTime] = useState<number>(0);
  const replayVideoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    if (!liveReplay) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setLiveReplay(null);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [liveReplay]);

  // ==================== REFS ====================
  const streamingCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const localVideoRef = useRef<HTMLVideoElement | null>(null);

  const languageRef = useRef(language);
  languageRef.current = language;
  const chatMessagesEndRef = useRef<HTMLDivElement | null>(null);
  const recognitionRef = useRef<any>(null);
  const audioIntervalsRef = useRef<Record<string, any>>({});
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Media capture nodes refs
  const localStreamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);

  // ==================== REALTIME SESSION (backend session plane) ====================
  // The server orchestrates VAD → ASR → VLM → TTS; this hook streams mic/frames
  // up one WebSocket and turns its events into captions, transcript and audio.
  const session = useSession({
    // the subtitle bar carries the MODEL's voice only — user words live in
    // the transcript; a user turn just clears any lingering model sentence
    onCaption: (speaker, text) => {
      if (speaker === 'user') {
        setCaptionsText('');
        return;
      }
      setStreamSpeaker('ai');
      setCaptionsText(text);
    },
    onUserTurn: (text) =>
      setStreamConversation((prev) => [...prev, { sender: 'user', text, time: getFormattedTime() }]),
    // one closed sentence = one transcript bubble, in step with the subtitle;
    // stamped with the model-side generation time when the event carries one.
    // A cap-carved chunk (subtitle 3-line hardcap, not model punctuation)
    // reads on with an ellipsis; its continuation renders as a follow-up
    // (water-bead glyph instead of the sender label, shallow accent spine).
    onAiSentence: (text, emittedAt, capped, cont) =>
      setStreamConversation((prev) => [
        ...prev,
        {
          sender: 'ai',
          text: capped ? `${text}…` : text,
          time: formatEventTime(emittedAt),
          capped: !!capped,
          cont: !!cont,
        },
      ]),
    // rolling ASR hypothesis: a pending entry at the tail of the transcript
    // log ONLY (never the subtitle bar); '' clears it — the final turn text
    // then lands through onUserTurn above
    onAsrPartial: (text) => setLiveAsrDraft(text),
    // a source change renders as its OWN dedicated system bubble (user
    // preference: never a prefix on the next user bubble); the server journals
    // the event, so reconnect replay and history redraw it the same way
    onSourceChanged: ({ kind, name }) => {
      setStreamConversation((prev) => [
        ...prev,
        { sender: 'user', system: true, text: sourceChangeLabel(kind, name), time: getFormattedTime() },
      ]);
    },
    // memory recall renders as its own compact card; the event is journaled
    // and REPLAYED on reconnect, so drop item ids already shown in earlier
    // memory entries (and within the batch itself)
    onMemoryNotice: (message) => {
      setStreamConversation((prev) => [...prev,
        { sender: 'ai', system: true, text: message, time: getFormattedTime() }]);
    },
    onMemoryRecalled: (items) => {
      setStreamConversation((prev) => {
        const seen = new Set<number>();
        for (const e of prev) if (e.memory) for (const it of e.memory) seen.add(it.id);
        const fresh = items.filter((it) => {
          if (seen.has(it.id)) return false;
          seen.add(it.id);
          return true;
        });
        if (fresh.length === 0) return prev;
        return [...prev, { sender: 'ai', text: '', memory: fresh, time: getFormattedTime() }];
      });
    },
    onLog: (text, type) => addLog(text, type),
  });
  const streamConnected = session.connected;
  // live mirror for callbacks armed before a connect (dictation commit)
  const sessionConnectedRef = useRef(session.connected);
  sessionConnectedRef.current = session.connected;

  /** Shared label for a source-change system chip (live bubble + replay). */
  const sourceChangeLabel = (kind?: string, name?: string) => {
    const zh = languageRef.current !== 'en';
    switch (kind) {
      case 'camera': return zh ? '已切换到摄像头' : 'Camera feed started';
      case 'screen': return zh ? '已开始屏幕共享' : 'Screen share started';
      case 'file': return zh ? `开始播放：${name ?? '视频文件'}` : `Now streaming: ${name ?? 'video file'}`;
      case 'image': return (zh ? `已分享图片：${name ?? ''}` : `Image shared: ${name ?? ''}`).trim();
      // 'none' with a name = a video that PLAYED TO ITS END (same chip family
      // as the start announcement); bare 'none' = the feed was switched off
      default: return name
        ? (zh ? `播放结束：${name}` : `Video ended: ${name}`)
        : (zh ? '视频源已关闭' : 'Video feed off');
    }
  };

  // Mid-session source announcements (`input.video.source`): one per SEGMENT —
  // a repeated live kind (camera flip front↔rear) stays quiet, while every
  // file/image pick is a new segment. The server records the segment, buffers
  // a timeline note for the model's next turn, and journals the dedicated
  // transcript bubble. The sampler clock must already sit on the new basis
  // when this fires (setSamplerClock BEFORE announceSource).
  const lastAnnouncedSourceRef = useRef<'none' | 'camera' | 'screen' | 'file' | 'image'>('none');
  // true once any source change happened DURING the session — the file
  // onEnded handler keeps multi-source sessions alive (a pure file session
  // keeps its classic auto-stop)
  const sourceChangedMidSessionRef = useRef(false);
  const announceSource = (
    kind: 'none' | 'camera' | 'screen' | 'file' | 'image',
    extra?: { name?: string; durationS?: number },
  ) => {
    if (!sessionConnectedRef.current) return;
    if (lastAnnouncedSourceRef.current === kind && kind !== 'file' && kind !== 'image') return;
    lastAnnouncedSourceRef.current = kind;
    sourceChangedMidSessionRef.current = true;
    session.sendSourceChange({ kind, ...extra });
  };

  // A FINISHED session's transcript stays on screen (we don't continue live
  // sessions) until the NEXT pre-session action opens a new one — then it
  // flushes. This ref marks "the log belongs to an ended session"; the edge
  // detector below sets it when streamConnected falls with a non-empty log,
  // and flushEndedLiveLog (called by every pre-session action) clears it.
  const liveLogEndedRef = useRef(false);
  const streamConversationRef = useRef(streamConversation);
  streamConversationRef.current = streamConversation;
  const prevStreamConnectedRef = useRef(streamConnected);
  useEffect(() => {
    const was = prevStreamConnectedRef.current;
    prevStreamConnectedRef.current = streamConnected;
    // true→false with turns still shown = a session just ended (hang-up, video
    // end, scripted stop, server drop). New Chat empties the log in the same
    // commit, so the length guard keeps this from arming on a reset.
    if (was && !streamConnected && streamConversationRef.current.length > 0) {
      liveLogEndedRef.current = true;
    }
  }, [streamConnected]);
  // A session's end unloads the media source — every new call starts from
  // the clean no-media state. Covers the non-button end paths too (server drop,
  // session gone/superseded); the hang-up button also calls these directly.
  const prevRealConnectedRef = useRef(session.connected);
  useEffect(() => {
    const was = prevRealConnectedRef.current;
    prevRealConnectedRef.current = session.connected;
    if (was && !session.connected) {
      unloadMediaFile();
      stopWebMedia();
      setVideoEnabled(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session.connected]);
  /** Flush a finished session's transcript when the next pre-session action
   *  begins (new query, new file, new camera/screen feed, direct start). A
   *  no-op while building the upcoming session, so pre-connect queued queries
   *  still accumulate. */
  const flushEndedLiveLog = () => {
    if (!liveLogEndedRef.current) return;
    liveLogEndedRef.current = false;
    setStreamConversation([]);
    setLiveAsrDraft('');
  };
  // live mirror of the model's "generating" state for the orb RAF loop (whose
  // closure would otherwise read a stale session object)
  const respondingRef = useRef(session.responding);
  respondingRef.current = session.responding;

  // Poll the transcript-title time tag: SESSION duration, wall clock from the
  // connect instant — independent of the media source (a file swap must not
  // reset or repace it). When the stream stops the tag FREEZES at the last
  // tick instead of vanishing — it clears only on a fresh call, a pre-session
  // file pick, or New Chat.
  useEffect(() => {
    if (!streamConnected) return;
    const startedAt = Date.now();
    const read = () => setLiveMediaTime({ kind: 'live', s: (Date.now() - startedAt) / 1000 });
    read();
    const id = setInterval(read, 250);
    return () => clearInterval(id);
  }, [streamConnected]);
  // File-source rewind discipline, one rule for EVERY stop path (hang-up,
  // server-side drop): a MID-stopped stream jumps back to the very start —
  // the file feed only ever re-runs from the top. A FINISHED file (ended →
  // auto-disconnect) keeps its last frame; the next ▶ rewinds at connect
  // anyway, so a post-finish restart still starts anew.
  useEffect(() => {
    if (streamConnected || !mediaFile) return;
    const v = localVideoRef.current;
    if (!v || v.ended) return; // finished → freeze on the last frame
    v.pause();
    try { v.currentTime = 0; } catch { /* not seekable yet */ }
  }, [streamConnected, mediaFile]);

  // stable useCallback identities — safe in effect deps without re-runs
  const { updateConfig: sessionUpdateConfig, getOutputLevel: sessionOutputLevel,
    getGenLevel: sessionGenLevel, pttDown: sessionPttDown, pttUp: sessionPttUp } = session;

  // Push-to-talk hold state (physical press: pointer OR held Space/V). Drives the
  // sonar-wave "held" affordance; `session.listening` stays lit through the
  // post-release ASR decode. pttHeldRef mirrors it so the once-bound key
  // listeners read the live value without re-binding.
  const [pttHeld, setPttHeld] = useState<boolean>(false);
  const pttHeldRef = useRef(false);
  // Pre-session hold-to-dictate (the live mic mirrors in-call PTT): held state
  // drives the same sonar-wave affordance. The begin/end closures live with
  // the dictation stack below; holdMicRef re-points to the fresh ones every
  // render so the once-bound window listeners never act on stale state.
  const [dictHeld, setDictHeld] = useState<boolean>(false);
  const dictHeldRef = useRef(false);
  const holdMicRef = useRef<{ begin: () => boolean; end: () => void }>({
    begin: () => false,
    end: () => {},
  });
  // The unified Input button's tap-vs-hold split, ARM-DELAY style: NOTHING
  // engages until the press has lasted this long — a shorter press is a pure
  // tap (no mic-icon flip, no input.audio.start → no barge-in and no empty
  // voice turn, no dictation warm-up). 400ms sits between Android's ~400ms
  // and iOS's 500ms long-press defaults; Space/V stay instant (deliberate
  // talk keys). Trade-off: a real hold's first ~400ms of audio isn't
  // captured — in practice speech starts later than that after a press.
  const HOLD_ARM_MS = 400;
  const holdArmTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const holdArmedRef = useRef(false);

  const beginPtt = useCallback(() => {
    if (!streamConnected || captureMode !== 'ptt' || micMuted || pttHeldRef.current) return;
    pttHeldRef.current = true;
    setPttHeld(true);
    sessionPttDown();
  }, [streamConnected, captureMode, micMuted, sessionPttDown]);

  const endPtt = useCallback(() => {
    if (!pttHeldRef.current) return;
    pttHeldRef.current = false;
    setPttHeld(false);
    sessionPttUp();
  }, [sessionPttUp]);

  // Hold Space OR V to talk (mouse-free): in-call PTT, or pre-session dictation
  // on the live page — one hold gesture either way. Bound ONCE on window (the
  // handlers reach the current closures through holdMicRef) so a release
  // anywhere commits the turn. Ignored while typing in a field, on key
  // auto-repeat, and when a MODIFIER is held (so Ctrl/⌘+V paste, ⌘/Ctrl+Space
  // input-switch/Spotlight etc. never arm it). One key OWNS the hold at a time
  // (activeHoldKey), so bumping the other key can't end it early. On engage we
  // preventDefault (stops Space page-scroll) AND blur the focused control:
  // Firefox fires a focused button's click on Space even after preventDefault
  // (bugzilla 1552419), and blurring removes the target (also stops Space
  // toggling the PIP <video>). Both keys are conflict-free across
  // Win11/Ubuntu/CentOS/macOS × Chrome/Firefox/Edge (no OS binds plain Space/V).
  useEffect(() => {
    const holdKeyId = (e: KeyboardEvent): 'Space' | 'KeyV' | null => {
      if (e.code === 'Space' || e.key === ' ' || e.key === 'Spacebar') return 'Space';
      if (e.code === 'KeyV' || e.key === 'v' || e.key === 'V') return 'KeyV';
      return null;
    };
    const isTypingTarget = () => {
      const el = document.activeElement as HTMLElement | null;
      return !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable);
    };
    let activeHoldKey: 'Space' | 'KeyV' | null = null;
    const onKeyDown = (e: KeyboardEvent) => {
      const id = holdKeyId(e);
      if (!id || e.repeat || e.ctrlKey || e.metaKey || e.altKey || isTypingTarget()) return;
      if (activeHoldKey) return; // already held by the other key
      if (holdMicRef.current.begin()) {
        activeHoldKey = id;
        e.preventDefault();
        (document.activeElement as HTMLElement | null)?.blur();
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      const id = holdKeyId(e);
      if (!id || id !== activeHoldKey) return; // ignore the non-owning key's release
      activeHoldKey = null;
      if (pttHeldRef.current || dictHeldRef.current) e.preventDefault();
      holdMicRef.current.end();
    };
    const onWinBlur = () => { activeHoldKey = null; }; // a missed keyup mustn't wedge it
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onWinBlur);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onWinBlur);
    };
  }, []);

  // Safety: never leave the mic latched open. A lost window focus, a disconnect,
  // or a capture-mode flip mid-hold ends the turn (or the held dictation).
  useEffect(() => {
    const onBlur = () => holdMicRef.current.end();
    window.addEventListener('blur', onBlur);
    return () => window.removeEventListener('blur', onBlur);
  }, []);
  useEffect(() => {
    if ((!streamConnected || captureMode !== 'ptt') && pttHeldRef.current) endPtt();
  }, [streamConnected, captureMode, endPtt]);

  // Speech and visualizer timing loops
  const requestRef = useRef<number | null>(null);
  const phaseRef = useRef<number>(0);      // slow ring rotation (always)
  const vibPhaseRef = useRef<number>(0);   // standing-wave flip clock; rate ← speech pace
  const loudRef = useRef<number>(0);       // 0 idle → 1 while generating (envelope) → wave amp + flip depth
  const orbIntensityRef = useRef<number>(0.15);
  const targetIntensityRef = useRef<number>(0.15);

  // Liquid Glass orb: slow aurora-blob drift phase,
  // the lerped glass amount (0 = focused free rings, 1 = minimized glass ball),
  // a handle on the CSS halo so the RAF loop can pulse it transform-only, and
  // the shell overlay canvas (normal compositing) whose opacity tracks sphereT.
  const blobPhaseRef = useRef<number>(0);
  const sphereTRef = useRef<number>(0);
  const orbGlowRef = useRef<HTMLDivElement | null>(null);
  const glassShellRef = useRef<HTMLCanvasElement | null>(null);
  // Live ref so the draw loop reads the CURRENT focus without restarting the RAF
  // effect on every focus swap (same pattern as the sidebar listeners below).
  const focusedFeedRef = useRef(focusedFeed);
  focusedFeedRef.current = focusedFeed;

  const sidebarHistoryListRef = useRef<HTMLDivElement | null>(null);
  const modalHistoryListRef = useRef<HTMLDivElement | null>(null);
  const liveLogRef = useRef<HTMLDivElement | null>(null);
  const liveLogStickRef = useRef<boolean>(true); // reading the tail → follow it

  // Handle collapsed popups click
  const handleIconClick = (e: React.MouseEvent<HTMLButtonElement>, type: 'history' | 'demos') => {
    if (!sidebarExpanded) {
      const rect = e.currentTarget.getBoundingClientRect();
      setModalTop(rect.top);
      setActiveModal(activeModal === type ? null : type);
    } else {
      if (type === 'history') setHistoryOpen(!historyOpen);
      if (type === 'demos') setDemosOpen(!demosOpen);
    }
  };

  // Collapsing the rail closes its inline flyouts: History/Demos dropdowns only
  // exist in the EXPANDED sidebar, but their open flags used to survive the
  // collapse — leaving both icons toggled-on (purple) in the collapsed rail
  // (whose clicks go through the popup-modal path instead).
  useEffect(() => {
    if (!sidebarExpanded) {
      setHistoryOpen(false);
      setDemosOpen(false);
    }
  }, [sidebarExpanded]);

  // Refresh the conversation list whenever a history surface opens (sidebar
  // dropdown or collapsed popup) — the backend is local, a fetch per open is cheap.
  useEffect(() => {
    if (historyOpen || activeModal === 'history') void refreshHistory();
  }, [historyOpen, activeModal, refreshHistory]);

  // Live refs so the global listeners always read the CURRENT state (never a stale
  // closure). Refs update synchronously in render, so there's no async-effect window
  // for an open/close to race against.
  const sidebarExpandedRef = useRef(sidebarExpanded);
  sidebarExpandedRef.current = sidebarExpanded;
  const activeModalRef = useRef(activeModal);
  activeModalRef.current = activeModal;

  // Outside-press dismiss + Escape. Keyed to POINTERDOWN, not click (the Radix /
  // Floating UI / native-popover light-dismiss semantic): dismissal depends on
  // where an interaction STARTS, never where it ends. A drag that begins on the
  // history rail (or a text selection that begins in the popup) can therefore
  // release outside without killing the popup — under the old click listener,
  // a cross-element press+release synthesized a click on their common ancestor
  // (outside the popup) and dismissed it. Uses composedPath() (captured at
  // dispatch, before any handler runs) instead of target.closest(), so a press
  // on a control whose icon unmounts mid-interaction (the sidebar toggle /
  // theme swap) is still correctly seen as "inside the sidebar".
  useEffect(() => {
    const inPath = (e: Event, cls: string) =>
      e.composedPath().some((n) => n instanceof HTMLElement && n.classList.contains(cls));

    const handlePointerDownOutside = (e: PointerEvent) => {
      const insideSidebar = inPath(e, 'sidebar-panel');
      if (activeModalRef.current && !inPath(e, 'sidebar-popup-modal') && !insideSidebar) {
        setActiveModal(null);
      }
      if (sidebarExpandedRef.current && !insideSidebar) {
        setSidebarExpanded(false);
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      const dismissed = activeModalRef.current || sidebarExpandedRef.current;
      if (activeModalRef.current) setActiveModal(null);
      if (sidebarExpandedRef.current) setSidebarExpanded(false);
      if (dismissed) (document.activeElement as HTMLElement | null)?.blur();
    };

    document.addEventListener('pointerdown', handlePointerDownOutside);
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('pointerdown', handlePointerDownOutside);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  // Reset typewriter when language toggles
  useEffect(() => {
    setDisplayedText('');
    setIsDeleting(false);
    setPhraseIndex(0);
  }, [language]);

  // Probe the optional cloud TTS lanes once at mount: each engine option in
  // the TTS group only appears when the backend built that lane (API key at
  // boot) and it answered with a voice list.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(resolveApiUrl('/api/status'));
        const body = await res.json();
        if (cancelled) return;
        const pick = (el: any) =>
          // ready too: a lane whose boot probe failed (bad key, unreachable)
          // must not surface as a selectable-but-broken option
          el && el.ready && Array.isArray(el.voices) && el.voices.length > 0
            ? { ready: true, model: el.model, voices: el.voices }
            : undefined;
        const eleven = pick(body?.voice?.tts_elevenlabs);
        const mm = pick(body?.voice?.tts_minimax);
        if (eleven || mm) {
          setCloudTts({ ...(eleven ? { elevenlabs: eleven } : {}), ...(mm ? { minimax: mm } : {}) });
        }
      } catch { /* backend without the lanes — the UI stays local-only */ }
    })();
    return () => { cancelled = true; };
  }, []);

  // Typewriter effect loop
  useEffect(() => {
    if (dialogueStarted) return;

    const phrases = language === 'en'
      ? ["Hello, I'm MOSS-VL!", "How can I help you?"]
      : ["你好，我是 MOSS-VL！", "我能帮您做些什么？"];

    const currentPhrase = phrases[phraseIndex];
    let timer: any;

    if (!isDeleting && displayedText === currentPhrase) {
      // Pause at full text
      timer = setTimeout(() => {
        setIsDeleting(true);
      }, 2000);
    } else if (isDeleting && displayedText === '') {
      // Pause when cleared, switch to next phrase
      timer = setTimeout(() => {
        setIsDeleting(false);
        setPhraseIndex((prev) => (prev + 1) % phrases.length);
      }, 600);
    } else {
      // Backspacing stays steady and fast (45ms). Typing is faster now than
      // before but still comfortably slower than backspace, keeping the cadence.
      let speed = isDeleting ? 45 : 55 + Math.random() * 40;

      // Introduce natural stops on space or punctuation
      if (!isDeleting && displayedText.length > 0) {
        const lastChar = currentPhrase[displayedText.length - 1];
        if (lastChar === ',' || lastChar === ' ') {
          speed += 150;
        }
      }

      timer = setTimeout(() => {
        setDisplayedText((prev) => {
          if (isDeleting) {
            return currentPhrase.substring(0, prev.length - 1);
          } else {
            return currentPhrase.substring(0, prev.length + 1);
          }
        });
      }, speed);
    }

    return () => clearTimeout(timer);
  }, [displayedText, isDeleting, phraseIndex, dialogueStarted, language]);

  // ==================== UTILS ====================
  function getFormattedTime() {
    return new Date().toTimeString().split(' ')[0];
  }

  // Bubble stamp: prefer the server's generation time (epoch seconds) — the
  // browser clock only marks arrival, which lags and bunches when finalize
  // bursts / sentence splits land several bubbles in one render tick.
  function formatEventTime(epochSeconds?: number) {
    return typeof epochSeconds === 'number' && epochSeconds > 0
      ? new Date(epochSeconds * 1000).toTimeString().split(' ')[0]
      : getFormattedTime();
  }

  // seconds → m:ss (or h:mm:ss for long files)
  function formatMediaTime(seconds: number): string {
    const s = Math.floor(seconds % 60);
    const m = Math.floor(seconds / 60) % 60;
    const h = Math.floor(seconds / 3600);
    const mm = h > 0 ? String(m).padStart(2, '0') : String(m);
    return (h > 0 ? `${h}:` : '') + `${mm}:${String(s).padStart(2, '0')}`;
  }

  // Diagnostics channel (the panel shows the conversation transcript, not these)
  function addLog(text: string, type?: string) {
    console.info(`[MOSS ${getFormattedTime()}]${type ? ` (${type})` : ''} ${text}`);
  }

  // Language Switch
  const toggleLanguage = () => {
    const nextLang = language === 'en' ? 'zh' : 'en';
    setLanguage(nextLang);
    addLog(nextLang === 'en' ? 'Language switched to English.' : '系统语言切换为中文。');
    
    // Translate welcome message if it is the only message
    if (chatMessages.length === 1 && chatMessages[0].id === 'welcome') {
      setChatMessages([
        {
          id: 'welcome',
          sender: 'ai',
          text: nextLang === 'en' 
            ? 'Welcome to MOSS Realtime Experience (Offline Chat Mode). Try asking for system performance logs, sound waveforms, or CSS layouts.' 
            : '欢迎来到 MOSS 实时体验（离线对话模式）。您可以尝试索取系统运行指标、音频波形控制，或玻璃微态 CSS 布局。',
        }
      ]);
    }
  };

  // ==================== THEME & BODY SYNC ====================
  useEffect(() => {
    document.body.className = '';
    if (theme === 'light') {
      document.body.classList.add('light-theme');
    }
    if (isStreamingMode) {
      document.body.classList.add('mode-streaming');
    }
    // light-dark() tokens resolve through the USED color scheme, which follows
    // the OS preference by default (:root { color-scheme: dark light }) — so a
    // light-OS device silently overrode the in-app dark toggle (and vice versa),
    // most visibly on phones. Pin the used scheme to the app's own theme — on
    // BOTH roots: some Chromium builds don't re-resolve light-dark() down the
    // inheritance chain when only the html-level inline scheme changes.
    document.documentElement.style.colorScheme = theme;
    document.body.style.colorScheme = theme;
  }, [theme, isStreamingMode]);

  // Auto-scroll chat window
  useEffect(() => {
    chatMessagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  // Keep the live transcript pinned to the newest line while sentence bubbles
  // stream in — but only while the user is reading the tail (a deliberate
  // scroll-up must not be fought per sentence). The stick flag is sampled in
  // onScroll (i.e., BEFORE new content lands); the programmatic pin below
  // re-fires onScroll at distance 0, so it re-arms.
  // Pin via DOM observation, not state deps: ANY growth past the log's bottom
  // edge — a bubble mounting, the pending caption revealing character by
  // character, text re-wrapping on a panel resize, the 更多设置 fold squeezing
  // the box — lands the tail back in view, as long as the reader was following
  // it (liveLogStickRef; scrolling up still releases the pin, and returning to
  // the bottom re-arms it via onScroll). MutationObserver catches content
  // changes, ResizeObserver the box itself; both fire before paint, so the
  // pin never flashes an unscrolled frame.
  useEffect(() => {
    const el = liveLogRef.current;
    if (!el) return;
    const pin = () => {
      if (liveLogStickRef.current) el.scrollTop = el.scrollHeight;
    };
    const mo = new MutationObserver(pin);
    mo.observe(el, { childList: true, subtree: true, characterData: true });
    const ro = new ResizeObserver(pin);
    ro.observe(el);
    pin();
    return () => {
      mo.disconnect();
      ro.disconnect();
    };
  }, []);

  // ==================== SPEECH DICTATION (STT) ====================
  // Primary path: the BACKEND FunASR — mic → 16 kHz PCM (the same worklet the
  // live page uses) → POST /api/asr every second with the whole buffer → the
  // rolling hypothesis fills the input box in realtime; stopping runs one
  // final decode and leaves the text for the user to edit and send (dictation
  // never auto-submits). The browser Web Speech API is only a fallback (it
  // needs vendor servers, which this deployment usually can't reach).
  interface DictationSession {
    mic: MicCapture | null;
    stream: MediaStream;
    chunks: Uint8Array[];
    timer: ReturnType<typeof setInterval> | null;
    seq: number; // decode serial — only the newest hypothesis may win
  }
  const dictationRef = useRef<DictationSession | null>(null);
  const dictationModeRef = useRef<'backend' | 'webspeech' | null>(null);
  // true once the async startBackendDictation() resolves (capture is actually
  // live). Lets a release that beats the warm-up defer the stop to the moment
  // the capture is ready, so the utterance commits instead of being dropped.
  const dictationReadyRef = useRef(false);
  // Where dictated text lands: the chat composer (chat mode) or the live
  // transcript's self-refining asr-pending bubble (pre-connect voice
  // queries). Set by the mic button that started the dictation.
  const dictationSinkRef = useRef<(text: string) => void>(() => {});
  // Fired ONCE when a dictation fully ends (final decode delivered) — the
  // live target commits the bubble to the queue here; chat has no commit
  // (its text stays in the composer for review).
  const dictationDoneRef = useRef<(() => void) | null>(null);
  const fireDictationDone = () => {
    dictationDoneRef.current?.();
    dictationDoneRef.current = null;
  };

  const dictationDecode = async (d: DictationSession, final: boolean) => {
    const total = d.chunks.reduce((n, c) => n + c.byteLength, 0);
    if (total < 6400) return; // < 0.2 s — nothing worth decoding yet
    const seq = ++d.seq;
    try {
      const res = await fetch(resolveApiUrl('/api/asr'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: new Blob(d.chunks as BlobPart[]),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const body = (await res.json()) as { text: string };
      // a stale rolling decode must not overwrite a newer hypothesis
      if (body.text && (final || seq === d.seq)) dictationSinkRef.current(body.text);
    } catch (e) {
      if (final) {
        console.warn('dictation final decode failed:', e);
        addLog(language === 'en' ? 'Dictation decode failed.' : '语音识别失败。', 'red');
      }
    }
  };

  const startBackendDictation = async (): Promise<boolean> => {
    // only try when the backend ASR is actually up (local fetch — cheap)
    try {
      const res = await fetch(resolveApiUrl('/api/status'));
      const status = (await res.json()) as { voice?: { asr?: { ready?: boolean } } };
      if (!status.voice?.asr?.ready) return false;
    } catch {
      return false;
    }
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      addLog(language === 'en' ? 'Microphone unavailable.' : '麦克风不可用。', 'red');
      return true; // mic denied — Web Speech would fail the same way, don't cascade
    }
    const d: DictationSession = { mic: null, stream, chunks: [], timer: null, seq: 0 };
    dictationRef.current = d;
    dictationModeRef.current = 'backend';
    try {
      d.mic = await startMicCapture(stream, (pcm) => d.chunks.push(new Uint8Array(pcm)));
    } catch (e) {
      console.warn('mic worklet failed:', e);
      stream.getTracks().forEach((t) => t.stop());
      dictationRef.current = null;
      dictationModeRef.current = null;
      return false;
    }
    setIsDictating(true);
    dictationSinkRef.current('');
    d.timer = setInterval(() => void dictationDecode(d, false), 1000);
    return true;
  };

  const stopBackendDictation = () => {
    const d = dictationRef.current;
    dictationRef.current = null;
    dictationModeRef.current = null;
    setIsDictating(false);
    if (!d) {
      fireDictationDone();
      return;
    }
    if (d.timer) clearInterval(d.timer);
    void d.mic?.stop();
    d.stream.getTracks().forEach((t) => t.stop());
    // final decode into the sink, THEN the commit (live target turns the
    // refined bubble into a queued query; chat keeps the composer text)
    void dictationDecode(d, true).finally(fireDictationDone);
  };

  const startSpeechRecognition = () => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      // Fallback if Speech API is unsupported (Mock recording)
      setIsDictating(true);
      dictationSinkRef.current(language === 'en' ? 'Simulating speech input (3s)...' : '正在模拟语音输入 (3秒)...');

      setTimeout(() => {
        setIsDictating(false);
        dictationSinkRef.current(language === 'en' ? 'Show me the performance charts' : '显示运行性能图表');
        fireDictationDone();
      }, 3000);
      return;
    }

    try {
      const recog = new SpeechRecognition();
      recog.continuous = false;
      recog.interimResults = true; // realtime hypothesis into the input box
      recog.lang = language === 'en' ? 'en-US' : 'zh-CN';

      recog.onstart = () => {
        setIsDictating(true);
        dictationSinkRef.current('');
      };

      recog.onresult = (event: any) => {
        // interim + final fragments concatenated = the rolling hypothesis;
        // the text stays in the input for review — no auto-submit
        const text = Array.from(event.results as ArrayLike<any>)
          .map((r: any) => r[0]?.transcript ?? '')
          .join('');
        if (text) dictationSinkRef.current(text);
      };

      recog.onerror = (err: any) => {
        console.error('Speech recognition error:', err);
        setIsDictating(false);
      };

      recog.onend = () => {
        // fires on natural end-of-utterance too (continuous=false), so each
        // spoken query commits its own bubble without a stop click
        setIsDictating(false);
        dictationModeRef.current = null;
        fireDictationDone();
      };

      recognitionRef.current = recog;
      dictationModeRef.current = 'webspeech';
      recog.start();
    } catch (e) {
      console.error(e);
      setIsDictating(false);
      fireDictationDone();
    }
  };

  const stopSpeechRecognition = () => {
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop(); // onend fires the commit
      } catch (e) {}
    } else {
      fireDictationDone();
    }
    setIsDictating(false);
    dictationModeRef.current = null;
  };

  /** Toggle dictation into the chat composer (review-then-send). The live
   *  page's pre-session mic is hold-to-dictate instead — beginLiveDictation
   *  below — mirroring the in-call PTT gesture. */
  const handleMicClick = () => {
    if (isDictating) {
      if (dictationModeRef.current === 'backend') stopBackendDictation();
      else stopSpeechRecognition();
    } else {
      dictationSinkRef.current = setChatInput;
      dictationDoneRef.current = null;
      void startBackendDictation().then((handled) => {
        if (!handled) startSpeechRecognition();
      });
    }
  };

  // ==================== WEBCAM & AUDIO CAPTURE ====================
  const stopWebMedia = () => {
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((track) => track.stop());
      localStreamRef.current = null;
    }
    if (audioCtxRef.current) {
      try {
        audioCtxRef.current.close();
      } catch (e) {}
      audioCtxRef.current = null;
    }
    if (localVideoRef.current) {
      localVideoRef.current.srcObject = null;
    }
    analyserRef.current = null;
    sourceNodeRef.current = null;
    dataArrayRef.current = null;
  };

  /** Mic analyser graph for the level visuals — shared by the camera preview
   *  (startWebMedia) and the screen-share path (mic acquired separately). */
  const attachAudioAnalyser = (stream: MediaStream) => {
    try {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      const ctx = new AudioCtx();
      const analyserNode = ctx.createAnalyser();
      analyserNode.fftSize = 64;

      const sourceNode = ctx.createMediaStreamSource(stream);
      sourceNode.connect(analyserNode);

      audioCtxRef.current = ctx;
      analyserRef.current = analyserNode;
      sourceNodeRef.current = sourceNode;
      dataArrayRef.current = new Uint8Array(analyserNode.frequencyBinCount);

      setMicMuted(false);
      addLog(language === 'en' ? 'Audio analyser active.' : '麦克风频谱采集就绪。');
    } catch (ae) {
      console.error('Audio analyser error:', ae);
    }
  };

  const startWebMedia = async () => {
    addLog(language === 'en' ? 'Attempting device connections...' : '正在尝试链接视音频输入设备...', 'live');
    let videoTrackSuccess = false;
    let audioTrackSuccess = false;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: 'user' },
        audio: true
      });
      localStreamRef.current = stream;
      videoTrackSuccess = true;
      audioTrackSuccess = true;
    } catch (e) {
      console.warn('getUserMedia failed, trying audio-only...', e);
      try {
        const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        localStreamRef.current = audioStream;
        audioTrackSuccess = true;
      } catch (ae) {
        addLog(language === 'en' ? 'Media inputs unavailable. Simulation mode.' : '媒体输入不可用，已进入模拟通道。', 'red');
      }
    }

    const stream = localStreamRef.current;
    if (!stream) return;

    if (videoTrackSuccess && stream.getVideoTracks().length > 0 && localVideoRef.current) {
      localVideoRef.current.srcObject = stream;
      setVideoEnabled(true);
      addLog(language === 'en' ? 'Webcam feed linked.' : '摄像头捕获就绪。');
    } else {
      setVideoEnabled(false);
    }
    // Every session starts on the front camera; offer the flip only where a
    // second camera exists. enumerateDevices() runs post-permission, so the
    // videoinput COUNT is reliable even on iOS (which hides labels/deviceIds).
    setCameraFacing('user');
    navigator.mediaDevices
      .enumerateDevices()
      .then((devs) => setHasMultipleCameras(devs.filter((dv) => dv.kind === 'videoinput').length > 1))
      .catch(() => setHasMultipleCameras(false));

    if (audioTrackSuccess && stream.getAudioTracks().length > 0) {
      attachAudioAnalyser(stream);
    } else {
      setMicMuted(true);
    }
  };

  // Front/rear camera flip (phones/tablets). iOS allows only ONE active camera:
  // the old video track must be stopped and detached BEFORE re-acquiring. The
  // audio track and its analyser graph stay untouched — only video swaps.
  const flipCamera = async () => {
    const stream = localStreamRef.current;
    if (!videoEnabled || !stream) return; // works in pre-connect preview too
    const next = cameraFacing === 'user' ? 'environment' : 'user';
    stream.getVideoTracks().forEach((t) => {
      t.stop();
      stream.removeTrack(t);
    });
    const acquire = async (facing: 'user' | 'environment') => {
      // plain (ideal) facingMode — `exact` throws OverconstrainedError on
      // single-camera devices instead of falling back
      const fresh = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: facing },
      });
      fresh.getVideoTracks().forEach((t) => stream.addTrack(t));
    };
    try {
      await acquire(next);
      setCameraFacing(next);
    } catch (err) {
      console.warn('Camera flip failed, restoring previous camera', err);
      addLog(language === 'en' ? 'Camera switch failed.' : '摄像头切换失败。', 'red');
      try {
        await acquire(cameraFacing);
      } catch {
        setVideoEnabled(false); // video lost entirely; audio session continues
      }
    }
    if (localVideoRef.current) {
      // re-assign to force Safari to pick up the replaced track
      localVideoRef.current.srcObject = null;
      localVideoRef.current.srcObject = stream;
    }
  };

  // Live-page sidebar demo: a scripted mini SESSION — start, the model
  // "speaks" one line word by word, hold, then end. It plays in WHICHEVER
  // focus state is current: model-focused, the words simply appear beneath
  // the big orb and vanish at session end; user-focused, the mini ball also
  // drifts left to make room and drifts back — no forced focus switch.
  const runSubtitleDemo = () => {
    if (streamConnected || subtitleDemoActive) return;
    subtitleDemoTimersRef.current.forEach(clearTimeout);
    subtitleDemoTimersRef.current = [];
    setActiveModal(null);
    setStreamSpeaker('ai');
    setCaptionsText('');
    setSubtitleDemoActive(true); // session start → drift left
    const timers = subtitleDemoTimersRef.current;
    const words = 'Hello! How can I help you?'.split(' ');
    words.forEach((w, i) => {
      timers.push(setTimeout(() => {
        setCaptionsText((prev) => (prev ? prev + ' ' : '') + w);
      }, 800 + i * 280)); // wait for the drift, then type word by word
    });
    timers.push(setTimeout(() => {
      setCaptionsText('');
      setSubtitleDemoActive(false); // session end → drift back to center
    }, 800 + words.length * 280 + 2400));
  };

  // A real connection preempts the demo; also clear timers on unmount.
  useEffect(() => {
    if (streamConnected && subtitleDemoActive) {
      subtitleDemoTimersRef.current.forEach(clearTimeout);
      subtitleDemoTimersRef.current = [];
      setSubtitleDemoActive(false);
    }
  }, [streamConnected, subtitleDemoActive]);
  useEffect(() => () => subtitleDemoTimersRef.current.forEach(clearTimeout), []);

  // FaceTime-style tap-to-hide for the phone live stage. Desktop pointers are
  // excluded at the handler level so hover-mouse behavior never changes.
  const toggleStageChrome = () => {
    if (!window.matchMedia('(pointer: coarse) and (max-width: 767.98px)').matches) return;
    setStageChromeHidden((v) => !v);
  };

  // Chrome always returns when the call state or focus changes — nobody should
  // hunt for invisible controls after a swap or a reconnect.
  useEffect(() => {
    setStageChromeHidden(false);
  }, [focusedFeed, streamConnected]);

  const handleToggleMic = () => {
    if (!streamConnected) return;
    const nextMute = !micMuted;
    setMicMuted(nextMute);
    if (localStreamRef.current) {
      localStreamRef.current.getAudioTracks().forEach((track) => {
        track.enabled = !nextMute;
      });
    }
    session.setMicMuted(nextMute); // also stops forwarding PCM upstream
    addLog(nextMute ? 'Microphone muted.' : 'Microphone stream active.', nextMute ? 'red' : 'live');
  };

  const handleToggleVideo = async () => {
    const nextVideo = !videoEnabled;
    if (!streamConnected) {
      // pre-connect PREVIEW: devices open locally, NOTHING streams — the
      // uplink (mic worklet + frame sampler) only starts inside session.connect
      if (nextVideo) {
        await startWebMedia(); // sets videoEnabled itself on success/failure
      } else {
        stopWebMedia();
        setVideoEnabled(false);
      }
      return;
    }
    setVideoEnabled(nextVideo);
    session.setVideoForwarding(nextVideo); // a disabled track renders black — don't stream it
    if (localStreamRef.current && localStreamRef.current.getVideoTracks().length > 0) {
      localStreamRef.current.getVideoTracks().forEach((track) => {
        track.enabled = nextVideo;
      });
      announceSource(nextVideo ? 'camera' : 'none');
      addLog(nextVideo ? 'Webcam active.' : 'Webcam suspended.');
    } else if (nextVideo) {
      await startWebMedia();
      if (localStreamRef.current?.getVideoTracks().length) announceSource('camera');
    }
  };

  // ---- live video source menu (camera / screen share) ----
  // The camera bead opens a Liquid-Glass menu morphing out of the bead
  // (WWDC25 menu behavior: the source button morphs into its overlay).
  // A screen share is just another video track in the same lane — swap
  // discipline identical to flipCamera, so mid-call switches stream on.

  /** One camera track; null = denied. */
  const acquireCameraTrack = async (): Promise<MediaStreamTrack | null> => {
    try {
      const cam = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: cameraFacing },
      });
      return cam.getVideoTracks()[0] ?? null;
    } catch {
      return null;
    }
  };

  /** Plumb an acquired track into the live media stream — mic merge on a cold
   *  start, in-place video-lane swap otherwise — and flip the UI state. */
  const adoptVideoTrack = async (kind: 'camera' | 'screen', track: MediaStreamTrack) => {
    // connecting a NEW camera/screen feed opens a pre-session → flush a
    // finished session's transcript (no-op mid-session: flag is false then)
    flushEndedLiveLog();
    let stream = localStreamRef.current;
    if (!stream) {
      // no media open yet: mic + video ride ONE stream so a later connect
      // reuses it as the pre-connect preview (a denied mic still leaves
      // typed input, same as video-file mode)
      stream = new MediaStream([track]);
      localStreamRef.current = stream;
      try {
        const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
        mic.getAudioTracks().forEach((t) => stream!.addTrack(t));
        attachAudioAnalyser(stream);
      } catch {
        addLog(language === 'en' ? 'Mic unavailable — use typed input.' : '麦克风不可用 — 可使用文字输入提问。', 'red');
        setMicMuted(true);
      }
    } else {
      // swap the video lane in place (flipCamera discipline) — the mic lane
      // and a live session's uplink never notice the source change
      stream.getVideoTracks().forEach((t) => { t.stop(); stream!.removeTrack(t); });
      stream.addTrack(track);
    }
    const v = localVideoRef.current;
    if (v) {
      // re-assign to force Safari to pick up the replaced track
      v.srcObject = null;
      v.srcObject = stream;
    }
    setVideoSourceKind(kind);
    setVideoEnabled(true);
    if (streamConnected) session.setVideoForwarding(true);
    announceSource(kind); // one segment per KIND — a camera flip stays quiet
    addLog(kind === 'screen'
      ? (language === 'en' ? 'Screen sharing active.' : '屏幕共享推流就绪。')
      : (language === 'en' ? 'Webcam active.' : '摄像头捕获就绪。'), 'live');
  };

  /** Our surface panel picked a type → getDisplayMedia carrying that HINT.
   *  The browser's native confirm still appears (mandatory — a page cannot
   *  enumerate windows), pre-selected to the chosen pane; the Chromium
   *  extras hide this tab and drop full screens where the pick makes them
   *  wrong. monitorTypeSurfaces:'exclude' must never ride together with
   *  displaySurface:'monitor' (spec: contradictory → TypeError). */
  const pickScreenSurface = async (surface: 'monitor' | 'window' | 'browser') => {
    setScreenPickerOpen(false);
    let track: MediaStreamTrack | null = null;
    try {
      const opts = {
        video: { displaySurface: surface },
        audio: false, // the mic stays on its own getUserMedia lane
        selfBrowserSurface: 'exclude', // this tab shows the demo — never share it
        surfaceSwitching: 'include',
        systemAudio: 'exclude',
        ...(surface === 'monitor' ? {} : { monitorTypeSurfaces: 'exclude' }),
      } as DisplayMediaStreamOptions;
      const disp = await navigator.mediaDevices.getDisplayMedia(opts);
      track = disp.getVideoTracks()[0] ?? null;
    } catch {
      track = null; // native confirm dismissed / not permitted
    }
    if (!track) {
      addLog(language === 'en' ? 'Screen share cancelled or unavailable.' : '屏幕共享已取消或不可用。', 'red');
      return;
    }
    armScreenEndedHandler(track);
    await adoptVideoTrack('screen', track);
  };

  /** The browser's own "stop sharing" chip / a closing window ends the track
   *  OUTSIDE our UI — mirror it into app state (an unhandled ended event
   *  leaves a zombie feed: MDN Screen Capture guidance). track.stop() from
   *  our own swaps never fires this, so no unarming is needed. */
  const armScreenEndedHandler = (track: MediaStreamTrack) => {
    track.addEventListener('ended', () => {
      const s = localStreamRef.current;
      if (s) s.removeTrack(track);
      setVideoEnabled(false);
      if (sessionConnectedRef.current) session.setVideoForwarding(false);
      announceSource('none'); // the feed fell to Off under a live session
      addLog(languageRef.current === 'en' ? 'Screen sharing ended.' : '屏幕共享已结束。');
    });
  };

  const selectVideoSource = async (kind: 'camera' | 'screen') => {
    setVideoMenuOpen(false);
    // a staged file/image yields to a live source — mid-session the swap keeps
    // the session (and its mic); only the stage and the sampler clock change.
    // The clock flips to 'live' BEFORE the new source announces, so the new
    // segment's session_ts_start sits on the rebased monotone base.
    if (mediaFileRef.current) {
      const wasFileClock = mediaFileRef.current.kind === 'video';
      stagedImageJpegRef.current = null;
      unloadMediaFile();
      if (sessionConnectedRef.current && wasFileClock) session.setSamplerClock('live');
    }
    // re-picking the ACTIVE source switches the feed off
    if (videoEnabled && videoSourceKind === kind) {
      if (streamConnected && kind === 'screen') {
        // privacy: a stopped share ENDS the display track (the browser's
        // sharing chip goes away) — unlike the camera's cheap enabled=false
        // suspend, which keeps the device warm for an instant re-enable
        const s = localStreamRef.current;
        s?.getVideoTracks().forEach((t) => { t.stop(); s.removeTrack(t); });
        setVideoEnabled(false);
        session.setVideoForwarding(false);
        announceSource('none');
        addLog(language === 'en' ? 'Screen sharing stopped.' : '屏幕共享已停止。');
      } else {
        await handleToggleVideo(); // camera off / pre-connect device release
      }
      return;
    }
    // SCREEN → our own surface panel (which then fires getDisplayMedia with the
    // chosen surface hint); the native confirm still follows.
    if (kind === 'screen') {
      setScreenPickerOpen(true);
      return;
    }
    // camera fresh-on with no media open → the existing preview path (cam+mic)
    if (!localStreamRef.current) {
      flushEndedLiveLog(); // a new camera feed opens a pre-session
      setVideoSourceKind('camera');
      await startWebMedia();
      return;
    }
    // camera swap while a screen/other feed is already open
    const track = await acquireCameraTrack();
    if (!track) {
      addLog(language === 'en' ? 'Camera unavailable.' : '摄像头不可用。', 'red');
      return;
    }
    await adoptVideoTrack('camera', track);
  };

  // the source menu closes on outside press / Escape (Liquid Glass menus are
  // light-dismiss surfaces)
  useEffect(() => {
    if (!videoMenuOpen) return;
    const onDown = (e: PointerEvent) => {
      const anchor = videoMenuAnchorRef.current;
      if (anchor && !anchor.contains(e.target as Node)) setVideoMenuOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setVideoMenuOpen(false);
    };
    document.addEventListener('pointerdown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('pointerdown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [videoMenuOpen]);

  const dismissScreenPicker = () => {
    setScreenPickerOpen(false);
  };
  useEffect(() => {
    if (!screenPickerOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') dismissScreenPicker();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [screenPickerOpen]);

  // ---- media-file (video / image) streaming mode ----

  /** One staged still image, pre-encoded to the sampler's JPEG shape at pick
   *  time, so connect-time and mid-session sends reuse the same bytes. */
  const stagedImageJpegRef = useRef<ArrayBuffer | null>(null);
  const encodeImageToJpeg = async (file: File): Promise<ArrayBuffer | null> => {
    try {
      const bmp = await createImageBitmap(file);
      const scale = Math.min(1, 512 / Math.max(bmp.width, bmp.height)); // sampler parity
      const canvas = document.createElement('canvas');
      canvas.width = Math.max(2, Math.round(bmp.width * scale));
      canvas.height = Math.max(2, Math.round(bmp.height * scale));
      const ctx = canvas.getContext('2d');
      if (!ctx) return null;
      ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
      bmp.close();
      const blob = await new Promise<Blob | null>((res) => canvas.toBlob(res, 'image/jpeg', 0.8));
      return blob ? await blob.arrayBuffer() : null;
    } catch {
      return null; // undecodable (e.g. HEIC outside Safari)
    }
  };

  /** The staged image goes up as ONE frame on the session clock. */
  const sendStagedImage = () => {
    const jpeg = stagedImageJpegRef.current;
    if (jpeg && sessionConnectedRef.current) session.sendImageFrame(jpeg);
  };

  const loadMediaFile = async (file: File) => {
    const isImage = (file.type || '').startsWith('image/');
    const live = sessionConnectedRef.current;
    // an image must decode BEFORE any teardown — an unsupported pick changes nothing
    let imageJpeg: ArrayBuffer | null = null;
    if (isImage) {
      imageJpeg = await encodeImageToJpeg(file);
      if (!imageJpeg) {
        addLog(language === 'en' ? 'Unsupported image format — try JPEG/PNG.' : '图片格式不支持 — 请尝试 JPEG/PNG。', 'red');
        return;
      }
    }
    if (live) {
      // mid-session swap: stop only the VIDEO lane — the mic (and a live
      // session's uplink) must never notice a source change
      const s = localStreamRef.current;
      s?.getVideoTracks().forEach((t) => { t.stop(); s.removeTrack(t); });
      session.setVideoForwarding(false); // pause frames while the stage swaps
    } else {
      stopWebMedia(); // release any camera preview
    }
    setVideoEnabled(false);
    if (mediaFile) URL.revokeObjectURL(mediaFile.url);
    if (!live) {
      setLiveMediaTime(null); // a fresh pick clears an ENDED session's frozen tag
      // a fresh clip opens a new pre-session: a finished session's transcript
      // flushes; queued pre-connect queries STAY (dashed bubbles are future
      // input — they ride the next connect as the opener)
      flushEndedLiveLog();
      setStreamConversation((prev) => prev.filter((e) => e.queued));
      setCaptionsText(
        language === 'en'
          ? 'Ready. Click the green call button to start a video chat.'
          : '已就绪，等待开启视频通话。点击绿色电话图标开始。',
      );
    }

    if (isImage) {
      stagedImageJpegRef.current = imageJpeg;
      const url = URL.createObjectURL(file);
      setMediaFile({ kind: 'image', name: file.name, url, size: file.size, mime: file.type || 'image/*' });
      // no CAS upload for images (v1: replay has no image lane); settle any
      // video upload UI left behind by a previous pick
      liveUploadXhrRef.current?.abort();
      liveUploadXhrRef.current = null;
      setLiveUploadPct(null);
      videoUploadRef.current = null;
      const v = localVideoRef.current;
      if (v) {
        v.pause();
        v.srcObject = null;
        v.removeAttribute('src');
        v.load();
      }
      setFocusedFeed('user');
      if (live) {
        session.setSamplerClock('live'); // a still sits on the live clock
        announceSource('image', { name: file.name });
        sendStagedImage();
      }
      addLog((language === 'en' ? 'Image loaded: ' : '图片已加载：') + file.name, 'live');
      return;
    }

    stagedImageJpegRef.current = null;
    const url = URL.createObjectURL(file);
    setMediaFile({ kind: 'video', name: file.name, url, size: file.size, mime: file.type || 'video/*' });
    // upload to the CAS in the background; once a session connects, the handle
    // is attached so the recorded history can replay the video (dedup = a
    // re-pick of the same file is a no-op server-side)
    liveUploadXhrRef.current?.abort(); // a re-pick abandons the previous transfer
    setLiveUploadPct(0);
    videoUploadRef.current = {
      key: url,
      promise: uploadMedia(file, file.name, {
        // 0..1 drives the PIP puck + file bead ring; all bytes sent while the
        // server still extracts the poster → 'processing' (indeterminate spin)
        onProgress: (p) => setLiveUploadPct(p >= 1 ? 'processing' : p),
        getXhr: (xhr) => { liveUploadXhrRef.current = xhr; },
      })
        .then((desc) => {
          setLiveUploadPct(null);
          return desc.hash;
        })
        .catch((e) => {
          // aborts settle on a microtask AFTER a re-pick has already armed the
          // next upload's indicator — the canceller owns the pct state, not us
          if (e instanceof Error && e.message === 'upload cancelled') return null;
          setLiveUploadPct(null);
          console.warn('video upload failed:', e);
          addLog(
            language === 'en'
              ? 'Video upload failed — this session\'s replay will have no video.'
              : '视频上传失败 — 本次会话的历史回放将不包含视频。',
            'red',
          );
          return null;
        }),
    };
    const v = localVideoRef.current;
    if (v) {
      v.srcObject = null;
      v.src = url; // pre-session: paused on the first frame until ▶
      v.muted = true;
      v.loop = false;
      v.load();
      if (live) {
        // mid-session swap: flip the clock to media time, announce the segment
        // (duration rides along once metadata lands), then play from the top —
        // the ordering keeps session_ts_start equal to the new segment's base
        const announceAndPlay = () => {
          if (!sessionConnectedRef.current) return;
          session.setSamplerClock('media');
          announceSource('file', {
            name: file.name,
            ...(Number.isFinite(v.duration) && v.duration > 0 ? { durationS: v.duration } : {}),
          });
          session.setVideoForwarding(true);
          void v.play().catch(() => undefined);
          armVideoAttach(url, file.name); // link the CAS blob to history once the upload lands
        };
        if (v.readyState >= 1) announceAndPlay();
        else v.addEventListener('loadedmetadata', announceAndPlay, { once: true });
      }
    }
    setFocusedFeed('user'); // the video IS the show (explicit user action)
    addLog((language === 'en' ? 'Video file loaded: ' : '视频文件已加载：') + file.name, 'live');
  };

  /** Link the uploaded source video (CAS handle) to THIS session's history —
   *  fires once the background upload lands. `key`/`name` pin the pick the
   *  upload belongs to (a re-pick abandons it); a session that ended before
   *  the upload finished attaches nothing. */
  const armVideoAttach = (key: string, name: string) => {
    const upload = videoUploadRef.current;
    if (!upload || upload.key !== key) return;
    void upload.promise.then((hash) => {
      if (!hash || videoUploadRef.current?.key !== upload.key) return;
      if (!sessionConnectedRef.current) return;
      const v = localVideoRef.current;
      session.attachVideo({
        media: hash,
        name: name || undefined,
        durationS: v && Number.isFinite(v.duration) && v.duration > 0 ? v.duration : undefined,
      });
    });
  };

  /** Unload the picked file (no guards — callers decide when it's legal). */
  const unloadMediaFile = () => {
    const f = mediaFileRef.current;
    if (!f) return;
    const v = localVideoRef.current;
    if (v) {
      v.pause();
      v.removeAttribute('src');
      v.load();
    }
    URL.revokeObjectURL(f.url);
    setMediaFile(null);
    stagedImageJpegRef.current = null;
    liveUploadXhrRef.current?.abort(); // stop a still-running transfer with it
    liveUploadXhrRef.current = null;
    setLiveUploadPct(null);
    videoUploadRef.current = null;
    setPipInfoOpen(false);
  };

  const exitVideoFileMode = () => {
    if (streamConnected || !mediaFile) return;
    unloadMediaFile();
    addLog(language === 'en' ? 'Back to camera mode.' : '已退出视频推流模式。');
  };

  /** Info rows for the loaded video file (element metadata + file facts). */
  const pipInfoRows = (): Array<[string, string]> => {
    const zh = language !== 'en';
    const v = localVideoRef.current;
    const f = mediaFileRef.current;
    if (!f) return [];
    const rows: Array<[string, string]> = [
      [zh ? '文件名' : 'Name', f.name.length > 24 ? f.name.slice(0, 21) + '…' : f.name],
      ['MIME', f.mime],
      [zh ? '大小' : 'Bytes', fmtBytes(f.size)],
    ];
    if (v && v.videoWidth) rows.push([zh ? '尺寸' : 'Size', `${v.videoWidth} × ${v.videoHeight}`]);
    if (v && Number.isFinite(v.duration) && v.duration > 0) {
      const s = Math.round(v.duration);
      rows.push([zh ? '时长' : 'Duration', `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`]);
    }
    return rows;
  };

  const pipInfoZoneEnter = () => {
    if (pipInfoHideTimer.current) clearTimeout(pipInfoHideTimer.current);
    pipInfoHideTimer.current = null;
    setPipInfoOpen(true);
  };

  const pipInfoZoneLeave = () => {
    if (pipInfoHideTimer.current) clearTimeout(pipInfoHideTimer.current);
    pipInfoHideTimer.current = setTimeout(() => setPipInfoOpen(false), 200);
  };

  // a server-side session end (grace/error) must also halt file playback
  useEffect(() => {
    if (!streamConnected && mediaFileRef.current) localVideoRef.current?.pause();
  }, [streamConnected]);

  // Switching OFF a live call: KEEP a keyboard draft (and its open pill) so the
  // half-typed question survives; with no draft, close the pill. Voice and
  // camera RESET — the camera closes fully (feed + view + button; the stream is
  // stopped by the hangup's stopWebMedia). The uploaded video file stays loaded.
  const resetLiveControls = () => {
    // keyboard — keep a non-empty draft (pill stays open); else toggle it off
    if (!liveTextDraft.trim()) setLiveTextOpen(false);
    // voice — mic UI back to default (unmuted, not held)
    setMicMuted(false);
    setPttHeld(false);
    pttHeldRef.current = false;
    // camera — off: button + view (stream already stopped by stopWebMedia)
    setVideoEnabled(false);
    setCameraFacing('user');
  };

  const handleConnectStream = async () => {
    if (session.connecting) return;
    // a still-held pre-session dictation ends NOW: its utterance commits and
    // rides the call as (part of) the opening request — real sessions flush it
    // at session.created, a scripted call swallows it via noteLiveSend —
    // instead of the capture leaking past connect with no way to stop it
    endLiveDictation();
    // a fresh call clears the frozen media-time tag; a hang-up keeps it
    // (frozen where the stream stopped)
    if (!streamConnected) setLiveMediaTime(null);

    if (!streamConnected) {
      // No source-ask gate: a session is legal with NO video source at all —
      // feeds can be added/swapped any time via the Media control (the model
      // accepts text/voice-only turns; frames simply never flow until then).
      setVideoMenuOpen(false);
      addLog(language === 'en' ? 'Opening realtime session...' : '正在建立实时会话通道...', 'glow');
      // a pre-connect preview (camera or screen) already holds the devices — reuse them
      const hadPreview = !!localStreamRef.current;
      if (!hadPreview) {
        // file / image / source-less start: mic only (voice questions); the
        // frames come from the file or nowhere — a denied mic still leaves
        // typed input
        try {
          localStreamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
          setMicMuted(false);
        } catch {
          addLog(language === 'en' ? 'Mic unavailable — use typed input.' : '麦克风不可用 — 可使用文字输入提问。', 'red');
        }
      }
      try {
        const attached = await session.connect({
          stream: localStreamRef.current,
          videoEl: localVideoRef.current,
          config: {
            asrLanguage,
            vadSensitivity,
            captureMode,
            ttsVoice: voicePreset,
            ttsEngine, // creation-time engine pick (local pool | elevenlabs)
            speakingRate: voiceRate,
            systemPrompt, // KV-prefilled at creation; '' → server default
            videoSource: activeSource, // the INITIAL source (history-facing); mid-session switches ride input.video.source
            frameFps: parseParam(streamFps, 0.5, 10, 1), // client sampler cadence
            temperature: parseParam(temperature, 0, 2, 0.7), // creation-time sampling
            topP: parseParam(topP, 0, 1, 0.8),
            topK: Math.round(parseParam(topK, 1, 100, 20)),
            maxTokensPerTurn: Math.round(parseParam(maxTokensRate, 1, 500, 4)), // backend generation rate cap (tokens/s); reveal is immediate, so this IS the visible text speed
          },
          initialClock: mediaFile?.kind === 'video' ? 'media' : 'live', // a still image sits on the live clock
        });
        if (!attached) return;
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        if (!hadPreview) stopWebMedia(); // keep a user-opened preview alive
        addLog(language === 'en' ? 'Connection failed: ' + message : '连接失败：' + message, 'red');
        // pin the failure in the transcript as a system chip: the session.error
        // effect re-stamps the caption with this exact '⚠' text, so the dedup
        // check below hides the (transient) caption twin — the chip is what
        // stays visible until the next call
        const failureText = `⚠ ${message}`;
        setStreamConversation((prev) => [
          ...prev,
          { sender: 'ai', system: true, text: failureText, time: getFormattedTime() },
        ]);
        setCaptionsText(failureText);
        return;
      }
      addLog(language === 'en' ? 'Realtime stream active.' : '实时连线建立成功。', 'live');
      setCaptionsText(''); // captions appear with the first real turn
      // Transcript reset + pre-connect queued-query flush BOTH happen in the
      // session.connected effect — deliberately NOT here: session.created can
      // land while connect() is still awaiting mic setup, so a reset after
      // connect() would wipe the server echo of the flushed queries (the
      // swallowed-queries bug). Clearing inside that effect, right before the
      // send, keeps clear → send → echo-append ordered.
      // (the initial source announcement + a staged image's single frame ride
      // the session.connected effect — sends before session.created would be
      // dropped on the not-yet-open socket)
      if (mediaFile?.kind === 'video' && localVideoRef.current) {
        // ▶ pressed: run the file from the top; the media-time gate paces frames
        localVideoRef.current.currentTime = 0;
        void localVideoRef.current.play().catch(() =>
          addLog(language === 'en' ? 'Video playback failed.' : '视频播放失败。', 'red'));
      }
      // link the uploaded source video to THIS session's history (fires when
      // the background upload lands; a failed upload already logged above)
      if (mediaFile?.kind === 'video') armVideoAttach(mediaFile.url, mediaFile.name);
      // NO focus reset: the view is the USER'S choice — a connect completing
      // right as the model's first turn lands must not yank a pre-selected
      // user view over to the orb. Focus changes only by clicking orb/PIP.
    } else {
      session.disconnect();
      stopWebMedia();                          // close the camera/mic stream + view
      unloadMediaFile();                       // a session end unloads the media source
      resetLiveControls();                     // keep draft · reset voice · close camera
      setCaptionsText(
        language === 'en'
          ? 'Session ended. Click call controls to establish a new streaming connection.'
          : '连线已安全断开。请使用底部电话图标重新发起呼叫。'
      );
      targetIntensityRef.current = 0.15;
      addLog(language === 'en' ? 'Session disconnected. WebSocket closed.' : '会话已断开，WebSocket 传输关闭。', 'red');
    }
  };

  // ---- live-page typed input (the morphing call pill) ----
  const sendLiveText = () => {
    const text = liveTextDraft.trim();
    if (!text) return;
    if (streamConnected) {
      session.sendText(text); // server echoes input.text.done → caption + transcript
    } else {
      // no session yet: queue it — one dashed bubble per query in the log,
      // the whole batch goes out as ONE request at connect. A first query
      // after a finished session flushes that session's transcript first.
      flushEndedLiveLog();
      setPendingLiveQueries((prev) => [...prev, text]);
      setStreamConversation((prev) => [...prev, { sender: 'user', text, time: getFormattedTime(), queued: true }]);
    }
    setLiveTextDraft('');
    liveTextInputRef.current?.focus();
  };

  // A finished pre-connect dictation: the self-refined asr-pending bubble
  // becomes a queued query bubble (one utterance = one bubble). Runs from
  // dictation-stack callbacks armed at mic-click time, so it reads only
  // refs and stable setters; a session that connected mid-utterance gets
  // the text directly instead.
  const commitLiveDictation = () => {
    const text = liveAsrDraftRef.current.trim();
    setLiveAsrDraftBoth('');
    if (!text) return;
    if (sessionConnectedRef.current) {
      session.sendText(text); // echo writes the transcript bubble
      return;
    }
    // a first voice query after a finished session flushes that log first
    flushEndedLiveLog();
    setPendingLiveQueries((prev) => [...prev, text]);
    setStreamConversation((prev) => [...prev, { sender: 'user', text, time: getFormattedTime(), queued: true }]);
  };

  // ---- pre-session hold-to-dictate (the live mic mirrors in-call PTT) ----
  // Press starts a dictation into the transcript's asr-pending bubble; release
  // stops the capture and the final decode commits the utterance as a queued
  // query (the batch goes out as ONE opening request at connect).
  const stopLiveDictationCapture = () => {
    if (dictationModeRef.current === 'backend') stopBackendDictation();
    else stopSpeechRecognition();
  };
  const beginLiveDictation = () => {
    // never stack captures — a running chat-composer dictation keeps its mic
    if (streamConnected || dictHeldRef.current || dictationModeRef.current) return;
    dictHeldRef.current = true;
    dictationReadyRef.current = false;
    setDictHeld(true);
    dictationSinkRef.current = setLiveAsrDraftBoth;
    dictationDoneRef.current = commitLiveDictation;
    void startBackendDictation().then((handled) => {
      if (!handled) startSpeechRecognition();
      dictationReadyRef.current = true;
      // Released DURING warm-up (a quick tap, or a first-time getUserMedia
      // prompt firing pointercancel/blur): the capture only just became live,
      // so stop it the NORMAL way now — final decode + commit — instead of
      // dropping it. A too-quick tap simply captured no audio and commits
      // nothing; a real utterance still reaches the pending-send queue.
      if (!dictHeldRef.current) stopLiveDictationCapture();
    });
  };
  const endLiveDictation = () => {
    if (!dictHeldRef.current) return;
    dictHeldRef.current = false;
    setDictHeld(false);
    // Stop now only if the capture is actually up. If it's still warming up,
    // the warm-up .then above stops it the moment it's ready — stopping
    // mid-warm-up would fire an empty commit and leave an orphaned capture,
    // which is exactly what dropped mic input before it reached pending-send.
    if (dictationReadyRef.current) stopLiveDictationCapture();
  };
  /** New Chat: DROP an in-flight live dictation — no commit, no draft. The
   *  sink/done refs detach first so the pending final decode lands nowhere. */
  const cancelLiveDictation = () => {
    dictHeldRef.current = false;
    setDictHeld(false);
    if (dictationDoneRef.current) { // set ⇔ a live-target dictation is in flight
      dictationDoneRef.current = null;
      dictationSinkRef.current = () => {};
      stopLiveDictationCapture();
    }
    setLiveAsrDraftBoth('');
  };
  /** Shared hold-mic dispatch (mic bead press + held Space): in-call PTT when
   *  a session is live, pre-session dictation on the live page otherwise.
   *  Returns whether the press engaged (keydown preventDefaults on true). */
  const beginHoldMic = (): boolean => {
    if (streamConnected) {
      if (captureMode !== 'ptt') return false;
      beginPtt();
      return true;
    }
    if (!isStreamingMode) return false;
    beginLiveDictation();
    return true;
  };
  const endHoldMic = () => {
    endPtt();
    endLiveDictation();
  };
  // the once-bound window listeners (Space/V / blur) reach the fresh closures here
  holdMicRef.current = { begin: beginHoldMic, end: endHoldMic };

  // autofocus when the field telescopes out (open works pre-connect too:
  // queued queries — so no auto-collapse when a session ends)
  useEffect(() => {
    if (liveTextOpen) liveTextInputRef.current?.focus();
  }, [liveTextOpen]);

  // Pre-connect queue flush — armed on session.connected (i.e. the server's
  // session.created arrived over the OPEN socket; any earlier and sendJSON
  // drops silently). The transcript reset lives HERE (not in handleConnectStream
  // after connect()) on purpose: session.created can land WHILE connect() is
  // still awaiting mic setup, so a reset after connect() would erase the server
  // echo of the queued send that this very effect already fired — the
  // swallowed-queries bug (worst on the camera/screen path, whose extra
  // startWebMedia + second connect shifts the timing so the echo always loses).
  // Clearing before the send keeps the order clear → send → echo-append.
  useEffect(() => {
    if (!session.connected) return;
    // fresh session starts on a clean transcript (drops the pre-connect dashed
    // queued bubbles / any lingering ended-session log)
    setStreamConversation([]);
    setLiveAsrDraft('');
    liveLogEndedRef.current = false;
    // announce the session's INITIAL source segment over the now-open socket
    // (none = quiet; adding a feed later announces it then) and, for a staged
    // still image, send its single frame — both would be dropped silently if
    // fired before session.created (see the effect comment above)
    lastAnnouncedSourceRef.current = activeSource;
    sourceChangedMidSessionRef.current = false;
    if (activeSource !== 'none') {
      const v0 = localVideoRef.current;
      session.sendSourceChange({
        kind: activeSource,
        ...(mediaFile ? { name: mediaFile.name } : {}),
        ...(mediaFile?.kind === 'video' && v0 && Number.isFinite(v0.duration) && v0.duration > 0
          ? { durationS: v0.duration }
          : {}),
      });
    }
    if (mediaFile?.kind === 'image') sendStagedImage();
    const queued = pendingLiveQueriesRef.current;
    if (queued.length === 0) return;
    // the batch goes out as ONE opening request; the server echo writes the
    // merged bubble back into the just-cleared log
    session.sendText(queued.join('\n'));
    setPendingLiveQueries([]);
    addLog(language === 'en'
      ? `Sent ${queued.length} queued ${queued.length === 1 ? 'query' : 'queries'} as one request.`
      : `已将 ${queued.length} 条排队消息合并为一条请求发送。`, 'live');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session.connected]);
  useEffect(() => {
    if (!streamConnected) setLiveAsrDraft(''); // no session → no pending hypothesis
  }, [streamConnected]);
  useEffect(() => {
    if (streamConnected) setMoreSettingsOpen(false); // transcript takes the stage
  }, [streamConnected]);

  // The server can also end the session (grace expiry, takeover, fatal model
  // error) — release the local devices when that happens.
  useEffect(() => {
    if (!streamConnected && localStreamRef.current) {
      stopWebMedia();
      targetIntensityRef.current = 0.15;
    }
  }, [streamConnected]);

  // Surface fatal session errors where the user is looking.
  useEffect(() => {
    if (session.error) {
      setCaptionsText(`⚠ ${session.error}`);
    }
  }, [session.error]);

  // Live-sync the session panel settings over `session.update` (debounced for
  // the sliders). Prompts are creation-time-only; these five are hot-swappable.
  useEffect(() => {
    if (!streamConnected) return;
    const timer = setTimeout(() => {
      sessionUpdateConfig({
        asrLanguage,
        vadSensitivity,
        captureMode,
        ttsVoice: voicePreset,
        speakingRate: voiceRate,
      });
    }, 300);
    return () => clearTimeout(timer);
  }, [asrLanguage, vadSensitivity, captureMode, voicePreset, voiceRate, streamConnected, sessionUpdateConfig]);

  // ==================== VISUALIZER DRAWING LOOP ====================
  // Siri-like Liquid Glass ball. MINIMIZED (user-focus)
  // the orb is a full glass sphere; FOCUSED it stays the free rings over the aurora
  // fill, no shell. Two canvases split the optics:
  //   #streaming-voice-canvas — INTERIOR LIGHT (aurora + rings), rides the page
  //     blend (screen on dark / multiply on light) so it behaves like light/pigment.
  //   #glass-shell-canvas — the SHELL (fresnel rim, lensing whisper, rim arcs,
  //     speculars, caustic) on NORMAL compositing, static per resize/theme.
  // The split is the load-bearing decision: a page-blended shell can only ADD light
  // on dark (washes out over bright video) and only REMOVE it on light (its "glow"
  // degenerated into a naked white ring) — the glass edge must not depend on what
  // the camera happens to show. The RAF loop fades the shell in with sphereT.
  useEffect(() => {
    const canvas = streamingCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Per-theme hue trios (the same colors Wave.draw strokes with) shared by the
    // aurora blobs, the lensing band and the caustic tint.
    const ORB_RGB: Record<string, [number, number, number][]> = {
      fluid: [[138, 75, 230], [74, 222, 128], [56, 189, 248]],
      cosmic: [[230, 95, 30], [244, 63, 94], [234, 179, 8]],
      quantum: [[14, 165, 233], [217, 70, 239], [20, 184, 166]],
    };
    const cols = ORB_RGB[orbTheme] ?? ORB_RGB.fluid;
    const rgba = (c: [number, number, number], a: number) => `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${a})`;
    const isLight = theme === 'light';
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const reduceTransparency = window.matchMedia('(prefers-reduced-transparency: reduce)').matches;

    class Wave {
      color: string;
      speed: number;
      amplitude: number;
      freq: number;

      constructor(color: string, speed: number, amplitude: number, frequencyMultiplier: number) {
        this.color = color;
        this.speed = speed;
        this.amplitude = amplitude;
        this.freq = frequencyMultiplier;
      }

      draw(target: CanvasRenderingContext2D, intensity: number, currentOrbTheme: string, baseRadius: number, dpr: number) {
        target.beginPath();

        let strokeColor = this.color;
        if (currentOrbTheme === 'cosmic') {
          if (this.color.includes('138')) strokeColor = 'rgba(230, 95, 30, 0.45)'; // Amber
          else if (this.color.includes('74')) strokeColor = 'rgba(244, 63, 94, 0.35)'; // Rose
          else strokeColor = 'rgba(234, 179, 8, 0.4)'; // Gold
        } else if (currentOrbTheme === 'quantum') {
          if (this.color.includes('138')) strokeColor = 'rgba(14, 165, 233, 0.45)'; // Cyan
          else if (this.color.includes('74')) strokeColor = 'rgba(217, 70, 239, 0.35)'; // Magenta
          else strokeColor = 'rgba(20, 184, 166, 0.4)'; // Teal
        }

        target.strokeStyle = strokeColor;

        // Refined line: a DPR-px FLOOR keeps the SMALL orb readable; a gentle proportional
        // term keeps the LARGE orb crisp instead of a thick blob.
        const lw = Math.max(1.6 * dpr, baseRadius * 0.024);

        const centerX = target.canvas.width / 2;
        const centerY = target.canvas.height / 2;
        void intensity; // orb energy now comes from the loudness envelope (loudRef)
        // Loudness (0..1) drives BOTH the wave amplitude and the flip depth, so
        // louder speech = larger, deeper vibration; quiet = near-smooth rings.
        const loud = Math.max(0, Math.min(1, loudRef.current));
        const radius = baseRadius; // all three rings share the radius; rings never balloon
        // Standing-wave flip: the ripple amplitude swings +1 → -1 in step (every
        // peak becomes a trough and back, top↔bottom), a REGULAR oscillation. Its
        // DEPTH scales with loudness (0 at rest → wave steady → just rotating).
        const vibDepth = Math.min(1, loud * 1.3);
        const vibSwing = 1 - vibDepth + vibDepth * Math.cos(vibPhaseRef.current * (0.9 + 0.25 * this.amplitude));
        // Small wave (~4× the tiny original floor) growing with loudness.
        const waveAmp = baseRadius * (0.048 + 0.04 * loud) * this.amplitude;
        for (let angle = 0; angle <= Math.PI * 2; angle += 0.04) {
          // slow travelling ripple → the ring rotates
          const rotate = Math.sin(angle * this.freq + phaseRef.current * this.speed);
          const offset = rotate * waveAmp * vibSwing;
          const r = radius + offset;
          const x = centerX + Math.cos(angle) * r;
          const y = centerY + Math.sin(angle) * r;

          if (angle === 0) {
            target.moveTo(x, y);
          } else {
            target.lineTo(x, y);
          }
        }
        target.closePath();
        // Soft gaussian-band stroke (live_page_orb.py L4): two faint halo passes
        // under a crisp core, instead of one hard line — a hard stroke at the mini
        // size collapses into a sharp "colorful ring", the exact anti-glass read.
        // The target is the wave scratch canvas on 'lighten' (per-channel max, the
        // study's np.maximum): passes shape ONE soft band, they never darken each
        // other — the flattened result is page-blended once in render().
        const prevAlpha = target.globalAlpha;
        target.lineWidth = lw * 2.3;
        target.globalAlpha = prevAlpha * 0.10;
        target.stroke();
        target.lineWidth = lw * 1.4;
        target.globalAlpha = prevAlpha * 0.30;
        target.stroke();
        target.lineWidth = lw * 0.8;
        target.globalAlpha = prevAlpha;
        target.stroke();
      }
    }

    const orbWaves = [
      new Wave('rgba(138, 75, 230, 0.45)', 0.12, 1.6, 5),   // Purple
      new Wave('rgba(74, 222, 128, 0.35)', 0.18, 2.0, 7),   // Green
      new Wave('rgba(56, 189, 248, 0.4)', -0.15, 1.4, 4)    // Blue
    ];

    // -------- Static glass shell: L2 fresnel + L5 lensing whisper + L6 rim arcs
    // + L7 speculars + L8 caustic — drawn once per resize/theme straight into the
    // visible #glass-shell-canvas overlay (NORMAL compositing, no page blend, no
    // per-frame drawImage; the RAF loop only writes its opacity = sphereT).
    // Every band is a sampled GAUSSIAN: solid strokes and linear conic tents are
    // what turned the rim into the old hard "candy ring" (live_page_orb.md §6,
    // "Shipped revision 2026-07-02").
    const shellCanvas = glassShellRef.current;
    let shellDirty = true;
    const angDiff = (a: number, b: number) => Math.abs((((a - b + 180) % 360) + 360) % 360 - 180);
    const buildGlassShell = () => {
      if (!shellCanvas) return;
      const w = canvas.width;
      const h = canvas.height;
      if (w === 0 || h === 0) return; // hidden workspace — retry when it has size
      shellCanvas.width = w;
      shellCanvas.height = h;
      const g = shellCanvas.getContext('2d');
      if (!g) return;
      const cx = w / 2;
      const cy = h / 2;
      // Silhouette ≈ container edge: the vessel's backdrop-blur boundary and the
      // fresnel peak now coincide, so the blur circle hides UNDER the rim glow —
      // the old 6% gap ring of naked blurred video ("white/black plate") is gone.
      const R = (w / 2) * 0.99;

      g.clearRect(0, 0, w, h);
      g.save();
      g.beginPath();
      g.arc(cx, cy, R, 0, Math.PI * 2);
      g.clip(); // nothing of the shell may leak outside the silhouette

      if (reduceTransparency) {
        // a11y: the ball reads solid on both themes
        g.fillStyle = isLight ? 'rgba(226, 236, 250, 0.55)' : 'rgba(184, 217, 255, 0.30)';
        g.fillRect(0, 0, w, h);
      }

      // Light theme: limb darkening — real glass dips slightly dark just before
      // the edge glow (painted, not multiplied: the shell no longer rides the
      // page blend, so darkness must be pigment like everything else here).
      if (isLight) {
        const limb = g.createRadialGradient(cx, cy, 0, cx, cy, R);
        const la = (q: number) => 0.20 * Math.pow(1 - Math.sqrt(Math.max(0, 1 - q * q)), 3.0);
        [0, 0.5, 0.7, 0.8, 0.87, 0.92, 0.96, 0.99, 1].forEach((q) => {
          limb.addColorStop(q, `rgba(71, 77, 107, ${la(q).toFixed(3)})`);
        });
        g.fillStyle = limb;
        g.fillRect(0, 0, w, h);
      }

      // L2 — fresnel rim haze, the sky-blue of plain glass: α(q) = 0.025 +
      // peak·(1−√(1−q²))^3.0, sampled densely (canvas gradients interpolate
      // LINEARLY between stops — sparse stops turn this smooth power curve into
      // mach bands that read as extra rings).
      const fresnelPeak = isLight ? 0.32 : 0.38;
      const fresnelAlpha = (q: number) => 0.025 + fresnelPeak * Math.pow(1 - Math.sqrt(Math.max(0, 1 - q * q)), 3.0);
      const fresnelTint = isLight ? '204, 230, 255' : '184, 217, 255';
      const fresnel = g.createRadialGradient(cx, cy, 0, cx, cy, R);
      [0, 0.45, 0.6, 0.7, 0.78, 0.84, 0.89, 0.93, 0.96, 0.98, 0.99, 1].forEach((q) => {
        fresnel.addColorStop(q, `rgba(${fresnelTint}, ${fresnelAlpha(q).toFixed(3)})`);
      });
      g.fillStyle = fresnel;
      g.fillRect(0, 0, w, h);

      // Ring stamp: paint an angular gradient on a scratch canvas, then keep an
      // exact radial gaussian band of it via destination-in — smooth in BOTH axes,
      // no stroke edges, no banding. (Angular profiles live in the conic stops.)
      const ringLayer = (paint: (o: CanvasRenderingContext2D) => void, bandR: number, sigma: number) => {
        const oc = document.createElement('canvas');
        oc.width = w;
        oc.height = h;
        const o = oc.getContext('2d');
        if (!o) return;
        paint(o);
        const ring = o.createRadialGradient(cx, cy, 0, cx, cy, R);
        ring.addColorStop(0, 'rgba(255, 255, 255, 0)');
        [-3.5, -2.8, -2.1, -1.5, -1, -0.6, -0.3, 0, 0.3, 0.6, 1, 1.5, 2.1, 2.8, 3.5].forEach((j) => {
          const q = (bandR + j * sigma) / R;
          if (q <= 0 || q >= 1) return;
          ring.addColorStop(q, `rgba(255, 255, 255, ${Math.exp(-0.5 * j * j).toFixed(4)})`);
        });
        const je = (R - bandR) / sigma; // gaussian tail value at the silhouette
        ring.addColorStop(1, `rgba(255, 255, 255, ${(je > 3.5 ? 0 : Math.exp(-0.5 * je * je)).toFixed(4)})`);
        o.globalCompositeOperation = 'destination-in';
        o.fillStyle = ring;
        o.fillRect(0, 0, w, h);
        g.drawImage(oc, 0, 0);
      };

      // L5 — lensing whisper: interior hues re-appear in a thin band inside the
      // rim (the Liquid Glass identity move, live_page_orb.md §8) — but PASTEL
      // (hues mixed 50% toward white) and at low alpha; at mini size a saturated
      // band stops reading as refraction and becomes a candy ring.
      const lensA = isLight ? 0.16 : 0.20;
      const pastel = (c: [number, number, number]) =>
        `rgba(${Math.round(c[0] * 0.5 + 127.5)}, ${Math.round(c[1] * 0.5 + 127.5)}, ${Math.round(c[2] * 0.5 + 127.5)}, ${lensA})`;
      ringLayer((o) => {
        const lens = o.createConicGradient(-Math.PI / 2, cx, cy);
        [0, 1, 2, 0].forEach((ci, k) => lens.addColorStop(k / 3, pastel(cols[ci])));
        o.fillStyle = lens;
        o.fillRect(0, 0, w, h);
      }, R * 0.93, R * 0.035);

      // L8 — caustic: refracted light pooling at the bottom interior (accent-
      // tinted white), gaussian exp(−0.9·d²) sampled to ~zero — truncating a
      // gradient mid-curve leaves a visible edge circle.
      const tint: [number, number, number] = [
        Math.round(0.55 * cols[0][0] + 0.45 * 255),
        Math.round(0.55 * cols[0][1] + 0.45 * 255),
        Math.round(0.55 * cols[0][2] + 0.45 * 255),
      ];
      const causticA = isLight ? 0.18 : 0.26;
      g.save();
      g.translate(cx, cy + 0.60 * R);
      g.scale(1, 0.522); // circular gradient → elliptical pool (ry/rx of the study)
      const caustic = g.createRadialGradient(0, 0, 0, 0, 0, 0.92 * R);
      ([[0, 1], [0.15, 0.922], [0.3, 0.723], [0.45, 0.483], [0.6, 0.274], [0.75, 0.132], [0.9, 0.054], [1, 0]] as [number, number][])
        .forEach(([q, k]) => caustic.addColorStop(q, rgba(tint, causticA * k)));
      g.fillStyle = caustic;
      g.beginPath();
      g.arc(0, 0, 0.92 * R, 0, Math.PI * 2);
      g.fill();
      g.restore();

      // L6 — rim-light arcs at 0.965R: key top-left (−125°, σ40°) + faint bounce
      // bottom-right (55°, σ30°). Gaussian in angle AND radius so the arcs die out
      // silently — the old linear tents covered 260° of the circle and fused with
      // the lens band into a near-complete white ring.
      ringLayer((o) => {
        const arcs = o.createConicGradient(-Math.PI / 2, cx, cy);
        for (let d = 0; d <= 360; d += 5) {
          const a = 0.60 * Math.exp(-0.5 * Math.pow(angDiff(d, 325) / 40, 2))
                  + 0.20 * Math.exp(-0.5 * Math.pow(angDiff(d, 145) / 30, 2));
          arcs.addColorStop(d / 360, `rgba(250, 250, 255, ${a.toFixed(3)})`);
        }
        o.fillStyle = arcs;
        o.fillRect(0, 0, w, h);
      }, R * 0.965, R * 0.022);

      // L7a — specular STREAK at 0.80R, −118° ± σ17°: a short window-reflection
      // smear along the curvature (long arcs read as yet another concentric ring).
      ringLayer((o) => {
        const streak = o.createConicGradient(-Math.PI / 2, cx, cy);
        for (let d = 0; d <= 360; d += 4) {
          const a = 0.50 * Math.exp(-0.5 * Math.pow(angDiff(d, 332) / 17, 2));
          streak.addColorStop(d / 360, `rgba(255, 255, 255, ${a.toFixed(3)})`);
        }
        o.fillStyle = streak;
        o.fillRect(0, 0, w, h);
      }, R * 0.80, R * 0.045);

      // L7b — hot window-reflection dot toward the key light (surface, pure white:
      // tinted speculars read as plastic, live_page_orb.md §8).
      const dotSigma = 0.045 * R;
      const dx = cx - 0.36 * R;
      const dy = cy - 0.44 * R;
      const dot = g.createRadialGradient(dx, dy, 0, dx, dy, 3 * dotSigma);
      ([[0, 1], [0.17, 0.878], [0.33, 0.606], [0.5, 0.325], [0.67, 0.135], [0.83, 0.045], [1, 0]] as [number, number][])
        .forEach(([q, k]) => dot.addColorStop(q, `rgba(255, 255, 255, ${(0.92 * k).toFixed(3)})`));
      g.fillStyle = dot;
      g.beginPath();
      g.arc(dx, dy, 3 * dotSigma, 0, Math.PI * 2);
      g.fill();

      g.restore();
      shellDirty = false;
    };

    // Wave scratch canvas: all ring passes flatten here with 'lighten' (per-channel
    // max — the canvas equivalent of the study's np.maximum combine), then the
    // result is page-blended ONCE. Per-pass multiply used to stack: three halo
    // passes multiplying each other were the "deep shadow grooves" on light theme
    // (the trap live_page_orb.md §6 lesson 4 warns about).
    const waveCanvas = document.createElement('canvas');
    const wctx = waveCanvas.getContext('2d');

    const render = () => {
      // Dynamic canvas backbuffer resizing matching actual container sizes each frame
      const dpr = window.devicePixelRatio || 1;
      const container = canvas.parentElement;
      if (container) {
        const rect = container.getBoundingClientRect();
        const displayW = Math.round(rect.width);
        const displayH = Math.round(rect.height);

        if (canvas.width !== displayW * dpr || canvas.height !== displayH * dpr) {
          canvas.width = displayW * dpr;
          canvas.height = displayH * dpr;
        }
      }
      if (shellCanvas && (shellDirty || shellCanvas.width !== canvas.width || shellCanvas.height !== canvas.height)) {
        buildGlassShell();
      }
      if (waveCanvas.width !== canvas.width || waveCanvas.height !== canvas.height) {
        waveCanvas.width = canvas.width;
        waveCanvas.height = canvas.height;
      }

      // The orb tracks TEXT-GENERATION THROUGHPUT, not audio. We're a VLM: the
      // model streams tokens in real time while the single downstream TTS is
      // serialized/late — so the live "energy" is the token stream. genLevel
      // (chars/sec → smoothed 0..1, from useSession / the scripted reveal) rises
      // when generation is fast and eases down through the gaps. A gentle floor
      // WHILE a response is open keeps the orb present between token bursts (the
      // TTFT gap, punctuation pauses) so it reads as "alive", not flickering.
      // Every other state (idle · listening · finished) just rotates.
      const responding = streamConnected && respondingRef.current;
      const genLevel = streamConnected ? sessionGenLevel() : 0;
      const target = Math.max(genLevel, responding ? 0.3 : 0);
      // envelope: swell in on a burst, settle smoothly when it eases
      loudRef.current += (target - loudRef.current) * (target > loudRef.current ? 0.22 : 0.06);

      // aurora glow keeps its floor-0.15 idle shimmer, brightening while generating
      targetIntensityRef.current = 0.15 + loudRef.current * 1.05;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      orbIntensityRef.current += (targetIntensityRef.current - orbIntensityRef.current) * 0.15;
      // Rings ALWAYS rotate (phaseRef). The standing-wave flip's DEPTH + AMPLITUDE
      // come from `loudRef` (throughput; in Wave.draw), and its FREQUENCY does too:
      // faster generation → quicker ripple, calmer when slow — a subtle ~2× range
      // (never frenetic) so it reads as fluid Liquid Glass, not a buzzer.
      phaseRef.current += reduceMotion ? 0.045 : 0.11;   // steady rotation
      vibPhaseRef.current += reduceMotion ? 0 : (0.075 + 0.075 * loudRef.current); // flip rate ← throughput

      // Glass amount: 1 = minimized ball (user-focus), 0 = focused free rings.
      // Lerped so the shell MATERIALIZES in step with the 0.5s focus-swap morph —
      // Liquid Glass thickness scaling played in reverse: the smaller the orb,
      // the thicker the glass (live_page_orb.md §6).
      const sphereTarget = focusedFeedRef.current === 'user' ? 1 : 0;
      sphereTRef.current += (sphereTarget - sphereTRef.current) * (reduceMotion ? 1 : 0.12);
      const t = sphereTRef.current;

      const half = canvas.width / 2;
      const cx = half;
      const cy = canvas.height / 2;
      const sphereR = half * 0.99; // silhouette ≈ container edge (see buildGlassShell)

      // Mini-ball intensity FLOOR (the study renders at i≈0.55): at idle i=0.15 the
      // three sinusoids flatten into circles at the SAME radius — one sharp colored
      // ring, the exact failure being designed away. Inside the glass the waves stay
      // separated and lively even in silence.
      const waveI = Math.max(orbIntensityRef.current, 0.45 * t);
      const clampedI = Math.min(waveI, 1.2);

      // Ring base: free 50% of the half-canvas (existing headroom) → 53% of the
      // ball radius once sealed inside the glass (study wave_r = 0.60, tightened
      // so the rings keep breathing room to the lens band instead of crowding
      // the sphere).
      // waveInset shrinks the internal waves ONLY in the minimized ball (t→1,
      // the video-call state) so the rings occupy a smaller area inside the
      // glass; the focused hero orb (t=0) is untouched (waveInset→1). Shell and
      // silhouette come from sphereR and stay full-size either way.
      // Minimized rings sized to ~50% of the glassball DIAMETER (radius ≈
      // 0.5·sphereR): inset 0.85 + base factor 0.59 (0.59·0.85 ≈ 0.50). The
      // focused hero orb (t=0) is untouched (waveInset→1, base = half·0.5).
      const waveInset = 1 - (1 - 0.85) * t;
      const baseRadius = (half * 0.5 + (sphereR * 0.59 - half * 0.5) * t) * waveInset;
      // Aurora core: hugs the rings when focused ("colorful fillings"), fills the
      // ball when minimized — 0.90·sphereR, so the plasma reaches the lens band
      // instead of dying mid-glass (only the rings keep the waveInset shrink).
      // match the waves: speech barely grows the plasma envelope (it brightens
      // and its blobs orbit faster instead — see blobA / blobPhase below)
      const ringR = baseRadius * (1 + clampedI * 0.10);
      const coreR = ringR * 1.15 + (sphereR * 0.90 - ringR * 1.15) * t;

      // ---- L3 aurora plasma: three theme-hue blobs drifting through the core ----
      // Orbit speed accumulates (never jumps) and scales with voice intensity.
      blobPhaseRef.current += reduceMotion ? 0 : 0.0028 * (1 + orbIntensityRef.current);
      // +0.06·t: the minimized ball carries a denser core so it stays a BALL over video
      const blobA = Math.min(0.38 * (1 + 0.6 * orbIntensityRef.current), 0.5) + 0.06 * t;
      ctx.globalCompositeOperation = 'screen';
      for (let i = 0; i < 3; i++) {
        const ang = blobPhaseRef.current * (0.7 + 0.3 * i) + i * ((Math.PI * 2) / 3);
        const bx = cx + Math.cos(ang) * 0.42 * coreR;
        const by = cy + Math.sin(ang) * 0.36 * coreR + 0.10 * coreR; // energy pools low
        // true gaussian profile exp(−1.6·d²) sampled to ~zero at 1.6× the blob
        // radius — truncating at 1× leaves a visible disc edge inside the ball
        const br = 0.62 * coreR;
        const blob = ctx.createRadialGradient(bx, by, 0, bx, by, br * 1.6);
        ([[0, 1], [0.16, 0.9], [0.31, 0.67], [0.44, 0.46], [0.53, 0.32], [0.63, 0.2], [0.78, 0.08], [1, 0]] as [number, number][])
          .forEach(([q, k]) => blob.addColorStop(q, rgba(cols[i], blobA * k)));
        ctx.fillStyle = blob;
        ctx.beginPath();
        ctx.arc(bx, by, br * 1.6, 0, Math.PI * 2);
        ctx.fill();
      }
      // Depth fade: the aurora dims toward its boundary so it reads as sitting
      // BEHIND glass, not painted on it (destination-in touches only the blobs
      // drawn above — the rings come after and stay crisp). Two curves lerped
      // by sphereT, because the boundary means different things per mode:
      //   focused — no shell hides the disc edge, and against a pure white/black
      //     page the old 0.4→0 drop over the last 1.5% read as a hard circle
      //     ("rendering edge"); the fade widens to 1.5× the core and decays as a
      //     gaussian bell, softened from the very start so the color halo simply
      //     evaporates. (Capped at the canvas edge: at shout intensity the disc
      //     must still die before the square backing store shows.)
      //   minimized — a DENSER bell (the core must survive over live video)
      //     that still decays to zero by fadeR = 0.90·sphereR, handing the last
      //     10% to the lens band + fresnel. The old study column ended 0.4→0
      //     over its final 1.5% — fine when the fade died AT the silhouette
      //     under vessel+fresnel, but once waveInset pulled the fade edge into
      //     open glass the cliff read as a hard circle mid-ball. Both columns
      //     now evaporate smoothly; they differ only in interior density.
      const fadeR = Math.min(coreR * (1.5 - 0.5 * t), half * 0.99);
      const fade = ctx.createRadialGradient(cx, cy, 0, cx, cy, fadeR);
      ([[0, 1, 1], [0.4, 0.98, 0.97], [0.6, 0.88, 0.90], [0.75, 0.66, 0.78], [0.85, 0.42, 0.55],
        [0.92, 0.22, 0.32], [0.96, 0.10, 0.14], [0.985, 0.04, 0.04], [1, 0, 0]] as [number, number, number][])
        .forEach(([q, af, am]) => fade.addColorStop(q, `rgba(255, 255, 255, ${(af + (am - af) * t).toFixed(3)})`));
      ctx.globalCompositeOperation = 'destination-in';
      ctx.fillStyle = fade;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // ---- L4 the rings — flattened on the scratch canvas with 'lighten', then
      // page-blended ONCE (multiply on light / screen on dark), clipped inside the
      // silhouette as the glass materializes. One blend application = the study's
      // np.maximum-then-single-composite; per-pass multiply deepened every stroke
      // crossing into a shadow groove.
      // (width guard: drawImage THROWS on a 0-size source, and the canvas collapses
      // to 0×0 while the streaming workspace is hidden in chat mode)
      if (wctx && waveCanvas.width > 0) {
        wctx.clearRect(0, 0, waveCanvas.width, waveCanvas.height);
        wctx.save();
        wctx.beginPath();
        wctx.arc(cx, cy, half * 1.5 + (sphereR * 0.975 - half * 1.5) * t, 0, Math.PI * 2);
        wctx.clip();
        wctx.globalCompositeOperation = 'lighten';
        orbWaves.forEach((wave) => wave.draw(wctx, waveI, orbTheme, baseRadius, dpr));
        wctx.restore();
        ctx.globalCompositeOperation = isLight ? 'multiply' : 'screen';
        ctx.drawImage(waveCanvas, 0, 0);
      }
      ctx.globalCompositeOperation = 'source-over';

      // ---- the glass shell overlay: static paint, opacity tracks sphereT ----
      if (shellCanvas) {
        shellCanvas.style.opacity = t < 0.004 ? '0' : t.toFixed(3);
      }

      // ---- L1 halo pulse: compositor-only transform on the CSS glow div ----
      if (orbGlowRef.current) {
        orbGlowRef.current.style.transform = `scale(${(1 + 0.1 * orbIntensityRef.current).toFixed(4)})`;
      }

      requestRef.current = requestAnimationFrame(render);
    };

    // Pause the RAF loop while the tab/app is hidden (phone battery + thermals;
    // background tabs get throttled anyway, backgrounded PWAs not always).
    const handleVisibility = () => {
      if (document.hidden) {
        if (requestRef.current) cancelAnimationFrame(requestRef.current);
        requestRef.current = null;
      } else if (requestRef.current === null) {
        render();
      }
    };
    document.addEventListener('visibilitychange', handleVisibility);

    render();

    return () => {
      document.removeEventListener('visibilitychange', handleVisibility);
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [streamConnected, micMuted, orbTheme, theme, sessionOutputLevel, sessionGenLevel]);

  // ==================== OFFLINE CHATBOT INTERACTION ====================
  // The four generative-card demos (metrics/audio/code/help) stay CLIENT-side —
  // they showcase UI components, not the model. Everything else streams from
  // the real backend: POST /api/chat/stream (SSE).

  // -------- media attachments (paperclip → POST /api/media) --------
  const MAX_ATTACHMENTS = 4;

  // One chat thread = one durable conversation_id (journal filename charset:
  // [A-Za-z0-9_-]); minted per thread, reset on `clear`/new-chat, and set to
  // the loaded conversation's id when a history session is opened (so
  // follow-ups CONTINUE that conversation).
  const newThreadId = () =>
    'web_' + (crypto.randomUUID?.() ?? `${Date.now()}${Math.random()}`).replace(/[^A-Za-z0-9]/g, '').slice(0, 28);
  const chatThreadIdRef = useRef<string>(newThreadId());
  // The wire-shaped conversation so far — resent with every request so the
  // model sees the full dialogue through the chat template. Committed only on
  // a successful turn (an errored request leaves the context untouched).
  // Canned demo cards never enter it: the model never saw them.
  const threadMessagesRef = useRef<WireMessage[]>([]);
  // In-flight SSE stream (non-null while the model generates) — the send
  // button morphs into ⏹ stop and aborts it.
  const chatAbortRef = useRef<AbortController | null>(null);
  // The last REAL model turn's inputs (canned demo cards excluded) — what the
  // regenerate button re-asks.
  const lastTurnRef = useRef<{ query: string; images?: string[]; videos?: string[] } | null>(null);
  // Whether the tail of threadMessagesRef holds the last exchange (success or
  // stop commit it; an errored request leaves the context untouched) — edit
  // and regenerate must only roll back what actually landed.
  const lastTurnCommittedRef = useRef<boolean>(false);

  // Downscale to ≤1280px JPEG before upload: a raw phone photo is 5-8 MB that
  // the box's processor would immediately shrink anyway; ~1280 keeps the
  // payload a few hundred KB with no loss for VLM understanding.
  const downscaleImage = async (file: File): Promise<Blob> => {
    const bitmap = await createImageBitmap(file);
    const scale = Math.min(1, 1280 / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(bitmap.width * scale));
    canvas.height = Math.max(1, Math.round(bitmap.height * scale));
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('canvas 2d unavailable');
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close();
    return await new Promise<Blob>((resolve, reject) =>
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('jpeg encode failed'))), 'image/jpeg', 0.85));
  };

  const blobToDataUrl = (blob: Blob): Promise<string> =>
    new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(String(r.result));
      r.onerror = () => reject(r.error);
      r.readAsDataURL(blob);
    });

  /** Hex sha256 of a blob via WebCrypto — null when unavailable (non-secure
   *  origin) or on any failure; callers fall back to a plain upload. The blob
   *  is buffered whole (subtle.digest has no streaming input); fine under the
   *  512 MB upload cap, ~1-2 s for the biggest files. */
  const hashBlobSha256 = async (blob: Blob): Promise<string | null> => {
    try {
      if (!crypto?.subtle?.digest) return null;
      const digest = await crypto.subtle.digest('SHA-256', await blob.arrayBuffer());
      return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
    } catch {
      return null;
    }
  };

  /** Upload one payload to the CAS; returns the media descriptor. XHR (not
   *  fetch) so `onProgress` gets real byte progress for the glass puck, and
   *  `getXhr` hands out the handle so a removed chip can abort mid-flight
   *  (abort rejects with 'upload cancelled').
   *
   *  Videos are client-side deduped first: their blobs are stored
   *  byte-identical, so a local sha256 + GET /api/media/{hex}/info can skip
   *  the transfer outright (re-picking an already-uploaded file is instant).
   *  Images can't probe — the server re-encodes them, so the stored identity
   *  never matches a local hash — and they're small post-downscale anyway. */
  const uploadMedia = async (
    payload: Blob,
    filename: string,
    opts?: { onProgress?: (frac: number) => void; getXhr?: (xhr: XMLHttpRequest) => void },
  ): Promise<{ hash: string; kind: 'image' | 'video'; url: string; thumb_url: string | null }> => {
    if (payload.type.startsWith('video/')) {
      const hex = await hashBlobSha256(payload);
      if (hex) {
        try {
          const res = await fetch(resolveApiUrl(`/api/media/${hex}/info`));
          if (res.ok) {
            const desc = await res.json();
            opts?.onProgress?.(1); // no bytes to send — flip straight to done
            return desc;
          }
        } catch {
          /* probe unreachable — fall through to the plain upload */
        }
      }
    }
    return new Promise<{ hash: string; kind: 'image' | 'video'; url: string; thumb_url: string | null }>(
      (resolve, reject) => {
        const form = new FormData();
        form.append('file', payload, filename);
        const xhr = new XMLHttpRequest();
        opts?.getXhr?.(xhr);
        xhr.open('POST', resolveApiUrl('/api/media'));
        xhr.responseType = 'json';
        if (opts?.onProgress) {
          xhr.upload.onprogress = (e) => {
            if (e.lengthComputable && e.total > 0) opts.onProgress!(e.loaded / e.total);
          };
          // the last progress event may stop short of total — loadend is the
          // reliable "all bytes sent" signal that flips pucks to processing
          xhr.upload.onloadend = () => opts.onProgress!(1);
        }
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response);
          else reject(new Error(xhr.response?.detail || `upload failed (HTTP ${xhr.status})`));
        };
        xhr.onerror = () => reject(new Error('upload failed (network)'));
        xhr.onabort = () => reject(new Error('upload cancelled'));
        xhr.send(form);
      },
    );
  };

  /** First decodable frame of a picked video, as a small JPEG data-URL — the
   *  chip's poster while the real thumbnail is still server-side. '' on any
   *  decode failure (the chip falls back to the film glyph). */
  const captureVideoPoster = (file: File): Promise<string> =>
    new Promise((resolve) => {
      const url = URL.createObjectURL(file);
      const v = document.createElement('video');
      let settled = false;
      const done = (poster: string) => {
        if (settled) return;
        settled = true;
        URL.revokeObjectURL(url);
        resolve(poster);
      };
      v.muted = true;
      v.playsInline = true;
      v.preload = 'auto'; // metadata alone may not decode frame 1
      v.onloadeddata = () => {
        try {
          const s = Math.min(1, 320 / Math.max(v.videoWidth || 1, v.videoHeight || 1));
          const c = document.createElement('canvas');
          c.width = Math.max(1, Math.round((v.videoWidth || 1) * s));
          c.height = Math.max(1, Math.round((v.videoHeight || 1) * s));
          c.getContext('2d')?.drawImage(v, 0, 0, c.width, c.height);
          done(c.toDataURL('image/jpeg', 0.72));
        } catch {
          done(''); // e.g. tainted canvas — glyph fallback
        }
      };
      v.onerror = () => done('');
      setTimeout(() => done(''), 3000); // codec stall guard
      v.src = url;
    });

  const handleFilesChosen = async (list: FileList | null) => {
    if (!list) return;
    // every pick runs independently — chip N+1 must not wait for chip N's
    // transfer (each mounts instantly and uploads concurrently)
    await Promise.all(Array.from(list).map(async (file) => {
      const isVideo = file.type.startsWith('video/');
      const id = `att-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      try {
        const payload = isVideo ? file : await downscaleImage(file);
        // The chip mounts IMMEDIATELY with a local preview and an uploading
        // glass puck — a slow transfer must never leave the pick invisible.
        const localUrl = URL.createObjectURL(payload);
        const poster = isVideo ? await captureVideoPoster(file) : localUrl;
        setAttachments((prev) => (prev.length >= MAX_ATTACHMENTS ? prev : [...prev, {
          id, kind: isVideo ? 'video' : 'image', hash: '',
          url: localUrl, previewUrl: poster, name: file.name,
          uploading: true, progress: 0,
        } satisfies Attachment]));
        // fast uploads never frost: the fill presentation engages only if
        // the transfer is still running when the grace window closes
        setTimeout(() => {
          setAttachments((prev) => prev.map((a) =>
            a.id === id && a.uploading ? { ...a, slow: true } : a));
        }, 300);
        try {
          const desc = await uploadMedia(payload, file.name, {
            // bytes done + response pending = server-side processing → null
            // flips the chip's ring to the indeterminate spin
            onProgress: (p) => setAttachments((prev) => prev.map((a) =>
              a.id === id ? { ...a, progress: p >= 1 ? null : p } : a)),
            getXhr: (xhr) => attachmentXhrsRef.current.set(id, xhr),
          });
          setAttachments((prev) => prev.map((a) => a.id === id ? {
            ...a, kind: desc.kind, hash: desc.hash,
            url: resolveApiUrl(desc.url),
            previewUrl: desc.thumb_url ? resolveApiUrl(desc.thumb_url) : a.previewUrl,
            uploading: false, progress: undefined,
          } : a));
          // the CAS copy took over both slots → the object URL can go (the
          // no-thumb fallback keeps previewUrl on the local blob — keep it)
          if (desc.thumb_url) URL.revokeObjectURL(localUrl);
        } catch (uploadErr) {
          if (uploadErr instanceof Error && uploadErr.message === 'upload cancelled') {
            URL.revokeObjectURL(localUrl); // chip already removed by its ×
            return;
          }
          if (isVideo) {
            // no inline fallback for video payloads — drop the chip
            setAttachments((prev) => prev.filter((a) => a.id !== id));
            URL.revokeObjectURL(localUrl);
            throw uploadErr;
          }
          // media store unreachable → legacy inline base64 (still works E2E)
          const dataUrl = await blobToDataUrl(payload);
          setAttachments((prev) => prev.map((a) => a.id === id
            ? { ...a, hash: '', url: dataUrl, previewUrl: dataUrl, uploading: false, progress: undefined }
            : a));
          URL.revokeObjectURL(localUrl);
          addLog(language === 'en' ? 'Media upload unavailable — image attached inline.' : '媒体上传不可用 — 图片已内联附加。', 'red');
        } finally {
          attachmentXhrsRef.current.delete(id);
        }
      } catch (e) {
        console.warn('attachment import failed:', e);
        addLog(
          language === 'en'
            ? `Attachment rejected: ${e instanceof Error ? e.message : 'unreadable file'}`
            : `附件被拒绝：${e instanceof Error ? e.message : '无法读取的文件'}`,
          'red',
        );
      }
    }));
  };
  const offlineCardResponse = (
    query: string,
  ): { responseText: string; cardType?: 'metrics' | 'audio' | 'code' } | null => {
    const q = query.toLowerCase();
    if (q.includes('performance') || q.includes('chart') || q.includes('/metrics') || q.includes('图表') || q.includes('性能')) {
      return {
        responseText: language === 'en'
          ? "I've compiled the latest performance diagnostics block for the MOSS local execution queue."
          : '我已生成本地 MOSS 运行管道的实时性能诊断看板。',
        cardType: 'metrics',
      };
    }
    if (q.includes('waveform') || q.includes('audio') || q.includes('/music') || q.includes('波形') || q.includes('音频')) {
      return {
        responseText: language === 'en'
          ? 'Here is the interactive audio player containing custom soundwave elements.'
          : '已为您生成具有自适应滑刷反馈的交互式音频波形播放器控件。',
        cardType: 'audio',
      };
    }
    if (q.includes('css') || q.includes('/code') || (q.includes('code') && q.includes('glass')) || q.includes('代码') || q.includes('布局')) {
      return {
        responseText: language === 'en'
          ? 'Here is the CSS specification for the Apple Liquid Glass design variables.'
          : '这是生成好的 Apple Liquid Glass 玻璃微态设计变量样式规范代码块：',
        cardType: 'code',
      };
    }
    if (q.includes('/help') || (q.includes('help') && q.includes('command')) || q.includes('帮助') || q.includes('指令')) {
      return {
        responseText: language === 'en'
          ? "Offline mode commands ready:\n\n" +
            "• Type 'metrics' to view the performance dashboard.\n" +
            "• Type 'css' to print the Liquid Glass code block.\n" +
            "• Type 'audio' to spawn the wave sound scrubber.\n" +
            "• Type 'clear' to purge the message stack.\n" +
            '• Anything else goes to the local MOSS model.\n' +
            '• Toggle the Switcher Button (vertical bar top) to start Streaming Video-Chat!'
          : "离线模式指令已就绪：\n\n" +
            "• 输入 '性能' / 'metrics' 查看运行指标图表。\n" +
            "• 输入 '代码' / 'css' 生成玻璃微态 CSS 属性。\n" +
            "• 输入 '音频' / 'audio' 调出声音播放器波形图。\n" +
            "• 输入 'clear' 或是 '/clear' 清空对话栈。\n" +
            '• 其它任意内容将交由本地 MOSS 模型推理回复。\n' +
            '• 点击左侧导航栏最上方的 🔄 图标切换为流式实时通话！',
      };
    }
    return null;
  };

  // Streams into ONE message (`aiId` — the vessel runChatTurn already added):
  // the empty vessel fills with type in place, the aurora ring keeps riding
  // the same bubble, and settle just stops the ring. No second bubble, no
  // remount, no entrance-animation replay (the old typing→reply swap).
  const streamChatFromBackend = async (
    query: string, aiId: string, images?: string[], videos?: string[],
  ) => {
    const t0 = performance.now();
    const aborter = new AbortController();
    chatAbortRef.current = aborter;
    lastTurnCommittedRef.current = false;
    let text = '';
    let started = false;
    const begin = () => {
      if (started) return;
      started = true;
      // first token: the SAME bubble flips thinking → streaming
      setChatMessages((prev) => prev.map((m) =>
        m.id === aiId ? { ...m, isTyping: false, isStreaming: true } : m));
    };
    // this turn's message: media ride IN the message as template parts —
    // images as CAS handles or inline data-URLs, videos as CAS handles only
    // (the backend resolves the blob and samples frames for the VLM) — so
    // multi-turn requests keep each attachment with the turn it belongs to
    const turnHasMedia = (images && images.length > 0) || (videos && videos.length > 0);
    const userMessage: WireMessage = turnHasMedia
      ? {
          role: 'user',
          content: [
            ...(images ?? []).map((s): WirePart =>
              s.startsWith('data:') ? { type: 'image', image: s } : { type: 'image', media: s }),
            ...(videos ?? []).map((s): WirePart => ({ type: 'video', media: s })),
            { type: 'text', text: query },
          ],
        }
      : { role: 'user', content: query };
    try {
      const res = await fetch(resolveApiUrl('/api/chat/stream'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          // full conversation (capped) + this turn — the backend applies the
          // chat template over all of it
          messages: [...threadMessagesRef.current.slice(-MAX_CONTEXT_MESSAGES), userMessage],
          // ties the turn pair to a durable history thread
          conversation_id: chatThreadIdRef.current,
        }),
        signal: aborter.signal,
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let sep;
        while ((sep = buf.indexOf('\n\n')) >= 0) {
          const frame = buf.slice(0, sep);
          buf = buf.slice(sep + 2);
          for (const line of frame.split('\n')) {
            if (!line.startsWith('data:')) continue;
            const payload = JSON.parse(line.slice(5).trim());
            if (payload.type === 'generation_delta') {
              begin();
              text += String(payload.delta ?? '');
              setChatMessages((prev) => prev.map((m) => (m.id === aiId ? { ...m, text } : m)));
            } else if (payload.type === 'generation_error') {
              throw new Error(String(payload.message ?? 'generation failed'));
            }
          }
        }
      }
      begin(); // an empty-but-successful generation still resolves the typing bubble
      // commit the turn pair to the thread context (errors skip this, so a
      // retry resends a clean conversation)
      threadMessagesRef.current = [
        ...threadMessagesRef.current,
        userMessage,
        ...(text ? [{ role: 'assistant', content: text } as WireMessage] : []),
      ];
      lastTurnCommittedRef.current = true;
      const latency = `${language === 'en' ? 'Local Model' : '本地模型'} • ${Math.round(performance.now() - t0)}ms`;
      // settle: shimmer tail comes off, plain type + latency pill remain
      setChatMessages((prev) => prev.map((m) => (m.id === aiId ? { ...m, latency, isStreaming: false } : m)));
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') {
        // user pressed stop: the partial text settles as the turn's content
        // and joins the context (the model DID say it), pill marks the cut
        threadMessagesRef.current = [
          ...threadMessagesRef.current,
          userMessage,
          ...(text ? [{ role: 'assistant', content: text } as WireMessage] : []),
        ];
        lastTurnCommittedRef.current = true;
        setChatMessages((prev) => prev.map((m) => (m.id === aiId ? {
          ...m,
          isTyping: false,
          isStreaming: false,
          ...(text ? {} : { halo: false }),
          text: text || (language === 'en' ? '(generation stopped)' : '（已停止生成）'),
          latency: language === 'en' ? 'Stopped' : '已停止',
        } : m)));
        return;
      }
      const message = e instanceof Error ? e.message : String(e);
      // the same bubble becomes the error notice (halo off — not a model turn)
      setChatMessages((prev) => prev.map((m) => (m.id === aiId ? {
        ...m,
        isTyping: false,
        isStreaming: false,
        halo: false,
        text: /offline chat not supported/i.test(message)
          ? (language === 'en'
            ? 'Offline chat is not enabled for this deployment. Switch to video-call mode.'
            : '当前部署未启用离线聊天，请切换到实时通话。')
          : (language === 'en'
          ? '⚠ The local model backend is unreachable — is the server running with a loaded checkpoint? '
          : '⚠ 本地模型后端暂不可用 — 请确认服务已启动并加载了模型。') + `(${message})`,
      } : m)));
    } finally {
      chatAbortRef.current = null;
      setChatInputBlocked(false);
    }
  };

  const runChatTurn = (query: string, images?: string[], videos?: string[]) => {
    // ONE message for the whole turn: born as the empty aurora vessel, fills
    // with streamed type, settles in place (key never changes → no remount)
    const aiId = 'ai-' + Date.now();
    setChatMessages((prev) => [
      ...prev,
      { id: aiId, sender: 'ai', text: '', isTyping: true, halo: true },
    ]);

    // a media turn always goes to the model — canned cards are text showcases
    const hasMedia = (images && images.length > 0) || (videos && videos.length > 0);
    const local = hasMedia ? null : offlineCardResponse(query);
    if (local) {
      // canned UI-showcase cards keep a short think-pause for the demo feel;
      // they never enter the wire context, so the tail is NOT this exchange
      lastTurnCommittedRef.current = false;
      setTimeout(() => {
        setChatMessages((prev) => prev.map((m) => (m.id === aiId ? {
          ...m,
          isTyping: false,
          text: local.responseText,
          generativeCard: local.cardType,
          cardId: 'card-' + Math.random().toString(36).substr(2, 9),
          latency: language === 'en' ? 'UI Demo Card' : '界面演示卡片',
        } : m)));
        setChatInputBlocked(false);
      }, 600);
    } else {
      lastTurnRef.current = { query, images, videos }; // regenerate re-asks this
      void streamChatFromBackend(query, aiId, images, videos);
    }
  };

  const submitChat = (messageText: string) => {
    if (chatInputBlocked) return;
    // chips still uploading (their rings say so) — the send waits for hashes
    if (attachments.some((a) => a.uploading)) return;
    const imageAtts = attachments.filter((a) => a.kind === 'image');
    const videoAtts = attachments.filter((a) => a.kind === 'video');
    // wire payloads: CAS handle when uploaded, inline data-URL as the fallback
    const wireImages = imageAtts.map((a) => (a.hash ? a.hash : a.url));
    const wireVideos = videoAtts.map((a) => a.hash).filter(Boolean);
    let query = messageText.trim();
    if (!query && attachments.length === 0) return;
    if (!query) {
      // media-only send: an honest implicit prompt (shown in the user bubble)
      query = language === 'en' ? 'Please describe the attached media.' : '请描述我发送的图片或视频。';
    }
    setChatInput('');

    if (query.toLowerCase() === 'clear' || query.toLowerCase() === '/clear') {
      chatAbortRef.current?.abort();
      stopReadAloud();
      setChatMessages([]);
      setAttachments([]);
      setChatInputBlocked(false);
      setDialogueStarted(false);
      setShowGreeting(true);
      chatThreadIdRef.current = newThreadId(); // next message starts a new thread
      threadMessagesRef.current = [];
      lastTurnRef.current = null;
      lastTurnCommittedRef.current = false;
      return;
    }

    setChatInputBlocked(true);
    setAttachments([]);
    const userMsg = {
      id: 'msg-' + Date.now(),
      sender: 'user' as const,
      text: query,
      ...(imageAtts.length > 0 ? { images: imageAtts.map((a) => a.previewUrl || a.url) } : {}),
      ...(videoAtts.length > 0 ? { videos: videoAtts.map((a) => a.url) } : {}),
    };

    if (dialogueStarted) {
      setChatMessages((prev) => [...prev, userMsg]);
      runChatTurn(query, wireImages, wireVideos);
    } else {
      // First message: greeting fades in place, THEN the layout shifts (the
      // greeting must never be seen being pushed up).
      setShowGreeting(false);
      setTimeout(() => {
        setDialogueStarted(true);
        setChatMessages([userMsg]);
        setTimeout(() => runChatTurn(query, wireImages, wireVideos), 500);
      }, 240);
    }
  };

  const handleSendClick = () => {
    submitChat(chatInput);
    focusChatInput(); // a button click moves focus off the field — put it back
  };

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      submitChat(chatInput);
      focusChatInput();
    }
  };

  // -------- chat bubble actions: stop / copy / edit / regenerate / read --------

  const stopChatGeneration = () => chatAbortRef.current?.abort();

  const [copiedMsgId, setCopiedMsgId] = useState<string | null>(null);
  const copyMessage = async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // clipboard API needs a secure context — textarea fallback
      const ta = document.createElement('textarea');
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
    }
    setCopiedMsgId(id);
    setTimeout(() => setCopiedMsgId((v) => (v === id ? null : v)), 1200);
  };

  /** ChatGPT-style edit of the LATEST user turn: the exchange is rolled back
   *  (view + wire context) and the text returns to the input box — sending
   *  re-asks the edited question. */
  const editLatestUserMessage = () => {
    if (chatInputBlocked) return;
    const idx = chatMessages.map((m) => m.sender).lastIndexOf('user');
    if (idx < 0) return;
    const msg = chatMessages[idx];
    if (lastTurnCommittedRef.current) {
      // the exchange reached the wire context — drop its assistant+user tail
      const msgs = [...threadMessagesRef.current];
      if (msgs.length && msgs[msgs.length - 1].role === 'assistant') msgs.pop();
      if (msgs.length && msgs[msgs.length - 1].role === 'user') msgs.pop();
      threadMessagesRef.current = msgs;
    }
    setChatMessages(chatMessages.slice(0, idx));
    setChatInput(msg.text);
    if ((msg.images?.length ?? 0) + (msg.videos?.length ?? 0) > 0) {
      addLog(language === 'en'
        ? 'Note: the edited turn\'s attachments were not restored — re-attach if needed.'
        : '提示：被编辑消息的附件未能恢复，如需请重新添加。');
    }
    lastTurnRef.current = null; // that exchange no longer exists
    if (idx === 0) {
      // the thread is empty again — return to the greeting stage
      setDialogueStarted(false);
      setShowGreeting(true);
    }
  };

  /** Re-ask the last real model turn (same query + attachments); the user
   *  bubble stays, the old reply is replaced by a fresh generation. */
  const regenerateLatest = () => {
    const last = lastTurnRef.current;
    if (!last || chatInputBlocked) return;
    if (lastTurnCommittedRef.current) {
      // streamChatFromBackend recommits the pair — remove the stale one first
      const msgs = [...threadMessagesRef.current];
      if (msgs.length && msgs[msgs.length - 1].role === 'assistant') msgs.pop();
      if (msgs.length && msgs[msgs.length - 1].role === 'user') msgs.pop();
      threadMessagesRef.current = msgs;
    }
    setChatMessages((prev) => {
      const revIdx = [...prev].reverse().findIndex((m) => m.sender === 'ai');
      if (revIdx < 0) return prev;
      return prev.filter((_, i) => i !== prev.length - 1 - revIdx);
    });
    setChatInputBlocked(true);
    runChatTurn(last.query, last.images, last.videos);
  };

  // -------- read-aloud (POST /api/tts → STREAMED WAV → WebAudio) --------
  // The endpoint streams WAV bytes while the engine is still synthesizing
  // (tts_serving_plan.md Stage 0), so playback starts on the first PCM chunk
  // (~1 s) instead of after the whole synth (~1 s of wait per 1 s of audio).
  // PCM frames are scheduled back-to-back on an AudioContext as they arrive.
  const [readingMsg, setReadingMsg] = useState<{ id: string; phase: 'loading' | 'playing' } | null>(null);
  const readStopRef = useRef<(() => void) | null>(null);
  const readTokenRef = useRef<object | null>(null);

  const stopReadAloud = () => {
    readTokenRef.current = null;
    const stop = readStopRef.current;
    readStopRef.current = null;
    stop?.();
    setReadingMsg(null);
  };

  const readAloud = async (id: string, text: string) => {
    if (readingMsg?.id === id) {
      stopReadAloud(); // second click = stop (also cancels a pending synth)
      return;
    }
    if (voicePreset === 'none') return; // TTS disabled: no read-aloud
    stopReadAloud();
    const token = {};
    readTokenRef.current = token;
    setReadingMsg({ id, phase: 'loading' });
    const ctrl = new AbortController();
    let ctx: AudioContext | null = null;
    let endTimer: ReturnType<typeof setTimeout> | null = null;
    readStopRef.current = () => {
      ctrl.abort();
      if (endTimer) clearTimeout(endTimer);
      void ctx?.close().catch(() => {});
    };
    try {
      const res = await fetch(resolveApiUrl('/api/tts'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.slice(0, 2000), voice: voicePreset }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) {
        const body = await res.json().catch(() => ({}) as { detail?: string });
        throw new Error(body.detail || `HTTP ${res.status}`);
      }
      const sampleRate = Number(res.headers.get('X-Audio-Sample-Rate')) || 48000;
      const channels = Math.max(1, Number(res.headers.get('X-Audio-Channels')) || 1);
      const frameBytes = channels * 2; // PCM16
      const reader = res.body.getReader();
      let skipped = 0; // the 44-byte streaming WAV header
      let carry = new Uint8Array(0); // partial frame between network chunks
      // Jitter buffer: build up PREROLL_S of audio BEFORE starting playback.
      // The endpoint streams clause-by-clause, so audio arrives in bursts with
      // gaps between clauses (up to ~0.5 s under GPU contention); starting on the
      // first chunk (old behaviour) underran the scheduler between clauses and
      // stuttered on the first words. Pre-rolling ~0.8 s gives a cushion; once
      // playing, synthesis (RTF < 1) stays ahead so the cushion holds.
      const PREROLL_S = 0.8;
      let playHead = 0;
      let playing = false;
      const pending: AudioBuffer[] = [];
      let pendingDur = 0;
      const flush = (c: AudioContext) => {
        playHead = c.currentTime + 0.05;
        for (const b of pending) {
          const s = c.createBufferSource();
          s.buffer = b;
          s.connect(c.destination);
          s.start(playHead);
          playHead += b.duration;
        }
        pending.length = 0;
        pendingDur = 0;
        playing = true;
        setReadingMsg({ id, phase: 'playing' });
      };
      for (;;) {
        const { done, value } = await reader.read();
        if (readTokenRef.current !== token) return; // stopped mid-stream
        if (done) break;
        if (!value?.length) continue;
        let bytes = value;
        if (skipped < 44) {
          const drop = Math.min(44 - skipped, bytes.length);
          bytes = bytes.subarray(drop);
          skipped += drop;
          if (!bytes.length) continue;
        }
        const merged = new Uint8Array(carry.length + bytes.length);
        merged.set(carry);
        merged.set(bytes, carry.length);
        const whole = merged.length - (merged.length % frameBytes);
        carry = merged.subarray(whole);
        if (!whole) continue;
        const pcm = new Int16Array(merged.buffer, 0, whole / 2);
        const frames = pcm.length / channels;
        ctx ??= new AudioContext();
        const buf = ctx.createBuffer(channels, frames, sampleRate);
        for (let c = 0; c < channels; c++) {
          const data = buf.getChannelData(c);
          for (let i = 0; i < frames; i++) data[i] = pcm[i * channels + c] / 32768;
        }
        if (playing) {
          const src = ctx.createBufferSource();
          src.buffer = buf;
          src.connect(ctx.destination);
          playHead = Math.max(playHead, ctx.currentTime + 0.02); // resync only on underrun
          src.start(playHead);
          playHead += buf.duration;
        } else {
          pending.push(buf);
          pendingDur += buf.duration;
          if (pendingDur >= PREROLL_S) flush(ctx);
        }
      }
      if (!ctx) {
        stopReadAloud(); // stream ended with no audio at all
        return;
      }
      if (!playing) flush(ctx); // short clip (< PREROLL_S): play what we buffered
      // idle down once the scheduled tail has played out
      const remainMs = Math.max(0, playHead - ctx.currentTime) * 1000 + 120;
      endTimer = setTimeout(() => {
        if (readTokenRef.current === token) stopReadAloud();
      }, remainMs);
    } catch (e) {
      if (readTokenRef.current === token) {
        stopReadAloud();
        addLog(
          (language === 'en' ? 'Read aloud failed: ' : '朗读失败：') +
            (e instanceof Error ? e.message : String(e)),
          'red',
        );
      }
    }
  };

  // last real-message indices — edit/regenerate only ever act on the tail
  const lastUserMsgIdx = chatMessages.map((m) => m.sender).lastIndexOf('user');
  const lastAiMsgIdx = chatMessages.map((m) => m.sender).lastIndexOf('ai');

  const handleNewChat = () => {
    if (isStreamingMode && session.connecting) return; // same guard as the call button
    setHistoryOpen(false);
    setDemosOpen(false);
    setActiveModal(null);
    if (isStreamingMode) {
      // Live page: end any call, then put the stage back to its first-load
      // state — the next connect starts a brand-new session (= a new history
      // conversation server-side; the finished one stays in the sidebar).
      if (streamConnected) {
        session.disconnect();
        stopWebMedia();
        addLog(language === 'en' ? 'Session disconnected. WebSocket closed.' : '会话已断开，WebSocket 传输关闭。', 'red');
      }
      subtitleDemoTimersRef.current.forEach(clearTimeout);
      subtitleDemoTimersRef.current = [];
      setSubtitleDemoActive(false);
      unloadMediaFile(); // release a picked file + its pending upload handle
      setVideoEnabled(false);
      setVideoSourceKind('camera');
      setVideoMenuOpen(false);
      setScreenPickerOpen(false);
      setMicMuted(false);
      setLiveReplay(null);
      setStreamConversation([]);
      liveLogEndedRef.current = false; // fresh log, no pending flush
      cancelLiveDictation(); // drop an in-flight dictation with its draft
      setLiveMediaTime(null); // New Chat clears the frozen media-time tag
      setStreamSpeaker('ai');
      setCaptionsText(
        language === 'en'
          ? 'Ready. Click the green call button to start a video chat.'
          : '已就绪，等待开启视频通话。点击绿色电话图标开始。',
      );
      setFocusedFeed('model');
      setLiveTextOpen(false);
      setLiveTextDraft('');
      setPendingLiveQueries([]);
      targetIntensityRef.current = 0.15;
    } else {
      chatAbortRef.current?.abort(); // a stream in flight dies with its thread
      stopReadAloud();
      setChatMessages([]);
      setChatInput('');     // new chat clears the composer …
      setAttachments([]);   // … and any staged attachment thumbnails
      setDialogueStarted(false);
      setShowGreeting(true);
      chatThreadIdRef.current = newThreadId(); // fresh conversation + context
      threadMessagesRef.current = [];
      lastTurnRef.current = null;
      lastTurnCommittedRef.current = false;
    }
  };

  // Load a recorded conversation (GET /api/history/{cid}) into the chat view
  // and CONTINUE it: the thread id becomes the loaded cid (new turns append to
  // the same conversation) and the wire context is rebuilt from the transcript
  // so the model sees the prior dialogue — including per-turn images via their
  // CAS handles. Media bubbles render straight off the CAS endpoints (thumb
  // for images, Range-seekable blob for videos).
  const loadHistorySession = async (cid: string, title: string | null) => {
    interface HistoryMedia { kind: string; hash: string; url: string; thumb_url: string | null }
    interface HistoryTurn {
      role: string; text: string; media: HistoryMedia[];
      metrics: { latency_ms?: number; ttft_ms?: number; asr_ms?: number } | null;
    }
    try {
      const res = await fetch(resolveApiUrl(`/api/history/${cid}`));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const conv = (await res.json()) as { kind: string; turns: HistoryTurn[] };
      if (conv.kind === 'realtime') {
        // a live session must never be continued as a chat thread — route it
        // to the read-only replay instead (normally unreachable: the chat
        // sidebar lists kind=chat only)
        void openLiveReplay(cid, title);
        return;
      }
      const label = language === 'en' ? 'Cached Session' : '历史会话';
      const msgs: ChatMessage[] = conv.turns.map((t, i) => {
        const images = t.media.filter((m) => m.kind === 'image')
          .map((m) => resolveApiUrl(m.thumb_url || m.url));
        const videos = t.media.filter((m) => m.kind === 'video')
          .map((m) => resolveApiUrl(m.url));
        const ms = t.metrics?.latency_ms ?? t.metrics?.ttft_ms;
        return {
          id: `hist-${cid}-${i}`,
          sender: t.role === 'user' ? 'user' as const : 'ai' as const,
          text: t.text || '',
          ...(images.length > 0 ? { images } : {}),
          ...(videos.length > 0 ? { videos } : {}),
          ...(t.role === 'assistant' ? { latency: ms ? `${label} • ${Math.round(ms)}ms` : label } : {}),
        };
      });
      setDialogueStarted(true);
      setShowGreeting(false);
      setChatMessages(msgs.length > 0 ? msgs : [{
        id: 'hist-empty', sender: 'ai',
        text: language === 'en' ? `No turns recorded in "${title ?? cid}".` : `会话 "${title ?? cid}" 中没有已记录的对话。`,
      }]);
      // continuation: new turns append to this conversation, and the model
      // gets the recorded dialogue (with per-turn image/video handles) as context
      chatThreadIdRef.current = cid;
      lastTurnRef.current = null;          // nothing to regenerate yet
      lastTurnCommittedRef.current = true; // the wire tail IS the loaded exchange
      threadMessagesRef.current = conv.turns
        .filter((t) => (t.text || '').trim()
          || t.media.some((m) => m.kind === 'image' || m.kind === 'video'))
        .map((t): WireMessage => {
          const role = t.role === 'user' ? 'user' : 'assistant';
          const media = t.media
            .filter((m) => m.kind === 'image' || m.kind === 'video')
            .map((m): WirePart => m.kind === 'image'
              ? { type: 'image', media: m.hash }
              : { type: 'video', media: m.hash });
          if (media.length > 0) {
            return { role, content: [...media, { type: 'text', text: t.text || '' }] };
          }
          return { role, content: t.text || '' };
        });
    } catch (e) {
      addLog(
        language === 'en'
          ? `Failed to load history: ${e instanceof Error ? e.message : e}`
          : `历史会话加载失败：${e instanceof Error ? e.message : e}`,
        'red',
      );
    }
  };

  // Load a recorded LIVE session into the read-only replay modal. The source
  // video (file-stream sessions) is the `source:"video"` media-only turn the
  // backend journals on video.attach; dialogue turns carry metrics.media_ts —
  // the video position each turn happened at.
  const openLiveReplay = async (cid: string, title: string | null) => {
    interface HistoryMedia { kind: string; url: string; duration_s: number | null }
    interface HistoryTurn {
      role: string; source: string | null; text: string; ts: number;
      media: HistoryMedia[]; metrics: { media_ts?: number; kind?: string; name?: string } | null;
    }
    try {
      const res = await fetch(resolveApiUrl(`/api/history/${cid}`));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const conv = (await res.json()) as { created_at: number; title: string | null; turns: HistoryTurn[] };
      let video: LiveReplayData['video'] = null;
      for (const t of conv.turns) {
        const m = t.media.find((x) => x.kind === 'video');
        if (m) {
          video = { url: resolveApiUrl(m.url), durationS: m.duration_s ?? null };
          break;
        }
      }
      const turns: LiveReplayTurn[] = conv.turns
        .filter((t) => (t.text || '').trim() || t.source === 'source_change')
        .map((t) => (t.source === 'source_change' || t.source === 'memory_notice')
          ? {
              role: 'user' as const,
              system: true,
              text: t.source === 'memory_notice' ? t.text : sourceChangeLabel(t.metrics?.kind, t.metrics?.name),
              ts: t.ts,
              mediaTs: null,
            }
          : {
              role: t.role === 'user' ? 'user' as const : 'assistant' as const,
              text: t.text,
              ts: t.ts,
              mediaTs: typeof t.metrics?.media_ts === 'number' ? t.metrics.media_ts : null,
            });
      setReplayTime(0);
      setLiveReplay({ cid, title: title ?? conv.title, createdAt: conv.created_at, video, turns });
    } catch (e) {
      addLog(
        language === 'en'
          ? `Failed to load session replay: ${e instanceof Error ? e.message : e}`
          : `会话回放加载失败：${e instanceof Error ? e.message : e}`,
        'red',
      );
    }
  };

  // History-item click router: live sessions (realtime) open the read-only
  // replay; chat threads load into the chat view and CONTINUE.
  const openHistoryItem = (item: HistoryEntry) => {
    if (item.kind === 'realtime') void openLiveReplay(item.conversation_id, item.title);
    else void loadHistorySession(item.conversation_id, item.title);
  };

  const seekReplay = (seconds: number) => {
    const v = replayVideoRef.current;
    if (!v) return;
    v.currentTime = Math.max(0, seconds);
    void v.play().catch(() => undefined);
  };

  const fmtReplayTs = (seconds: number) => {
    const s = Math.max(0, Math.floor(seconds));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  };

  // The dialogue line the replay video is currently AT: the last turn whose
  // media_ts has been reached. -1 when there is no video to track.
  let replayActiveIdx = -1;
  if (liveReplay?.video) {
    liveReplay.turns.forEach((t, i) => {
      if (t.mediaTs !== null && t.mediaTs <= replayTime + 0.001) replayActiveIdx = i;
    });
  }
  const replayLogRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = replayLogRef.current?.querySelector('.replay-active');
    el?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [replayActiveIdx]);

  // ==================== WAVEFORM PLAYER CONTROLLER ====================
  const toggleAudioCardPlay = (cardId: string) => {
    setAudioPlayers((prev) => {
      const current = prev[cardId] || { isPlaying: false, currentIndex: 0 };
      const nextPlay = !current.isPlaying;
      
      if (nextPlay) {
        if (audioIntervalsRef.current[cardId]) clearInterval(audioIntervalsRef.current[cardId]);
        
        audioIntervalsRef.current[cardId] = setInterval(() => {
          setAudioPlayers((latest) => {
            const stateObj = latest[cardId] || { isPlaying: true, currentIndex: 0 };
            const nextIndex = stateObj.currentIndex + 1;
            
            if (nextIndex <= 10) {
              return {
                ...latest,
                [cardId]: { isPlaying: true, currentIndex: nextIndex }
              };
            } else {
              clearInterval(audioIntervalsRef.current[cardId]);
              delete audioIntervalsRef.current[cardId];
              return {
                ...latest,
                [cardId]: { isPlaying: false, currentIndex: 0 }
              };
            }
          });
        }, 600);
      } else {
        if (audioIntervalsRef.current[cardId]) {
          clearInterval(audioIntervalsRef.current[cardId]);
          delete audioIntervalsRef.current[cardId];
        }
      }

      return {
        ...prev,
        [cardId]: { isPlaying: nextPlay, currentIndex: current.currentIndex }
      };
    });
  };

  const scrubAudioCard = (cardId: string, index: number) => {
    if (audioIntervalsRef.current[cardId]) {
      clearInterval(audioIntervalsRef.current[cardId]);
      delete audioIntervalsRef.current[cardId];
    }
    
    setAudioPlayers((prev) => ({
      ...prev,
      [cardId]: { isPlaying: false, currentIndex: index + 1 }
    }));
  };

  const copyCodeToClipboard = (e: React.MouseEvent<HTMLButtonElement>, codeText: string) => {
    navigator.clipboard.writeText(codeText).then(() => {
      const button = e.currentTarget;
      const span = button.querySelector('span');
      if (span) {
        span.textContent = language === 'en' ? 'Copied!' : '已复制!';
        button.style.borderColor = 'var(--accent-live)';
        button.style.color = 'var(--accent-live)';
        setTimeout(() => {
          span.textContent = language === 'en' ? 'Copy' : '复制';
          button.style.borderColor = '';
          button.style.color = '';
        }, 2000);
      }
    });
  };

  // ==================== SIDEBAR ICONS (Phosphor Duotone) ====================
  // Glyphs come from one set at one weight/size via <IconContext> in the render
  // tree, so the whole rail stays optically consistent. Color is inherited from
  // each button's `currentColor` (muted -> primary on hover -> accent when active).

  // Shows the CURRENT mode at rest, swaps to the TARGET mode on hover (see .icon-swap CSS)
  const renderModeSwitchIcon = () => (
    <span className="icon-swap">
      <span className="swap-current">
        {isStreamingMode ? <VideoCamera className="nav-icon" /> : <ChatCircleDots className="nav-icon" />}
      </span>
      <span className="swap-target">
        {isStreamingMode ? <ChatCircleDots className="nav-icon" /> : <VideoCamera className="nav-icon" />}
      </span>
    </span>
  );

  const renderPlusIcon = () => <NotePencil className="nav-icon" />;
  const renderDemosIcon = () => <SquaresFour className="nav-icon" />;
  // History rows: chat threads keep the speech bubble; live sessions show
  // their INITIAL capture source (mid-session switches don't retitle the row).
  const renderHistoryItemIcon = (item: HistoryEntry) => {
    if (item.kind !== 'realtime') {
      return <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>;
    }
    switch (item.video_source) {
      case 'file': return <FileVideo size={12} />;
      case 'image': return <FileImage size={12} />;
      case 'screen': return <Monitor size={12} />;
      case 'none': return <VideoCameraSlash size={12} />;
      default: return <VideoCamera size={12} />;
    }
  };
  const renderThemeIcon = () => (
    <span className="icon-swap">
      <span className="swap-current">
        {theme === 'dark' ? <Moon className="nav-icon" /> : <Sun className="nav-icon" />}
      </span>
      <span className="swap-target">
        {theme === 'dark' ? <Sun className="nav-icon" /> : <Moon className="nav-icon" />}
      </span>
    </span>
  );
  const renderLanguageIcon = () => <Translate className="nav-icon" />;

  // Drawer demo chips (smaller optical size)
  const renderChartIcon = () => <ChartBar className="chip-icon" size={18} />;
  const renderWaveIcon = () => <Waveform className="chip-icon" size={18} />;
  const renderCodeIcon = () => <Code className="chip-icon" size={18} />;
  const renderHelpIcon = () => <Question className="chip-icon" size={18} />;

  const renderGenerativeCard = (cardType: 'metrics' | 'audio' | 'code', cardId?: string) => {
    if (cardType === 'metrics') {
      return (
        <div className="gen-ui-card">
          <div className="card-glass-specular"></div>
          <div className="gen-card-header">
            <span className="card-icon">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
              </svg>
            </span>
            <span className="card-title">Real-Time Performance Dashboard</span>
          </div>
          <div className="metrics-grid">
            <div className="metric-item">
              <div className="metric-label-row">
                <span className="metric-label">Speech-To-Text</span>
                <span className="metric-value">42ms</span>
              </div>
              <div className="metric-bar-container"><div className="metric-bar" style={{ width: '42%' }}></div></div>
            </div>
            <div className="metric-item">
              <div className="metric-label-row">
                <span className="metric-label">Model LLM Time</span>
                <span className="metric-value">128ms</span>
              </div>
              <div className="metric-bar-container"><div className="metric-bar" style={{ width: '78%' }}></div></div>
            </div>
            <div className="metric-item">
              <div className="metric-label-row">
                <span className="metric-label">Text-To-Speech</span>
                <span className="metric-value">35ms</span>
              </div>
              <div className="metric-bar-container"><div className="metric-bar" style={{ width: '35%' }}></div></div>
            </div>
          </div>
          <div className="card-footer-info">
            <span className="trust-score">✓ 99.4% confidence</span>
            <span className="audio-scrub-trigger" onClick={() => submitChat("Play an audio waveform snippet.")}>
              Listen to summary 
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="5 3 19 12 5 21 5 3"/>
              </svg>
            </span>
          </div>
        </div>
      );
    }
    if (cardType === 'audio' && cardId) {
      const activeState = audioPlayers[cardId] || { isPlaying: false, currentIndex: 0 };
      const currentSeconds = Math.floor((activeState.currentIndex / 10) * 12);
      const timeStr = `0:${currentSeconds < 10 ? '0' + currentSeconds : currentSeconds} / 0:12`;
      const barHeights = [12, 24, 18, 30, 14, 20, 28, 16, 22, 12];
      return (
        <div className="gen-ui-card audio-player-card" id={cardId}>
          <div className="card-glass-specular"></div>
          <div className="player-controls">
            <button className="player-play-btn" onClick={() => toggleAudioCardPlay(cardId)}>
              {activeState.isPlaying ? (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="6" y="4" width="4" height="16"/>
                  <rect x="14" y="4" width="4" height="16"/>
                </svg>
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <polygon points="5 3 19 12 5 21 5 3"/>
                </svg>
              )}
            </button>
            <div className="waveform-scrubber">
              {barHeights.map((h, i) => (
                <span
                  key={i}
                  className={`wave-bar ${i < activeState.currentIndex ? 'active' : ''}`}
                  style={{ height: `${h}px` }}
                  onClick={() => scrubAudioCard(cardId, i)}
                ></span>
              ))}
            </div>
            <span className="player-time">{timeStr}</span>
          </div>
        </div>
      );
    }
    if (cardType === 'code') {
      const codeString = `.liquid-glass-panel {
  background: oklch(14% 0.01 240 / 0.5);
  backdrop-filter: blur(24px) saturate(180%);
  border: 1px solid oklch(22% 0.015 240 / 0.4);
  border-radius: 24px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
}`;
      return (
        <div className="gen-ui-card code-panel-card">
          <div className="card-glass-specular"></div>
          <div className="code-card-header">
            <div className="code-title-row">
              <span className="card-icon" style={{ width: '20px', height: '20px' }}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                  <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
                  <polyline points="14 2 14 8 20 8"/>
                </svg>
              </span>
              <span className="code-filename">liquid-glass.css</span>
            </div>
            <button className="btn-copy-code" onClick={(e) => copyCodeToClipboard(e, codeString)}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
              </svg>
              <span>Copy</span>
            </button>
          </div>
          <pre className="code-content-view">
            <code>
              <span className="keyword">.liquid-glass-panel</span> {'{\n'}
              {'  '}<span className="property">background</span>: <span className="value">oklch(14% 0.01 240 / 0.5)</span>;{'\n'}
              {'  '}<span className="property">backdrop-filter</span>: <span className="value">blur(24px) saturate(180%)</span>;{'\n'}
              {'  '}<span className="property">border</span>: <span className="value">1px solid oklch(22% 0.015 240 / 0.4)</span>;{'\n'}
              {'  '}<span className="property">border-radius</span>: <span className="value">24px</span>;{'\n'}
              {'  '}<span className="property">box-shadow</span>: <span className="value">0 10px 30px rgba(0, 0, 0, 0.2)</span>;{'\n'}
              {'}'}
            </code>
          </pre>
        </div>
      );
    }
    return null;
  };

  return (
    <IconContext.Provider value={{ color: 'currentColor', weight: 'duotone', size: 24, mirrored: false }}>
      {/* Background blobs */}
      <div className="glow-bg">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
        <div className="blob blob-3"></div>
      </div>
      <div className="noise-overlay"></div>

      <div className="app-container">
        
        {/* ==================== LEFT SIDEBAR OVERHAUL ==================== */}
        <aside
          className={`sidebar-panel ${sidebarExpanded && !isStreamingMode ? 'expanded' : ''}`}
          onMouseLeave={() => { if (!isStreamingMode) setSidebarExpanded(false); }}
        >
          <div className="card-glass-specular"></div>

          {/* Logo Area — hover peeks the rail open (offline only); click refreshes. */}
          <div
            className="sidebar-logo-container"
            title={language === 'en' ? 'Refresh' : '刷新页面'}
            onMouseEnter={() => { if (!isStreamingMode) setSidebarExpanded(true); }}
            onClick={() => window.location.reload()}
            style={{ cursor: 'pointer' }}
          >
            <div className="sidebar-logo-icon">
              <svg width="40" height="40" viewBox="0 0 100 100" fill="none">
                <rect x="2" y="2" width="96" height="96" rx="22" fill="var(--logo-glass-fill)" stroke="url(#sidebar-squircle-border-grad)" strokeWidth={1.5} />
                <rect x="3.5" y="3.5" width="93" height="93" rx="20.5" fill="none" stroke="var(--logo-inner-border)" strokeWidth={1} />
                <g filter="url(#sidebar-aura-blur)" opacity="var(--logo-aura-opacity)">
                  <circle cx="35" cy="40" r="18" fill="#8a4be6" />
                  <circle cx="65" cy="60" r="16" fill="#4ade80" />
                  <circle cx="50" cy="50" r="14" fill="#38bdf8" />
                </g>
                <rect x="3" y="3" width="94" height="94" rx="21" fill="rgba(255, 255, 255, 0.01)" opacity="0.8" />
                <path d="M4,4 L96,4 L4,96 Z" fill="var(--logo-glass-sheen)" clipPath="url(#sidebar-squircle-clip)" />
                {/* Liquid Glass top-edge specular cap (Icon Composer-style top
                    light), clipped to the squircle so nothing pokes past corners */}
                <rect x="8" y="5.5" width="84" height="26" rx="14" fill="url(#sidebar-top-specular)" clipPath="url(#sidebar-squircle-clip)" />
                {/* Glyph safe zone: the mark sits at ~74% of the tile, centered —
                    HIG asks for breathing room at a Liquid Glass icon's edges */}
                <g transform="translate(13, 36.9) scale(0.68)">
                  <path fill-rule="evenodd" clip-rule="evenodd" d="M31.7897 19.1844C29.6758 16.4326 27.8676 13.8021 26.265 11.4667C23.218 7.03328 21.5891 5.76809 19.6544 5.05642C16.3754 3.85976 11.9683 8.56206 3.59171 31.4937C2.69553 33.9503 2.61118 35.1048 3.13307 35.2946C3.56008 35.4527 4.15577 35.4422 4.88853 35.2893C3.70241 35.6267 2.75352 35.711 2.08929 35.4738C0.992787 35.0837 0.829367 33.5444 1.93114 30.4394C10.3552 6.66954 14.3617 -0.278483 18.3259 1.12905C20.5664 1.92506 22.9597 5.34109 26.1596 9.90106C27.9994 12.5263 30.0975 15.5206 32.6016 18.6256L31.7844 19.1791L31.7897 19.1844ZM17.9306 0.0272717C18.6106 -0.0518029 19.2907 0.0430867 19.9812 0.296126C22.791 1.3241 25.2951 4.9668 28.6425 9.8378C30.2504 12.1784 32.0638 14.8195 34.1725 17.5608L33.4134 18.0774C30.9041 14.9671 28.8007 11.9675 26.9609 9.34226C23.6556 4.63469 21.1832 1.10796 18.6528 0.211779C18.4103 0.127433 18.1731 0.0641732 17.9306 0.0272717ZM77.1996 19.1844C79.3136 16.4326 81.1217 13.8021 82.7243 11.4667C85.7713 7.03328 87.4003 5.76809 89.335 5.05642C92.6139 3.85976 97.021 8.56206 105.398 31.4937C106.294 33.9503 106.378 35.1048 105.856 35.2946C105.429 35.4527 104.834 35.4422 104.101 35.2893C105.287 35.6267 106.236 35.711 106.895 35.4738C107.991 35.0837 108.155 33.5444 107.053 30.4394C98.6341 6.66954 94.6277 -0.278483 90.6634 1.12905C88.423 1.92506 86.0296 5.34109 82.8298 9.90106C80.99 12.5263 78.8866 15.5206 76.3878 18.6256L77.2049 19.1791L77.1996 19.1844ZM91.0588 0.0272717C90.3787 -0.0518029 89.6987 0.0430867 89.0081 0.296126C86.1983 1.3241 83.6943 4.9668 80.3468 9.8378C78.739 12.1784 76.9255 14.8195 74.8168 17.5608L75.576 18.0774C78.0853 14.9671 80.1887 11.9675 82.0285 9.34226C85.3285 4.62942 87.8009 1.10796 90.3366 0.206508C90.5791 0.122161 90.8163 0.0589016 91.0588 0.0220001V0.0272717ZM54.4947 38.5419C57.1463 38.5419 59.798 37.3821 62.5919 35.0573C65.4122 32.7115 68.7334 29.1004 73.1616 24.0976C73.8627 23.3016 74.5375 22.5108 75.1859 21.7254L75.4864 21.9257C88.6391 30.6871 102.182 39.7017 106.842 37.9989C109.003 37.2082 109.652 35.0204 108.26 30.9507C109.241 34.1559 108.777 35.8428 107.232 36.3911C102.999 37.8935 89.704 28.8526 76.7884 20.0753L75.7763 19.3847C74.9592 20.3811 74.0999 21.3827 73.1932 22.3896C68.7334 27.3344 65.3964 30.898 62.5919 33.1806C59.7558 35.4949 57.1305 36.5545 54.5 36.3489C51.8694 36.5492 49.2441 35.4949 46.408 33.1806C43.6035 30.8927 40.2665 27.3344 35.8067 22.3896C34.9 21.3827 34.0407 20.3811 33.2236 19.3847L32.2114 20.0753C19.2907 28.8526 5.99557 37.8882 1.76245 36.3911C0.212584 35.8428 -0.25132 34.1506 0.734477 30.9507C-0.657237 35.0204 -0.0088244 37.2134 2.15255 37.9989C6.81268 39.7017 20.3555 30.6819 33.5083 21.9257L33.8088 21.7254C34.4572 22.5108 35.1267 23.3068 35.8331 24.0976C40.2612 29.1004 43.5824 32.7167 46.4027 35.0573C49.1967 37.3821 51.8483 38.5419 54.5 38.5419H54.4947ZM54.4947 10.4651C51.9274 10.4546 49.3706 12.073 43.3293 15.3414C41.8375 16.3641 40.1558 17.508 38.2317 18.7785C37.5622 19.2213 36.8874 19.6694 36.2126 20.1175C36.782 20.8081 37.3724 21.4987 37.9839 22.1893C42.3277 27.0972 45.5698 30.6292 48.2372 32.8485C49.2968 33.7289 50.2141 34.43 51.0523 34.9624C49.7871 34.4511 48.4639 33.6076 47.0142 32.4268C44.2677 30.1863 40.9518 26.6491 36.5237 21.7359C35.6486 20.7659 34.8156 19.7959 34.0249 18.8365C34.8789 18.2566 35.8067 17.6293 36.9138 16.8807C38.8274 15.5892 40.4985 14.4241 41.9798 13.3909C48.3269 8.96798 51.2368 6.95948 54.4947 7.33904C57.7473 6.96475 60.6572 8.96798 67.0096 13.3909C68.4962 14.4241 70.162 15.5892 72.0756 16.8807C73.1774 17.624 74.1105 18.2513 74.9645 18.8365C74.1737 19.7959 73.3408 20.7659 72.4657 21.7359C68.0322 26.6491 64.7217 30.1863 61.9751 32.4268C60.5307 33.6076 59.2075 34.4511 57.9371 34.9624C58.7753 34.43 59.6978 33.7289 60.7521 32.8485C63.4196 30.6292 66.6669 27.1024 71.0055 22.1893C71.617 21.4987 72.2074 20.8081 72.7767 20.1175C72.102 19.6694 71.4272 19.2213 70.7577 18.7785C68.8335 17.508 67.1572 16.3588 65.66 15.3414C59.6187 12.073 57.0567 10.4546 54.4947 10.4651Z" fill="url(#sidebar-fluid-glow-grad)" />
                  <path fill-rule="evenodd" clip-rule="evenodd" d="M31.7897 19.1844C29.6758 16.4326 27.8676 13.8021 26.265 11.4667C23.218 7.03328 21.5891 5.76809 19.6544 5.05642C16.3754 3.85976 11.9683 8.56206 3.59171 31.4937C2.69553 33.9503 2.61118 35.1048 3.13307 35.2946C3.56008 35.4527 4.15577 35.4422 4.88853 35.2893L4.15577 35.4422C3.56008 35.4527 3.13307 35.2946 3.59171 31.4937C11.9683 8.56206 16.3754 3.85976 19.6544 5.05642C21.5891 5.76809 23.218 7.03328 26.265 11.4667C27.8676 13.8021 29.6758 16.4326 31.7897 19.1844Z" fill="rgba(255, 255, 255, 0.45)" />
                </g>
                <defs>
                  <linearGradient id="sidebar-fluid-glow-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#8a4be6" />
                    <stop offset="50%" stop-color="#38bdf8" />
                    <stop offset="100%" stop-color="#4ade80" />
                  </linearGradient>
                  <linearGradient id="sidebar-glass-panel-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="rgba(255, 255, 255, 0.15)" />
                    <stop offset="100%" stop-color="rgba(255, 255, 255, 0.02)" />
                  </linearGradient>
                  <linearGradient id="sidebar-glass-border-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="rgba(255, 255, 255, 0.55)" />
                    <stop offset="40%" stop-color="rgba(255, 255, 255, 0.1)" />
                    <stop offset="100%" stop-color="rgba(255, 255, 255, 0.35)" />
                  </linearGradient>
                  <linearGradient id="sidebar-squircle-border-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="var(--logo-glass-border-start)" />
                    <stop offset="100%" stopColor="var(--logo-glass-border-end)" />
                  </linearGradient>
                  <linearGradient id="sidebar-top-specular" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="rgba(255, 255, 255, 0.38)" />
                    <stop offset="100%" stopColor="rgba(255, 255, 255, 0)" />
                  </linearGradient>
                  <clipPath id="sidebar-squircle-clip">
                    <rect x="3" y="3" width="94" height="94" rx="21" />
                  </clipPath>
                  <filter id="sidebar-aura-blur" x="-50%" y="-50%" width="200%" height="200%">
                    <feGaussianBlur stdDeviation="12" result="blur" />
                  </filter>
                  <filter id="sidebar-glass-refraction" x="-30%" y="-30%" width="160%" height="160%">
                    <feGaussianBlur stdDeviation="3" result="blur" />
                    <feSpecularLighting in="blur" specularExponent="35" specularConstant="1.2" lighting-color="#ffffff" result="spec">
                      <fePointLight x="25" y="25" z="40" />
                    </feSpecularLighting>
                    <feComposite in="SourceGraphic" in2="spec" operator="arithmetic" k1="0" k2="1" k3="0.8" k4="0" />
                  </filter>
                </defs>
              </svg>
            </div>
            {sidebarExpanded && (
              <span className="sidebar-logo-text">
                MOSS<span className="sidebar-logo-text-accent">-VL</span>
              </span>
            )}
          </div>
          

          {/* 10% Spacing from Top */}
          <div className="sidebar-spacer-10"></div>

          {/* Mode Switcher Button */}
          <div className="nav-group nav-group-scroll">
            <button
              className={`nav-item-btn ${swapLocked === 'mode' ? 'swap-locked' : ''}`}
              title={isStreamingMode ? (language === 'en' ? "Switch to Chat" : "切换为对话模式") : (language === 'en' ? "Switch to Video Call" : "切换为视频通话")}
              onMouseLeave={() => setSwapLocked((v) => (v === 'mode' ? null : v))}
              onClick={() => {
                if (streamConnected) return;
                stopReadAloud(); // a chat read-aloud must not ride into the live page
                setSwapLocked('mode');
                if (isStreamingMode) {
                  // leaving the live page counts as switching OFF: reset the
                  // TRANSIENT controls — stop any pre-connect camera/screen/mic
                  // preview, drop a held dictation, reset the source to camera,
                  // and close the source menu / screen picker. KEPT (reset only
                  // on New Chat): keyboard draft, uploaded video, transcript log,
                  // queued pre-connect queries.
                  stopWebMedia();
                  resetLiveControls();
                  cancelLiveDictation();
                  setVideoSourceKind('camera');
                  setVideoMenuOpen(false);
                  setScreenPickerOpen(false);
                }
                setIsStreamingMode((prev) => {
                  const nextVal = !prev;
                  if (nextVal) {
                    setSidebarExpanded(false);
                  }
                  return nextVal;
                });
              }}
            >
              {renderModeSwitchIcon()}
              {sidebarExpanded && <span>{isStreamingMode ? (language === 'en' ? 'Video Call' : '流式通话') : (language === 'en' ? 'Chat Mode' : '对话模式')}</span>}
            </button>
            
            <button 
              className="nav-item-btn" 
              title={language === 'en' ? "New Chat" : "开启新对话"}
              onClick={handleNewChat}
            >
              {renderPlusIcon()}
              {sidebarExpanded && <span>{language === 'en' ? 'New Chat' : '新对话'}</span>}
            </button>
            
            {sidebarExpanded && historyOpen ? (
              <div className="nav-item-search-bar">
                <div className="icon-wrapper-dynamic history-search-active" onClick={() => { setSwapLocked('history'); setHistoryOpen(false); }} title="Close Search" style={{ cursor: 'pointer' }}>
                  <ClockCounterClockwise className="icon-dynamic-clock nav-icon" />
                  <MagnifyingGlass className="icon-dynamic-search nav-icon" />
                </div>
                <input 
                  type="text" 
                  value={historyQuery} 
                  onChange={(e) => setHistoryQuery(e.target.value)} 
                  placeholder={language === 'en' ? "Search..." : "搜索..."} 
                  className="sidebar-search-input"
                  autoFocus
                />
                <button className="btn-search-close" onClick={() => { setSwapLocked('history'); setHistoryOpen(false); }} title="Close">
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                  </svg>
                </button>
                
                {/* Index rail: dot-on-a-rail pager for the dropdown list below */}
                <HistoryScrollRail
                  listRef={sidebarHistoryListRef}
                  itemsCount={filteredHistory.length}
                  listId="sidebar-history-list"
                  label={language === 'en' ? 'History list position' : '历史列表位置'}
                />
              </div>
            ) : (
              <button
                className={`nav-item-btn ${historyOpen ? 'active' : ''} ${swapLocked === 'history' ? 'swap-locked' : ''}`}
                title={language === 'en' ? "Chat History" : "历史记录"}
                onMouseLeave={() => setSwapLocked((v) => (v === 'history' ? null : v))}
                onClick={(e) => { setSwapLocked('history'); handleIconClick(e, 'history'); }}
              >
                <div className="icon-wrapper-dynamic">
                  <ClockCounterClockwise className="icon-dynamic-clock nav-icon" />
                  <MagnifyingGlass className="icon-dynamic-search nav-icon" />
                </div>
                {sidebarExpanded && <span>{language === 'en' ? 'History' : '历史记录'}</span>}
              </button>
            )}

            {sidebarExpanded && historyOpen && (
              <div className="sidebar-dropdown-list">
                <div className="sidebar-dropdown-list-scroll" id="sidebar-history-list" ref={sidebarHistoryListRef}>
                  {filteredHistory.map((item) => (
                    <button key={item.conversation_id} className="sidebar-dropdown-item" onClick={() => openHistoryItem(item)}>
                      {renderHistoryItemIcon(item)}
                      <span>{item.title || (language === 'en' ? '(untitled)' : '（无标题）')}</span>
                    </button>
                  ))}
                  {filteredHistory.length === 0 && (
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)', paddingLeft: '28px' }}>
                      {language === 'en' ? 'No history found' : '未找到历史记录'}
                    </span>
                  )}
                </div>
              </div>
            )}

            <button 
              className={`nav-item-btn ${demosOpen ? 'active' : ''}`} 
              title={language === 'en' ? "Interactive Demos" : "交互式演示"}
              onClick={(e) => handleIconClick(e, 'demos')}
            >
              {renderDemosIcon()}
              {sidebarExpanded && <span>{language === 'en' ? 'Demos' : '交互演示'}</span>}
            </button>

            {sidebarExpanded && demosOpen && (
              <div className="sidebar-dropdown-list">
                <button className="sidebar-dropdown-item" onClick={() => submitChat("Show me the latency performance charts.")}>
                  {renderChartIcon()}
                  <span>{language === 'en' ? 'Performance Chart' : '运行性能看板'}</span>
                </button>
                <button className="sidebar-dropdown-item" onClick={() => submitChat("Play a voice/audio sample waveform.")}>
                  {renderWaveIcon()}
                  <span>{language === 'en' ? 'Audio Waveform' : '波形声音播放'}</span>
                </button>
                <button className="sidebar-dropdown-item" onClick={() => submitChat("Give me the Liquid Glass CSS code block.")}>
                  {renderCodeIcon()}
                  <span>{language === 'en' ? 'Glass CSS Styles' : '玻璃微态样式'}</span>
                </button>
                <button className="sidebar-dropdown-item" onClick={() => submitChat("Help. List all system commands.")}>
                  {renderHelpIcon()}
                  <span>{language === 'en' ? 'System Help Commands' : '离线控制帮助'}</span>
                </button>
              </div>
            )}
          </div>

          {/* Push the rest to bottom */}
          <div style={{ flex: 1 }}></div>

          {/* Bottom Settings Group */}
          <div className="nav-group">
            <button
              className={`nav-item-btn ${swapLocked === 'theme' ? 'swap-locked' : ''}`}
              title={theme === 'dark' ? (language === 'en' ? "Switch to Light Mode" : "切换为浅色模式") : (language === 'en' ? "Switch to Dark Mode" : "切换为深色模式")}
              onMouseLeave={() => setSwapLocked((v) => (v === 'theme' ? null : v))}
              onClick={() => { setSwapLocked('theme'); setTheme((prev) => (prev === 'dark' ? 'light' : 'dark')); }}
            >
              {renderThemeIcon()}
              {sidebarExpanded && <span>{theme === 'dark' ? (language === 'en' ? 'Dark Mode' : '深色模式') : (language === 'en' ? 'Light Mode' : '浅色模式')}</span>}
            </button>

            <button 
              className="nav-item-btn" 
              title={language === 'en' ? "切换语言" : "Switch Language"}
              onClick={toggleLanguage}
            >
              {renderLanguageIcon()}
              {sidebarExpanded && <span>{language === 'en' ? 'English (EN)' : '中文 (ZH)'}</span>}
            </button>
          </div>
        </aside>



        {/* ==================== WORKSPACE CONTAINER ==================== */}
        <div className="workspace-container">
          
          {/* ==================== WORKSPACE 1: OFFLINE CHAT ==================== */}
          <div id="chat-workspace" className="workspace-pane">
            <div className={`main-chat-container ${dialogueStarted ? 'started' : 'initial'}`}>
              
              <div className={`chat-empty-greeting ${!showGreeting ? 'fade-out' : ''}`}>
                <h2>
                  <span className="gradient-text-fill">{displayedText}</span>
                  <span className="typing-cursor"></span>
                </h2>
              </div>

              {chatMessages.length > 0 && (
                <div className="chat-messages-container" id="chat-messages">
                  {chatMessages.map((m, idx) => (
                    <div key={m.id} className={`chat-bubble ${m.sender === 'user' ? 'user-bubble' : 'ai-bubble'}`}>
                      <div className="bubble-meta">
                        {m.sender === 'user' ? (language === 'en' ? 'You' : '您') : 'MOSS'}
                        {m.latency && <span className="latency-pill" style={{ marginLeft: '8px' }}>{m.latency}</span>}
                      </div>
                      {(m.text || m.isTyping || (m.images?.length ?? 0) > 0 || (m.videos?.length ?? 0) > 0) && (
                        /* ONE bubble for the whole model turn: while thinking it
                           is the empty aurora vessel (.vessel-live min-box); type
                           streams into the SAME element; settle stops the ring.
                           User turns keep the button-fill glass card. */
                        <div
                          className={`bubble-content${m.sender === 'user' ? ' glass-card' : ''}${m.isTyping || m.isStreaming ? ' vessel-live' : ''}`}
                          {...(m.isTyping ? { role: 'status', 'aria-label': '模型推理中' } : {})}
                        >
                          {((m.images && m.images.length > 0) || (m.videos && m.videos.length > 0)) && (
                            <div className="bubble-attachments">
                              {(m.images ?? []).map((src, i) => (
                                <img
                                  key={`i${i}`}
                                  src={src}
                                  className="bubble-attachment-img media-clickable"
                                  alt=""
                                  onClick={() => openMediaViewer('image', src)}
                                />
                              ))}
                              {(m.videos ?? []).map((src, i) => (
                                /* inline preview only — click opens the viewer,
                                   where the CAS blob plays with controls
                                   (Range/206 → native seeking) */
                                <span key={`v${i}`} className="bubble-video-wrap media-clickable" onClick={() => openMediaViewer('video', src)}>
                                  <video src={src} className="bubble-attachment-img" preload="metadata" muted playsInline />
                                  <span className="bubble-video-badge" aria-hidden="true">▶</span>
                                </span>
                              ))}
                            </div>
                          )}
                          <p style={{ whiteSpace: 'pre-wrap' }}>
                            {m.isStreaming && m.text.length > 2 ? (
                              <>
                                {m.text.slice(0, -2)}
                                <span className="shimmer-tail">{m.text.slice(-2)}</span>
                              </>
                            ) : (
                              m.text
                            )}
                          </p>
                          {/* the newest model turn carries the halo: LIVE ring
                              while thinking AND streaming, fading to the
                              TR/BL-corner static remnant once it settles */}
                          {m.halo && idx === chatMessages.length - 1 && (
                            <AuroraStaticHalo
                              orbTheme={orbTheme}
                              dark={theme !== 'light'}
                              animated={!!(m.isTyping || m.isStreaming)}
                            />
                          )}
                        </div>
                      )}
                      {/* cards are glass already — siblings BELOW the bubble, never nested
                          (glass-on-glass doubles the blur and breaks the radius ladder) */}
                      {m.generativeCard && renderGenerativeCard(m.generativeCard, m.cardId)}
                      {/* bubble actions: copy everywhere; edit / regenerate only on
                          the LATEST turn; read-aloud on model replies (POST /api/tts) */}
                      {!m.isTyping && !m.isStreaming && !!m.text && (
                        <div className="msg-actions">
                          <button
                            className="msg-action-btn"
                            onClick={() => void copyMessage(m.id, m.text)}
                            title={language === 'en' ? 'Copy' : '复制'}
                          >
                            {copiedMsgId === m.id ? <Check size={13} weight="bold" /> : <Copy size={13} />}
                          </button>
                          {m.sender === 'user' && idx === lastUserMsgIdx && !chatInputBlocked && (
                            <button
                              className="msg-action-btn"
                              onClick={editLatestUserMessage}
                              title={language === 'en' ? 'Edit & re-ask' : '编辑并重新提问'}
                            >
                              <PencilSimple size={13} />
                            </button>
                          )}
                          {m.sender === 'ai' && (
                            <button
                              className={`msg-action-btn ${readingMsg?.id === m.id ? 'active' : ''}`}
                              onClick={() => void readAloud(m.id, m.text)}
                              title={readingMsg?.id === m.id
                                ? (language === 'en' ? 'Stop reading' : '停止朗读')
                                : (language === 'en' ? 'Read aloud' : '朗读')}
                            >
                              {readingMsg?.id === m.id
                                ? (readingMsg.phase === 'loading'
                                    ? <CircleNotch size={13} className="spin" />
                                    : <Stop size={13} weight="fill" />)
                                : <SpeakerHigh size={13} />}
                            </button>
                          )}
                          {m.sender === 'ai' && idx === lastAiMsgIdx && !m.generativeCard
                            && lastTurnRef.current && !chatInputBlocked && (
                            <button
                              className="msg-action-btn"
                              onClick={regenerateLatest}
                              title={language === 'en' ? 'Regenerate' : '重新生成'}
                            >
                              <ArrowsClockwise size={13} />
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                  <div ref={chatMessagesEndRef} />
                </div>
              )}

              {/* Chat Input Console */}
              <div className="chat-input-wrapper">
                {attachments.length > 0 && (
                  <div className="attachment-strip">
                    {attachments.map((att) => (
                      <div key={att.id} className="attachment-chip">
                        {att.uploading && att.slow ? (
                          /* slow transfer: the thumbnail IS the progress — a
                             frosted glassy copy underneath, the sharp copy
                             filling it bottom-up with the byte fraction;
                             fully sharp under a light shimmer while the
                             server cuts the real thumbnail */
                          <div
                            className={`chip-fill-stack${att.progress == null ? ' processing' : ''}`}
                            style={{ '--fill': `${Math.round((att.progress ?? 1) * 100)}%` } as React.CSSProperties}
                            role="progressbar"
                            aria-valuenow={att.progress == null ? undefined : Math.round(att.progress * 100)}
                          >
                            <img src={att.previewUrl || FILM_GLYPH} alt="" className="chip-img-frost" aria-hidden="true" />
                            <img
                              src={att.previewUrl || FILM_GLYPH}
                              alt=""
                              className="chip-img-sharp media-clickable"
                              onClick={() => openMediaViewer(att.kind, att.url)}
                            />
                          </div>
                        ) : (
                          /* settled (or within the fast-upload grace window):
                             the plain thumbnail; film glyph when a video has
                             no poster (probe unavailable) */
                          <img
                            src={att.previewUrl || FILM_GLYPH}
                            alt=""
                            className="media-clickable"
                            onClick={() => openMediaViewer(att.kind, att.url)}
                          />
                        )}
                        <button
                          className="attachment-remove"
                          title={language === 'en' ? 'Remove' : '移除'}
                          onClick={() => {
                            attachmentXhrsRef.current.get(att.id)?.abort(); // mid-flight × cancels the transfer
                            setAttachments((prev) => prev.filter((a) => a.id !== att.id));
                          }}
                        >
                          <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round">
                            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                          </svg>
                        </button>
                      </div>
                    ))}
                  </div>
                )}
                <div className="chat-input-bar glass-card">
                  <button
                    className={`input-action-btn ${attachments.length > 0 ? 'has-attachments' : ''}`}
                    title={language === 'en' ? 'Attach images or videos' : '添加图片或视频'}
                    onClick={() => fileInputRef.current?.click()}
                    /* NOT disabled while generating — same rule as the text
                       field above: attachments only STAGE for the next message
                       (they even upload in the background); submitChat's
                       chatInputBlocked guard is what prevents a concurrent
                       send. Disabling here made mid-stream composing lopsided:
                       you could type but not attach. */
                  >
                    <Paperclip className="input-icon" size={20} />
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*,video/mp4,video/webm,video/quicktime"
                    multiple
                    hidden
                    onChange={(e) => {
                      void handleFilesChosen(e.target.files);
                      e.target.value = ''; // allow re-picking the same file
                    }}
                  />

                  <input
                    ref={chatInputRef}
                    type="text"
                    id="chat-input-field"
                    placeholder={language === 'en' ? "Type a message or click mic to dictate..." : "输入消息或点击麦克风开始听写..."}
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={handleInputKeyDown}
                    // Intentionally NOT disabled while generating: disabling blurs
                    // the field (losing the cursor), and the user should be able to
                    // compose the next message mid-stream. submitChat's own
                    // `if (chatInputBlocked) return` guard prevents a concurrent send.
                  />

                  <button
                    className={`input-action-btn mic-btn ${isDictating ? 'recording' : ''}`}
                    title="Voice input (Speech-To-Text)"
                    onClick={handleMicClick}
                  >
                    <Microphone className="input-icon" size={20} />
                    <div className="mic-ripple-ring"></div>
                  </button>

                  {/* Send ↔ Stop morph: while the model streams, the same
                      round button becomes ⏹ with a breathing ring — one glass
                      object, two states (the house morph pattern). Canned demo
                      cards resolve in ~0.6 s; aborting them is a no-op. */}
                  <button
                    className={`send-message-btn ${chatInputBlocked ? 'generating' : ''}`}
                    disabled={!chatInputBlocked && attachments.some((a) => a.uploading)}
                    title={chatInputBlocked
                      ? (language === 'en' ? 'Stop generating' : '停止生成')
                      : attachments.some((a) => a.uploading)
                        ? (language === 'en' ? 'Waiting for attachments to finish uploading…' : '附件上传中，请稍候…')
                        : (language === 'en' ? 'Send message' : '发送')}
                    onClick={chatInputBlocked ? stopChatGeneration : handleSendClick}
                  >
                    {chatInputBlocked
                      ? <Stop size={14} weight="fill" />
                      : <PaperPlaneTilt className="input-icon" size={18} />}
                  </button>
                </div>
              </div>
            </div>

          </div>

          {/* ==================== WORKSPACE 2: STREAMING VIDEO-CHAT ==================== */}
          <div id="streaming-workspace" className="workspace-pane">
            
            <div className="main-video-container">
              
              <div
                className={`ai-stream-box glass-card theme-${orbTheme} ${stageChromeHidden ? 'chrome-hidden' : ''}`}
                data-focus={focusedFeed}
                data-session={streamConnected || subtitleDemoActive ? 'live' : 'idle'}
                onClick={(e) => { if (e.target === e.currentTarget) toggleStageChrome(); }}
              >
                <div className="card-glass-specular"></div>
                
                {/* The status pill is gone (the orb + captions carry state);
                    only its narrow-viewport settings toggle survives, as a
                    standalone bead — desktop CSS keeps it hidden. */}
                <button
                  className="panel-toggle-btn stage-panel-toggle"
                  onClick={() => setShowSessionPanel(!showSessionPanel)}
                  title={language === 'en' ? "Settings & Logs" : "设置与日志"}
                >
                  <Sliders size={18} />
                </button>

                {/* Voice Visualizer Canvas */}
                <div 
                  className="streaming-visualizer-container"
                  onClick={() => { if (focusedFeed === 'user') setFocusedFeed('model'); }}
                >
                  <canvas ref={streamingCanvasRef} id="streaming-voice-canvas" width="480" height="480"></canvas>
                  <canvas ref={glassShellRef} id="glass-shell-canvas" width="480" height="480"></canvas>
                  <div className="avatar-orb-glow" ref={orbGlowRef}></div>
                </div>

                {/* (The stage subtitle bar is retired — the model's rolling
                    caption renders as the provisional tail bubble in the
                    session transcript instead.) */}

                {/* Local user camera feed */}
                <div 
                  className="user-pip-box glass-card" 
                  id="user-pip"
                  onClick={() => {
                    if (focusedFeed === 'model') setFocusedFeed('user');
                    else toggleStageChrome(); // fullscreen video = the stage background on phones
                  }}
                >
                  <div className="card-glass-specular"></div>
                  
                  <video
                    ref={localVideoRef}
                    id="local-video"
                    autoPlay={!mediaFile} /* a file waits paused on frame 1 until ▶ */
                    playsInline
                    muted
                    /* native controls stay OFF for now (no scrub bar over the
                       stage): ▶ is the call button; a PURE file session still
                       hangs up when the file ends, but a multi-source session
                       (any source change since connect) stays live — the feed
                       just freezes and the user can ask about it or swap */
                    onEnded={() => {
                      if (!mediaFileRef.current) return;
                      if (!session.connected) return;
                      if (sourceChangedMidSessionRef.current) {
                        // multi-source session: keep the call up on the frozen
                        // last frame (the media clock stops → no more frames);
                        // the end announces like a source change — the SAME
                        // dedicated transcript chip as the start, not a caption
                        announceSource('none', { name: mediaFileRef.current.name });
                        addLog(language === 'en'
                          ? 'Video finished — session stays live.'
                          : '视频播放完毕 — 会话保持连线。', 'live');
                        return;
                      }
                      session.disconnect(); // teardown also stops the frame sampler
                      stopWebMedia();
                      unloadMediaFile(); // a session end unloads the media source
                      resetLiveControls(); // keyboard/voice back to defaults
                      targetIntensityRef.current = 0.15;
                      setCaptionsText(language === 'en'
                        ? 'Video finished — session disconnected.'
                        : '视频播放完毕，已停止推流并断开会话。');
                      addLog(language === 'en'
                        ? 'Video finished — streaming stopped, session disconnected.'
                        : '视频播放完毕 — 已自动停止推流并断开会话。', 'live');
                    }}
                    className={`local-camera-feed ${(!videoEnabled && !mediaFile) || mediaFile?.kind === 'image' ? 'hidden' : ''} ${mediaFile ? 'is-file' : ''} ${!mediaFile && videoEnabled && videoSourceKind === 'screen' ? 'is-screen' : ''} ${cameraFacing === 'environment' ? 'facing-environment' : ''}`}
                  ></video>
                  {mediaFile?.kind === 'image' && (
                    /* a staged still image takes the stage in place of the
                       <video> — same feed styling, screen-share letterboxing */
                    <img
                      className="local-camera-feed is-file is-screen"
                      src={mediaFile.url}
                      alt={mediaFile.name}
                    />
                  )}

                  {/* CAS transfer still running: the file's first frame is
                      already the backdrop — a glass puck floats centered on it
                      (determinate ring → indeterminate while the server cuts
                      the poster), mirroring the chat chips. */}
                  {mediaFile && liveUploadPct !== null && (
                    <div className="pip-upload-overlay">
                      <GlassProgress progress={liveUploadPct === 'processing' ? null : liveUploadPct} size={56} />
                    </div>
                  )}

                  {!videoEnabled && !mediaFile && (
                    <div className="user-avatar-fallback" id="user-fallback-view">
                      <div className="fallback-icon">
                        <User size={24} weight="duotone" />
                      </div>
                      <span className="fallback-name">{language === 'en' ? 'You' : '您'}</span>
                    </div>
                  )}

                  {/* video-file beads: the viewer's macOS-geometry glass dots —
                      × exits the mode (idle only), i reveals the file info */}
                  {mediaFile && (
                    <div className="viewer-dots pip-dots" onClick={(e) => e.stopPropagation()}>
                      <button
                        className="viewer-dot"
                        disabled={streamConnected}
                        title={streamConnected
                          ? (language === 'en' ? 'Hang up to remove the video' : '挂断后可移除视频')
                          : (language === 'en' ? 'Remove video, back to camera' : '移除视频，返回摄像头模式')}
                        onClick={exitVideoFileMode}
                      >
                        <span className="viewer-dot-glyph">×</span>
                      </button>
                      <div
                        className="viewer-info-zone"
                        onMouseEnter={pipInfoZoneEnter}
                        onMouseLeave={pipInfoZoneLeave}
                      >
                        <button
                          className="viewer-dot"
                          onClick={() => setPipInfoOpen((v) => !v)}
                          title={language === 'en' ? 'Video info' : '视频信息'}
                        >
                          <span className="viewer-dot-glyph">i</span>
                        </button>
                        {pipInfoOpen && (
                          <div className="media-info-anchor">
                            <div className="media-info-card" role="tooltip">
                              {pipInfoRows().map(([k, v]) => (
                                <div key={k} className="media-info-row">
                                  <span className="media-info-key">{k}</span>
                                  <span className="media-info-val">{v}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  <div className="pip-label" title={mediaFile?.name}>
                    {mediaFile
                      ? mediaFile.kind === 'image'
                        ? (language === 'en' ? 'Image' : '图片')
                        : (language === 'en' ? 'Video File' : '视频文件')
                      : (language === 'en' ? 'Camera Feed' : '本地画面')}
                  </div>
                </div>
              </div>

              {/* Call Controls bar */}
              <div className="call-controls-wrapper">
                <div className="call-controls-bar glass-card">
                  {/* ① INPUT at the LEFT end — one button, two gestures:
                      TAP toggles the text field (the pill itself morphs — the
                      field telescopes out inside the same glass object);
                      HOLD (pointer / Space / V) talks — in-call PTT opens the
                      turn and barge-ins any reply, release commits to ASR;
                      disconnected the hold dictates into the asr-pending
                      bubble and release queues the utterance. Voice and typing
                      mix freely (both feed the same lane); works disconnected
                      too — queued queries merge into one request at connect. */}
                  <button
                    className={`call-control-btn ptt-btn input-btn ${pttHeld || dictHeld ? 'ptt-holding' : ''} ${session.listening || isDictating ? 'active' : ''} ${liveTextOpen ? 'text-open' : ''}`}
                    id="btn-live-input"
                    aria-pressed={pttHeld || dictHeld}
                    aria-expanded={liveTextOpen}
                    title={streamConnected
                      ? (language === 'en' ? 'Tap to type · hold to talk (mouse, Space, or V)' : '点按打字 · 按住说话（鼠标、空格键或 V 键）')
                      : (language === 'en' ? 'Tap to type · hold to talk — queued queries send together at connect' : '点按打字 · 按住说话 — 排队消息将在连线后合并发送')}
                    onPointerDown={(e) => {
                      e.preventDefault();
                      // capture keeps the release on this button even if the
                      // pointer drifts; a failed capture (stale/synthetic
                      // pointer id throws) must not abort the arm timer
                      try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* noop */ }
                      // auto (server-VAD) mode has no hold semantics in-call —
                      // the mic is already open; a press is always a tap there
                      if (streamConnected && captureMode === 'auto') return;
                      holdArmTimerRef.current = setTimeout(() => {
                        holdArmTimerRef.current = null;
                        holdArmedRef.current = true;
                        beginHoldMic(); // the hold ARMS here: icon flips, PTT/dictation starts
                      }, HOLD_ARM_MS);
                    }}
                    onPointerUp={() => {
                      if (holdArmTimerRef.current) {
                        clearTimeout(holdArmTimerRef.current);
                        holdArmTimerRef.current = null;
                      }
                      if (holdArmedRef.current) {
                        holdArmedRef.current = false;
                        endHoldMic(); // a completed talk-hold never toggles the pill
                        return;
                      }
                      setLiveTextOpen((v) => !v); // pure tap
                    }}
                    onPointerCancel={() => {
                      if (holdArmTimerRef.current) {
                        clearTimeout(holdArmTimerRef.current);
                        holdArmTimerRef.current = null;
                      }
                      if (holdArmedRef.current) {
                        holdArmedRef.current = false;
                        endHoldMic();
                      }
                    }}
                    onContextMenu={(e) => e.preventDefault()}
                  >
                    <span className="ptt-wave" aria-hidden="true"></span>
                    <span className="ptt-wave" aria-hidden="true"></span>
                    <span className="ptt-wave" aria-hidden="true"></span>
                    {pttHeld || dictHeld
                      ? <Microphone size={22} weight="duotone" />
                      : <Keyboard size={22} weight="duotone" />}
                  </button>
                  <div className={`live-text-wrap ${liveTextOpen ? 'open' : ''}`}>
                    <input
                      ref={liveTextInputRef}
                      type="text"
                      className="live-text-field"
                      placeholder={streamConnected
                        ? (language === 'en' ? 'Type a message, Enter to send…' : '输入消息，回车发送…')
                        : (language === 'en' ? 'Queued — sends as one request at connect…' : '输入消息，连线后合并发送…')}
                      value={liveTextDraft}
                      onChange={(e) => setLiveTextDraft(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') sendLiveText();
                        else if (e.key === 'Escape') setLiveTextOpen(false);
                      }}
                      tabIndex={liveTextOpen ? 0 : -1}
                    />
                    <button
                      className="live-text-send"
                      onClick={sendLiveText}
                      disabled={!liveTextDraft.trim()}
                      title={language === 'en' ? 'Send' : '发送'}
                      tabIndex={liveTextOpen ? 0 : -1}
                    >
                      <PaperPlaneTilt size={16} weight="duotone" />
                    </button>
                    <div className="divider"></div>
                  </div>
                  {streamConnected && captureMode === 'auto' && (
                    <button
                      /* connected in auto (server-VAD) mode the mic is always
                         open — a dedicated mute bead appears (hold has no
                         meaning there; the Input button is tap-only) */
                      className={`call-control-btn ${micMuted ? 'muted' : 'active'}`}
                      id="btn-toggle-mic"
                      title={micMuted ? (language === 'en' ? "Unmute Microphone" : "开启麦克风") : (language === 'en' ? "Mute Microphone" : "禁用麦克风")}
                      onClick={handleToggleMic}
                    >
                      {micMuted ? <MicrophoneSlash size={22} weight="duotone" /> : <Microphone size={22} weight="duotone" />}
                    </button>
                  )}

                  {/* ② MEDIA SOURCE bead: opens a glass menu (camera / screen
                      share / upload file) that morphs out of the bead; the
                      bead itself wears the LIVE source glyph in the violet
                      source-live state — camera, monitor, video file, or
                      image. For a picked file the violet marks a COMPLETED
                      upload (while bytes move the bead stays neutral; the PIP
                      puck carries the progress). */}
                  <div className="video-source-anchor" ref={videoMenuAnchorRef}>
                    <button
                      className={`call-control-btn ${activeSource !== 'none' && !(mediaFile && liveUploadPct !== null) ? 'source-live' : ''} ${activeSource === 'none' ? 'video-off' : ''}`}
                      id="btn-toggle-video"
                      aria-haspopup="menu"
                      aria-expanded={videoMenuOpen}
                      title={mediaFile
                        ? (language === 'en' ? `Media source (current: ${mediaFile.name})` : `媒体源（当前：${mediaFile.name}）`)
                        : (language === 'en' ? 'Media source — camera, screen share, or file' : '媒体源 — 摄像头、屏幕共享或文件')}
                      onClick={() => setVideoMenuOpen((v) => !v)}
                    >
                      {activeSource === 'camera' ? <VideoCamera size={22} weight="duotone" />
                        : activeSource === 'screen' ? <Monitor size={22} weight="duotone" />
                        : activeSource === 'file' ? <FileVideo size={22} weight="duotone" />
                        : activeSource === 'image' ? <FileImage size={22} weight="duotone" />
                        : <VideoCameraSlash size={22} weight="duotone" />}
                    </button>
                    {videoMenuOpen && (
                      <div className="video-source-menu" role="menu">
                        <button
                          className={`video-source-option ${activeSource === 'camera' ? 'active' : ''}`}
                          role="menuitem"
                          onClick={() => void selectVideoSource('camera')}
                        >
                          <VideoCamera size={18} weight="duotone" />
                          <span>{language === 'en' ? 'Camera' : '摄像头'}</span>
                          {activeSource === 'camera' && <span className="video-source-live-dot" aria-hidden="true" />}
                        </button>
                        <button
                          className={`video-source-option ${activeSource === 'screen' ? 'active' : ''}`}
                          role="menuitem"
                          onClick={() => void selectVideoSource('screen')}
                        >
                          <Monitor size={18} weight="duotone" />
                          <span>{language === 'en' ? 'Screen share' : '屏幕共享'}</span>
                          {activeSource === 'screen' && <span className="video-source-live-dot" aria-hidden="true" />}
                        </button>
                        <button
                          className={`video-source-option ${activeSource === 'file' || activeSource === 'image' ? 'active' : ''}`}
                          role="menuitem"
                          onClick={() => {
                            setVideoMenuOpen(false);
                            // re-picking the ACTIVE row switches the source OFF
                            // (same rule as the Camera/Screen rows) — back to
                            // the no-media state; upload again to load another
                            if (activeSource === 'file' || activeSource === 'image') {
                              const wasFileClock = mediaFileRef.current?.kind === 'video';
                              unloadMediaFile();
                              if (sessionConnectedRef.current) {
                                if (wasFileClock) session.setSamplerClock('live');
                                announceSource('none');
                              }
                              return;
                            }
                            mediaFileInputRef.current?.click();
                          }}
                        >
                          <UploadSimple size={18} weight="duotone" />
                          <span>{activeSource === 'file' || activeSource === 'image'
                            ? (language === 'en' ? 'Turn off file' : '关闭文件')
                            : (language === 'en' ? 'Upload file' : '上传文件')}</span>
                          {(activeSource === 'file' || activeSource === 'image') && <span className="video-source-live-dot" aria-hidden="true" />}
                        </button>
                        {streamConnected && (
                          /* mid-session file pick re-bases the timeline: the
                             session clock keeps running monotonically and the
                             model is told where the file's 0:00 landed on it */
                          <div className="video-source-hint">
                            {language === 'en'
                              ? 'A file picked now joins the session timeline — the model sees it start at the current session time.'
                              : '此时选择的文件将接入会话时间轴 — 模型将从当前会话时间开始观看。'}
                          </div>
                        )}
                      </div>
                    )}
                    <input
                      ref={mediaFileInputRef}
                      type="file"
                      accept="video/*,image/*"
                      hidden
                      onChange={(e) => {
                        const f = e.target.files?.[0];
                        if (f) void loadMediaFile(f);
                        e.target.value = ''; // allow re-picking the same file
                      }}
                    />
                  </div>

                  {!mediaFile && videoEnabled && videoSourceKind === 'camera' && hasMultipleCameras && (
                    <button
                      className="call-control-btn camera-flip-btn"
                      id="btn-flip-camera"
                      title={cameraFacing === 'user'
                        ? (language === 'en' ? 'Switch to Rear Camera' : '切换后置摄像头')
                        : (language === 'en' ? 'Switch to Front Camera' : '切换前置摄像头')}
                      onClick={flipCamera}
                    >
                      <CameraRotate size={22} weight="duotone" />
                    </button>
                  )}

                  <button
                    className={`call-control-btn call-action-btn ${streamConnected ? 'end-call' : 'start-call'}`}
                    id="btn-connect-stream"
                    /* ONE control, ONE meaning: start/stop the SESSION — the
                       glyph never changes with the media source (sources come
                       and go under a session via the Media control) */
                    title={streamConnected
                      ? (language === 'en' ? "Hang Up (ends the session)" : "挂断连线（结束会话）")
                      : (language === 'en' ? "Connect Session" : "连接实时通话")}
                    onClick={handleConnectStream}
                  >
                    {streamConnected
                      ? <PhoneDisconnect size={22} weight="duotone" />
                      : <PhoneCall size={22} weight="duotone" />}
                  </button>
                </div>
              </div>
            </div>

            {/* Overhauled Session Config & Log Panel */}
            <aside className={`session-panel glass-card ${showSessionPanel ? 'open' : ''}`} id="streaming-sidebar">
              <div className="card-glass-specular"></div>
              
              <div className="panel-header">
                <h3>{language === 'en' ? 'Session Controls' : '会话控制面板'}</h3>
                <p>{language === 'en' ? 'Voice model pipelines' : '实时语音传输通道配置'}</p>
                {/* Touch dismissal for the slide-over / bottom-sheet presentations;
                    display: none on desktop where the panel is a fixed column */}
                <button
                  className="panel-close-btn"
                  onClick={() => setShowSessionPanel(false)}
                  title={language === 'en' ? 'Close panel' : '关闭面板'}
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                  </svg>
                </button>
              </div>

              <div className="panel-content">
                {/* System prompt — applies at session START (KV prefill), so the
                    field locks while a session is live; edits take the next call */}
                <div className="panel-group-title">{language === 'en' ? 'System Prompt' : '系统提示词'}</div>
                <div className={`panel-group-card prompt-card${streamConnected ? ' locked' : ''}`}>
                  <textarea
                    className="liquid-textarea prompt-field"
                    rows={5}
                    value={systemPrompt}
                    readOnly={streamConnected}
                    spellCheck={false}
                    onChange={(e) => setSystemPrompt(e.target.value)}
                    aria-label={language === 'en' ? 'System prompt' : '系统提示词'}
                    title={streamConnected
                      ? (language === 'en' ? 'Locked while the session is live' : '会话进行中锁定 — 修改在下次连线生效')
                      : undefined}
                  />
                </div>

                {/* Scrolling Transcript Log — with the system prompt, the
                    panel's centerpiece: it takes all remaining height */}
                <div className="log-title-row">
                  <span className="log-title">{language === 'en' ? 'Session Transcript' : '会话实时转写日志'}</span>
                  <div className="log-title-actions">
                    {/* Context commands (/compact, /clear) ride the session
                        socket as plain text.input — the server owns the
                        reseat and busy-locks other events while one runs, so
                        both buttons stay disabled until it's done. The
                        running/completed/failed report lands in the log
                        itself via context.command → onMemoryNotice. */}
                    <button
                      type="button"
                      className={`context-cmd-btn${session.contextCommand === '/compact' ? ' running' : ''}`}
                      disabled={!streamConnected || session.contextCommand !== null}
                      onClick={() => session.sendText('/compact')}
                      title={language === 'en'
                        ? 'Summarize & compact the context (/compact)'
                        : '总结并压缩上下文（/compact）'}
                      aria-label={language === 'en' ? 'Compact context' : '总结上下文'}
                    >
                      {session.contextCommand === '/compact'
                        ? <CircleNotch size={14} weight="bold" />
                        : <ListBullets size={14} weight="bold" />}
                    </button>
                    <button
                      type="button"
                      className={`context-cmd-btn danger${session.contextCommand === '/clear' ? ' running' : ''}`}
                      disabled={!streamConnected || session.contextCommand !== null}
                      onClick={() => session.sendText('/clear')}
                      title={language === 'en'
                        ? 'Clear the context and restart (/clear)'
                        : '清空上下文并重启会话上下文（/clear）'}
                      aria-label={language === 'en' ? 'Clear context' : '清空上下文'}
                    >
                      {session.contextCommand === '/clear'
                        ? <CircleNotch size={14} weight="bold" />
                        : <ArrowsClockwise size={14} weight="bold" />}
                    </button>
                    {liveMediaTime !== null && (
                      /* glassy tag: how long THIS SESSION has lasted (wall clock
                         from connect) — independent of the media source; frozen
                         at the last tick once the session ends */
                      <span
                        className="media-time-tag"
                        title={language === 'en' ? 'session time' : '会话时长'}
                      >
                        <span className="media-time-glyph" />
                        {formatMediaTime(liveMediaTime.s)}
                      </span>
                    )}
                  </div>
                </div>
                <div
                  className="session-conversation-log"
                  ref={liveLogRef}
                  onScroll={(e) => {
                    const el = e.currentTarget;
                    liveLogStickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
                  }}
                >
                  {streamConversation.map((entry, idx) => (
                    entry.memory ? (
                      /* memory recall: one centered tag, expands on click */
                      <MemoryRecallLogEntry
                        key={idx}
                        items={entry.memory}
                        language={language}
                        formatTs={formatMediaTime}
                      />
                    ) : entry.system ? (
                      /* dedicated source-change bubble: a centered glass chip,
                         no sender label — an event, not an utterance */
                      <div key={idx} className="log-entry system">
                        <div className="log-entry-system-chip">{entry.text}</div>
                      </div>
                    ) : (
                    <div
                      key={idx}
                      className={`log-entry ${entry.sender}${entry.queued ? ' queued' : ''}${entry.capped ? ' capped' : ''}${entry.cont ? ' cont' : ''}`}
                    >
                      <div className="log-entry-meta">
                        <span className="log-entry-speaker">
                          {entry.cont ? (
                            /* follow-up of a split sentence: one water-bead in
                               place of the repeated sender label */
                            <span
                              className="log-entry-cont-glyph"
                              title={language === 'en' ? 'continued' : '接上句'}
                              aria-label={language === 'en' ? 'continued' : '接上句'}
                            />
                          ) : entry.sender === 'user' ? (
                            language === 'en' ? 'You' : '您'
                          ) : (
                            'MOSS'
                          )}
                        </span>
                        <span className="log-entry-time">
                          {entry.queued ? (language === 'en' ? 'queued' : '待发送') : entry.time}
                        </span>
                      </div>
                      <div className="log-entry-text">{entry.text}</div>
                    </div>
                    )
                  ))}
                  {captionsText && (streamConnected || subtitleDemoActive || session.error) &&
                    streamConversation[streamConversation.length - 1]?.text !== captionsText && (
                    /* the stage subtitle bar is retired — the model's current
                       sentence reveals HERE as a provisional tail bubble;
                       each sentence's committed bubble replaces it as it
                       closes. (The caption lingers on a just-closed sentence
                       until the next one starts — the last-entry comparison
                       hides that duplicate beat.) A fatal session.error also
                       reveals here — it lands precisely when the connection
                       (and with it the normal gate) is gone. */
                    <div className={`log-entry ${streamSpeaker} caption-pending`}>
                      <div className="log-entry-meta">
                        <span className="log-entry-speaker">
                          {streamSpeaker === 'user' ? (language === 'en' ? 'You' : '您') : 'MOSS'}
                        </span>
                        <span className="log-entry-time">{language === 'en' ? 'speaking…' : '播报中…'}</span>
                      </div>
                      <div className="log-entry-text">{captionsText}</div>
                    </div>
                  )}
                  {liveAsrDraft && (
                    /* rolling ASR hypothesis — replaced by the final turn */
                    <div className="log-entry user asr-pending">
                      <div className="log-entry-meta">
                        <span className="log-entry-speaker">{language === 'en' ? 'You' : '您'}</span>
                        <span className="log-entry-time">{language === 'en' ? 'listening…' : '识别中…'}</span>
                      </div>
                      <div className="log-entry-text">{liveAsrDraft}</div>
                    </div>
                  )}
                  {streamConversation.length === 0 && !liveAsrDraft && (
                    streamConnected ? (
                      /* connected, awaiting the first question — prompt to ask */
                      <div className="live-log-prompt">
                        <span className="live-log-prompt-orb" aria-hidden="true" />
                        <span className="live-log-prompt-text">
                          {language === 'en' ? 'Ask a question' : '请提问'}
                        </span>
                        <span className="live-log-prompt-hint">
                          {language === 'en' ? 'speak or type your question' : '可语音或键盘输入提问'}
                        </span>
                      </div>
                    ) : (
                      <div className="live-log-waiting">
                        {language === 'en' ? 'Awaiting session connection…' : '等待连线开启中…'}
                      </div>
                    )
                  )}
                </div>

                {/* 更多设置: ASR/TTS/visual config folds away at the bottom.
                    Expanding squeezes the log (which keeps a minimum and
                    scrolls internally); overflow rides the panel's own
                    hidden-scrollbar column. */}
                <button
                  className="more-settings-toggle"
                  onClick={() => setMoreSettingsOpen((v) => !v)}
                  aria-expanded={moreSettingsOpen}
                >
                  <span>{language === 'en' ? 'More Settings' : '更多设置'}</span>
                  <CaretDown size={12} weight="bold" className={`more-caret ${moreSettingsOpen ? 'open' : ''}`} />
                </button>
                <div className={`more-settings ${moreSettingsOpen ? 'open' : ''}`}>
                  <div className="more-settings-inner">

                {/* ASR input settings group */}
                <div className="panel-group-title">{language === 'en' ? 'ASR Settings' : '语音识别配置 (ASR)'}</div>
                <div className="panel-group-card">
                  <div className="settings-row-inline">
                    <label>{language === 'en' ? 'Input Language' : '输入语言'}</label>
                    <select
                      className="liquid-select"
                      value={asrLanguage}
                      onChange={(e) => setAsrLanguage(e.target.value)}
                    >
                      {/* SenseVoiceSmall's language set */}
                      <option value="zh">简体中文 (CN)</option>
                      <option value="en">English (US)</option>
                      <option value="yue">粤语 (HK)</option>
                      <option value="ja">日本語 (JP)</option>
                      <option value="ko">한국어 (KR)</option>
                      <option value="auto">Auto Detect</option>
                    </select>
                  </div>

                  <div className="settings-group">
                    <label>{language === 'en' ? 'VAD Sensitivity' : '自动激活灵敏度'}</label>
                    <div className="slider-container">
                      <input
                        type="range"
                        className="liquid-slider"
                        min="0"
                        max="100"
                        value={vadSensitivity}
                        onChange={(e) => setVadSensitivity(parseInt(e.target.value))}
                      />
                      <span className="slider-display">{vadSensitivity}%</span>
                    </div>
                  </div>

                  <div className="settings-row-inline">
                    <label>{language === 'en' ? 'Capture Mode' : '捕获模式'}</label>
                    <select
                      className="liquid-select"
                      value={captureMode}
                      onChange={(e) => setCaptureMode(e.target.value as any)}
                    >
                      <option value="ptt">PTT (Hold-To-Talk)</option>
                      <option value="auto">VAD (Auto)</option>
                    </select>
                  </div>
                </div>

                {/* TTS voice settings group */}
                <div className="panel-group-title">{language === 'en' ? 'TTS Voice' : '语音合成配置 (TTS)'}</div>
                <div className="panel-group-card">
                  {/* engine pick rides session creation — lock it while live,
                      same as the system prompt */}
                  {(cloudTts.elevenlabs || cloudTts.minimax) && (
                    <div className="settings-row-inline">
                      <label>{language === 'en' ? 'Engine' : '合成引擎'}</label>
                      <select
                        className="liquid-select"
                        value={ttsEngine}
                        disabled={streamConnected}
                        title={streamConnected
                          ? (language === 'en' ? 'Locked while the session is live' : '会话进行中锁定 — 修改在下次连线生效')
                          : undefined}
                        onChange={(e) => {
                          const v = e.target.value as 'local' | 'elevenlabs' | 'minimax';
                          setTtsEngine(v);
                          // voices don't cross engines — re-seat the default:
                          // cloud lanes speak voice_ids, the local pool names
                          setVoicePreset(v === 'local'
                            ? 'Yuewen'
                            : (cloudTts[v]?.voices[0]?.voice_id ?? ''));
                        }}
                      >
                        <option value="local">{language === 'en' ? 'Local (MOSS-TTS)' : '本地 (MOSS-TTS)'}</option>
                        {cloudTts.minimax && (
                          <option value="minimax">{language === 'en' ? 'MiniMax (best 中文)' : 'MiniMax（中文最佳）'}</option>
                        )}
                        {cloudTts.elevenlabs && <option value="elevenlabs">ElevenLabs ☁</option>}
                      </select>
                    </div>
                  )}
                  <div className="settings-row-inline">
                    <label>{language === 'en' ? 'Voice' : '合成音色'}</label>
                    {/* local: the sidecar's builtin voice names; cloud lanes:
                        their voice registry (voice_id on the wire) */}
                    <select
                      className="liquid-select"
                      value={voicePreset}
                      onChange={(e) => setVoicePreset(e.target.value)}
                    >
                      {ttsEngine !== 'local' && cloudTts[ttsEngine] ? (
                        <>
                          {cloudTts[ttsEngine]!.voices.map((v) => (
                            <option key={v.voice_id} value={v.voice_id}>{v.name}</option>
                          ))}
                          <option value="none">{language === 'en' ? '(none — no voice)' : '（无 — 关闭语音）'}</option>
                        </>
                      ) : (
                        ['Yuewen', 'Junhao', 'Zhiming', 'Weiguo', 'Xiaoyu', 'Lingyu',
                          'Ava', 'Bella', 'Adam', 'Nathan', 'Sakura', 'Yui', 'Aoi',
                          'Hina', 'Mei', 'Trump', 'none'].map((v) => (
                          <option key={v} value={v}>
                            {v === 'none'
                              ? (language === 'en' ? '(none — no voice)' : '（无 — 关闭语音）')
                              : `${v}${v === 'Yuewen' ? (language === 'en' ? ' (default)' : '（默认）') : ''}`}
                          </option>
                        ))
                      )}
                    </select>
                  </div>

                  <div className="settings-group">
                    <label>{language === 'en' ? 'Speaking Rate' : '合成倍速'}</label>
                    <div className="slider-container">
                      <input
                        type="range"
                        className="liquid-slider"
                        min="0.5"
                        max="2.0"
                        step="0.1"
                        value={voiceRate}
                        onChange={(e) => setVoiceRate(parseFloat(e.target.value))}
                      />
                      <span className="slider-display">{voiceRate}x</span>
                    </div>
                  </div>
                </div>

                {/* Visualizer theme settings group */}
                <div className="panel-group-title">{language === 'en' ? 'Visuals' : '界面视觉特效'}</div>
                <div className="panel-group-card">
                  <div className="settings-row-inline">
                    <label>{language === 'en' ? 'Orb Theme' : '视觉球主题'}</label>
                    <select
                      className="liquid-select"
                      value={orbTheme}
                      onChange={(e) => setOrbTheme(e.target.value)}
                    >
                      <option value="fluid">Fluid Siri</option>
                      <option value="cosmic">Cosmic Aura</option>
                      <option value="quantum">Quantum Nebula</option>
                    </select>
                  </div>
                </div>

                {/* Streaming model params: fps · temperature · top_p · top_k ·
                    tokens/s rate cap. Creation-time — the fields lock while a
                    session is live (same as the system prompt), applied on the
                    next connect. fps drives the client frame sampler; the rest
                    ride the session's `params`. */}
                <div className="panel-group-title">{language === 'en' ? 'Streaming Model Parameters' : '流式模型参数'}</div>
                <div className={`panel-group-card${streamConnected ? ' locked' : ''}`}>
                  <div className="param-grid-2x2">
                    <div className="param-cell">
                      <label>{language === 'en' ? 'FPS' : '帧率 FPS'}</label>
                      <input
                        type="number" className="liquid-number"
                        min="0.5" max="10" step="0.5"
                        value={streamFps} disabled={streamConnected}
                        onChange={(e) => setStreamFps(e.target.value)}
                        onBlur={() => setStreamFps(String(parseParam(streamFps, 0.5, 10, 1)))}
                        title={streamConnected ? (language === 'en' ? 'Locked while live — applies next call' : '会话中锁定 — 下次连线生效') : undefined}
                      />
                    </div>
                    <div className="param-cell">
                      <label>{language === 'en' ? 'Temperature' : '温度 temperature'}</label>
                      <input
                        type="number" className="liquid-number"
                        min="0" max="2" step="0.05"
                        value={temperature} disabled={streamConnected}
                        onChange={(e) => setTemperature(e.target.value)}
                        onBlur={() => setTemperature(String(parseParam(temperature, 0, 2, 0.7)))}
                        title={streamConnected ? (language === 'en' ? 'Locked while live — applies next call' : '会话中锁定 — 下次连线生效') : undefined}
                      />
                    </div>
                    <div className="param-cell">
                      <label>top_p</label>
                      <input
                        type="number" className="liquid-number"
                        min="0" max="1" step="0.05"
                        value={topP} disabled={streamConnected}
                        onChange={(e) => setTopP(e.target.value)}
                        onBlur={() => setTopP(String(parseParam(topP, 0, 1, 0.8)))}
                        title={streamConnected ? (language === 'en' ? 'Locked while live — applies next call' : '会话中锁定 — 下次连线生效') : undefined}
                      />
                    </div>
                    <div className="param-cell">
                      <label>top_k</label>
                      <input
                        type="number" className="liquid-number"
                        min="1" max="100" step="1"
                        value={topK} disabled={streamConnected}
                        onChange={(e) => setTopK(e.target.value)}
                        onBlur={() => setTopK(String(Math.round(parseParam(topK, 1, 100, 20))))}
                        title={streamConnected ? (language === 'en' ? 'Locked while live — applies next call' : '会话中锁定 — 下次连线生效') : undefined}
                      />
                    </div>
                    <div className="param-cell">
                      <label>{language === 'en' ? 'Generation rate (tokens/s)' : '生成速率 tokens/s'}</label>
                      <input
                        type="number" className="liquid-number"
                        min="1" max="500" step="1"
                        value={maxTokensRate} disabled={streamConnected}
                        onChange={(e) => setMaxTokensRate(e.target.value)}
                        onBlur={() => setMaxTokensRate(String(Math.round(parseParam(maxTokensRate, 1, 500, 4))))}
                        title={streamConnected ? (language === 'en' ? 'Locked while live — applies next call' : '会话中锁定 — 下次连线生效') : undefined}
                      />
                    </div>
                  </div>
                </div>

                  </div>
                </div>
              </div>
            </aside>

          </div>

        </div>

        {/* ==================== COLLAPSED SIDEBAR MODAL POPUPS ==================== */}
        {!sidebarExpanded && activeModal === 'history' && (
          <div className="sidebar-popup-modal" style={{ top: `${modalTop}px` }}>
            <div className="modal-header-row">
              <h3>{isStreamingMode
                ? (language === 'en' ? 'Live Sessions' : '实时会话记录')
                : (language === 'en' ? 'Recent Chats' : '历史会话记录')}</h3>
              <button className="modal-close-btn" onClick={() => setActiveModal(null)} title="Close">
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
              </button>
            </div>
            
            <div className="modal-search-row">
              <MagnifyingGlass className="nav-icon" size={16} />
              <input 
                type="text" 
                value={historyQuery} 
                onChange={(e) => setHistoryQuery(e.target.value)} 
                placeholder={language === 'en' ? "Search..." : "搜索..."} 
                className="modal-search-input"
                autoFocus
              />
              
              {/* Index rail: dot-on-a-rail pager for the list below */}
              <HistoryScrollRail
                listRef={modalHistoryListRef}
                itemsCount={filteredHistory.length}
                listId="modal-history-list"
                label={language === 'en' ? 'History list position' : '历史列表位置'}
              />
            </div>

            <div className="modal-scroll-content" id="modal-history-list" ref={modalHistoryListRef}>
              {filteredHistory.map((item) => (
                <button key={item.conversation_id} className="sidebar-dropdown-item" onClick={() => { openHistoryItem(item); setActiveModal(null); }} style={{ margin: 0, width: '100%' }}>
                  {renderHistoryItemIcon(item)}
                  <span>{item.title || (language === 'en' ? '(untitled)' : '（无标题）')}</span>
                </button>
              ))}
              {filteredHistory.length === 0 && (
                <span style={{ fontSize: '11px', color: 'var(--text-muted)', textAlign: 'center', padding: '10px 0' }}>
                  {language === 'en' ? 'No history found' : '未找到历史记录'}
                </span>
              )}
            </div>
          </div>
        )}

        {!sidebarExpanded && activeModal === 'demos' && (
          <div className="sidebar-popup-modal" style={{ top: `${modalTop}px` }}>
            <div className="modal-header-row">
              <h3>{language === 'en' ? 'Interactive Demos' : '交互式功能演示'}</h3>
              <button className="modal-close-btn" onClick={() => setActiveModal(null)} title="Close">
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
              </button>
            </div>
            
            <div className="modal-scroll-content" style={{ maxHeight: 'none' }}>
              {isStreamingMode ? (
                /* The offline generative-card demos render into the CHAT
                   workspace and can't appear on the live stage — on the live
                   page the popup offers the subtitle-display demo instead:
                   a simulated session (start → model line → end) that also
                   shows off the mini ball's drift choreography. */
                <button
                  className="sidebar-dropdown-item"
                  onClick={runSubtitleDemo}
                  disabled={streamConnected}
                  title={streamConnected ? (language === 'en' ? 'Unavailable during a live session' : '实时会话中不可用') : undefined}
                  style={{ margin: 0, width: '100%', opacity: streamConnected ? 0.45 : 1 }}
                >
                  <ChatCircleDots size={12} />
                  <span>{language === 'en' ? 'Model Subtitle Display' : '模型字幕输出演示'}</span>
                </button>
              ) : (
                <>
                  <button className="sidebar-dropdown-item" onClick={() => { submitChat("Show me the latency performance charts."); setActiveModal(null); }} style={{ margin: 0, width: '100%' }}>
                    {renderChartIcon()}
                    <span>{language === 'en' ? 'Performance Chart' : '运行性能看板'}</span>
                  </button>
                  <button className="sidebar-dropdown-item" onClick={() => { submitChat("Play a voice/audio sample waveform."); setActiveModal(null); }} style={{ margin: 0, width: '100%' }}>
                    {renderWaveIcon()}
                    <span>{language === 'en' ? 'Audio Waveform' : '波形声音播放'}</span>
                  </button>
                  <button className="sidebar-dropdown-item" onClick={() => { submitChat("Give me the Liquid Glass CSS code block."); setActiveModal(null); }} style={{ margin: 0, width: '100%' }}>
                    {renderCodeIcon()}
                    <span>{language === 'en' ? 'Glass CSS Styles' : '玻璃微态样式'}</span>
                  </button>
                  <button className="sidebar-dropdown-item" onClick={() => { submitChat("Help. List all system commands."); setActiveModal(null); }} style={{ margin: 0, width: '100%' }}>
                    {renderHelpIcon()}
                    <span>{language === 'en' ? 'System Help Commands' : '离线控制帮助'}</span>
                  </button>
                </>
              )}
            </div>
          </div>
        )}

        {/* ==================== LIVE SESSION REPLAY (read-only glass modal) ==================== */}
        {/* File-stream sessions: the stored video plays with the dialogue
            anchored at each turn's media_ts — a chip click seeks, playback
            highlights the line being passed. Camera sessions: transcript
            only, timed from session start. Nothing here writes back. */}
        {/* Screen-surface pre-picker: OUR centered glass sheet chooses the
            surface TYPE (screen / window / tab) — the web can't enumerate
            windows, so the browser's native confirm still follows, pre-shaped
            by the displaySurface hint we pass. Backdrop click / Escape / ✕
            dismiss (and abandon a call-armed connect). */}
        {screenPickerOpen && (
          <div className="media-viewer-backdrop screen-picker-backdrop" onClick={dismissScreenPicker}>
            <div
              className="screen-picker glass-card"
              role="dialog"
              aria-modal="true"
              aria-label={language === 'en' ? 'Choose what to share' : '选择共享内容'}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="card-glass-specular"></div>
              <div className="screen-picker-header">
                <div className="screen-picker-title-block">
                  <h3 className="screen-picker-title">
                    {language === 'en' ? 'Share your screen' : '共享你的屏幕'}
                  </h3>
                  <span className="screen-picker-subtitle">
                    {language === 'en'
                      ? 'Pick a source — your browser will then confirm the exact one.'
                      : '选择共享类型 — 随后浏览器会确认具体窗口。'}
                  </span>
                </div>
                <button
                  className="modal-close-btn"
                  onClick={dismissScreenPicker}
                  title={language === 'en' ? 'Cancel' : '取消'}
                >
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                  </svg>
                </button>
              </div>
              <div className="screen-picker-grid">
                <button className="screen-surface-card" onClick={() => void pickScreenSurface('monitor')}>
                  <div className="screen-surface-icon"><Monitor size={30} weight="duotone" /></div>
                  <span className="screen-surface-label">{language === 'en' ? 'Entire screen' : '整个屏幕'}</span>
                  <span className="screen-surface-hint">{language === 'en' ? 'A full display' : '完整显示器画面'}</span>
                </button>
                <button className="screen-surface-card" onClick={() => void pickScreenSurface('window')}>
                  <div className="screen-surface-icon"><AppWindow size={30} weight="duotone" /></div>
                  <span className="screen-surface-label">{language === 'en' ? 'Window' : '应用窗口'}</span>
                  <span className="screen-surface-hint">{language === 'en' ? 'One app window' : '单个应用窗口'}</span>
                </button>
                <button className="screen-surface-card" onClick={() => void pickScreenSurface('browser')}>
                  <div className="screen-surface-icon"><Browser size={30} weight="duotone" /></div>
                  <span className="screen-surface-label">{language === 'en' ? 'Browser tab' : '浏览器标签页'}</span>
                  <span className="screen-surface-hint">{language === 'en' ? 'A single tab' : '单个标签页'}</span>
                </button>
              </div>
            </div>
          </div>
        )}

        {liveReplay && (
          <div className="media-viewer-backdrop" onClick={() => setLiveReplay(null)}>
            <div
              className={`replay-viewer ${liveReplay.video ? 'has-video' : ''}`}
              role="dialog"
              aria-modal="true"
              aria-label={language === 'en' ? 'Live session replay' : '实时会话回放'}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="card-glass-specular"></div>
              <div className="replay-header">
                <div className="replay-title-block">
                  <h3 className="replay-title">
                    {liveReplay.title || (language === 'en' ? '(untitled session)' : '（无标题会话）')}
                  </h3>
                  <span className="replay-subtitle">
                    {liveReplay.video
                      ? (language === 'en' ? 'Video stream replay · read-only' : '视频推流回放 · 只读')
                      : (language === 'en' ? 'Video chat log · read-only' : '视频通话记录 · 只读')}
                    {' · '}
                    {new Date(liveReplay.createdAt * 1000).toLocaleString(language === 'en' ? 'en-US' : 'zh-CN')}
                  </span>
                </div>
                <button
                  className="modal-close-btn"
                  onClick={() => setLiveReplay(null)}
                  title={language === 'en' ? 'Close' : '关闭'}
                >
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                  </svg>
                </button>
              </div>
              {liveReplay.video && (
                <video
                  ref={replayVideoRef}
                  className="replay-video"
                  src={liveReplay.video.url}
                  controls
                  playsInline
                  preload="metadata"
                  onTimeUpdate={(e) => setReplayTime(e.currentTarget.currentTime)}
                />
              )}
              <div className="session-conversation-log replay-log" ref={replayLogRef}>
                {liveReplay.turns.map((t, i) => (
                  t.system ? (
                    <div key={i} className="log-entry system">
                      <div className="log-entry-system-chip">{t.text}</div>
                    </div>
                  ) : (
                  <div
                    key={i}
                    className={`log-entry ${t.role === 'user' ? 'user' : 'ai'} ${i === replayActiveIdx ? 'replay-active' : ''}`}
                  >
                    <div className="log-entry-meta">
                      <span className="log-entry-speaker">
                        {t.role === 'user' ? (language === 'en' ? 'You' : '您') : 'MOSS'}
                      </span>
                      {liveReplay.video && t.mediaTs !== null ? (
                        <button
                          className="replay-ts-chip"
                          onClick={() => seekReplay(t.mediaTs!)}
                          title={language === 'en' ? 'Jump the video to this moment' : '视频跳转到此时刻'}
                        >
                          <Play size={8} weight="fill" />
                          {fmtReplayTs(t.mediaTs)}
                        </button>
                      ) : (
                        <span className="log-entry-time">
                          {fmtReplayTs(Math.max(0, t.ts - liveReplay.createdAt))}
                        </span>
                      )}
                    </div>
                    <div className="log-entry-text">{t.text}</div>
                  </div>
                  )
                ))}
                {liveReplay.turns.length === 0 && (
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)', textAlign: 'center', padding: '20px 0', fontFamily: 'var(--font-body)' }}>
                    {language === 'en' ? 'No dialogue recorded in this session.' : '本次会话没有已记录的对话。'}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ==================== MEDIA VIEWER (glass lightbox) ==================== */}
        {/* Liquid-glass frame over a dimmed blur scrim. Two neutral bead dots at
            the top-left (macOS traffic-light geometry, house-glass color): the
            first closes, the second reveals the info card on hover. Backdrop
            click + Escape also close. */}
        {mediaViewer && (
          <div className="media-viewer-backdrop" onClick={closeMediaViewer}>
            <div
              className="media-viewer"
              role="dialog"
              aria-modal="true"
              aria-label={language === 'en' ? 'Media viewer' : '媒体查看器'}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="viewer-dots">
                <button
                  className="viewer-dot"
                  onClick={closeMediaViewer}
                  title={language === 'en' ? 'Close' : '关闭'}
                  aria-label={language === 'en' ? 'Close' : '关闭'}
                >
                  <span className="viewer-dot-glyph">×</span>
                </button>
                {/* dot + card share one hover zone (with an invisible bridge
                    over the gap), so the pointer can travel onto the card and
                    it STAYS — values stay selectable/copyable */}
                <div
                  className="viewer-info-zone"
                  onMouseEnter={viewerInfoZoneEnter}
                  onMouseLeave={viewerInfoZoneLeave}
                >
                  <button
                    className="viewer-dot"
                    onClick={() => (viewerInfoOpen ? setViewerInfoOpen(false) : void loadViewerInfo())}
                    title={language === 'en' ? 'Media info' : '媒体信息'}
                    aria-label={language === 'en' ? 'Media info' : '媒体信息'}
                  >
                    <span className="viewer-dot-glyph">i</span>
                  </button>
                  {viewerInfoOpen && (
                    <div className="media-info-anchor">
                      <div className="media-info-card" role="tooltip">
                        {(viewerInfo ?? [[language === 'en' ? 'Loading…' : '读取中…', '']]).map(([k, v]) => (
                          <div key={k} className="media-info-row">
                            <span className="media-info-key">{k}</span>
                            <span className="media-info-val">{v}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
              {mediaViewer.kind === 'image' ? (
                <img
                  ref={(el) => { viewerMediaRef.current = el; }}
                  src={mediaViewer.url}
                  className="media-viewer-content"
                  alt=""
                />
              ) : (
                <video
                  ref={(el) => { viewerMediaRef.current = el; }}
                  src={mediaViewer.url}
                  className="media-viewer-content"
                  controls
                  autoPlay
                  playsInline
                />
              )}
            </div>
          </div>
        )}

      </div>
    </IconContext.Provider>
  );
}
