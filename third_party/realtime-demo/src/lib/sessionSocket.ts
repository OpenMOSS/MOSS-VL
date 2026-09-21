// Session WebSocket client — the browser half of server/protocol.py
// (frozen wire protocol). One socket multiplexes everything:
//   up:   JSON control events · binary 0x01 mic PCM · 0x02 JPEG (+u32 BE ts-ms)
//   down: JSON events (each carries a monotonic `seq`) · binary 0x11 TTS PCM,
//         always preceded by its `response.audio.delta` descriptor.
//
// Reconnect contract: `seq` is monotonic but NOT gapless (transient events are
// never replayed) — we track max(seq) and reconnect with ?last_seq=N so the
// server replays only what we missed. Close code 4000 means another socket
// superseded this one (stop retrying); 4404 means the session is gone
// (the caller must create a new one over REST); 1013 means the server
// refused for capacity (stop retrying — an immediate retry lands the same).

export const TAG_MIC_PCM = 0x01;
export const TAG_VIDEO_JPEG = 0x02;
export const TAG_TTS_PCM = 0x11;

export interface ServerEvent {
  v?: number;
  type: string;
  seq?: number;
  [key: string]: unknown;
}

export type SocketState = 'connecting' | 'open' | 'reconnecting' | 'closed' | 'superseded' | 'gone' | 'busy';

export interface SessionSocketHandlers {
  onEvent: (ev: ServerEvent) => void;
  /** Binary TTS PCM, paired with the `response.audio.delta` descriptor that preceded it. */
  onAudio: (pcm: ArrayBuffer, descriptor: ServerEvent) => void;
  onStateChange?: (state: SocketState) => void;
}

/** Resolve an API path against the page origin — the gateway serves the app
 *  under a deep prefix (…/proxy/5173/), so `/api/...` must stay RELATIVE to
 *  the page, never absolute on the domain root. VITE_API_BASE overrides. */
export function resolveApiUrl(path: string): string {
  const base = (import.meta.env.VITE_API_BASE as string | undefined) || document.baseURI;
  return new URL(path.replace(/^\//, ''), base.endsWith('/') ? base : `${base}/`).toString();
}

export function toWsUrl(httpUrl: string): string {
  return httpUrl.replace(/^http/, 'ws');
}

const PING_INTERVAL_MS = 15_000;
const RECONNECT_BASE_MS = 500;
const RECONNECT_MAX_MS = 8_000;
// Control events queued while the socket is down (reconnect window) — capped
// so a dead session can't hoard memory. Binary media is never queued (frames/
// mic are continuous streams; stale chunks are worse than dropped ones).
const MAX_QUEUED_SENDS = 64;
// Ephemeral event types that must NOT survive a reconnect: pings are
// heartbeats for THIS socket and playback reports go stale immediately.
const QUEUE_SKIP = new Set(['ping', 'playback.status']);

export class SessionSocket {
  maxSeq = 0;
  private readonly wsUrl: string;
  private readonly handlers: SessionSocketHandlers;
  private ws: WebSocket | null = null;
  private state: SocketState = 'closed';
  private closedByUser = false;
  private retries = 0;
  private pingSeq = 0;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private pendingAudioDescriptor: ServerEvent | null = null;
  // control sends buffered during a reconnect window, flushed in order on
  // reopen — a typed message mid-blip must DELIVER, not vanish silently
  private queuedSends: string[] = [];

  constructor(wsUrl: string, handlers: SessionSocketHandlers) {
    this.wsUrl = wsUrl;
    this.handlers = handlers;
  }

  get readyState(): SocketState {
    return this.state;
  }

  get bufferedAmount(): number {
    return this.ws?.bufferedAmount ?? 0;
  }

  connect(): void {
    this.closedByUser = false;
    this.open();
  }

  close(): void {
    this.closedByUser = true;
    this.clearTimers();
    this.queuedSends = [];
    this.ws?.close(1000);
    this.ws = null;
    this.setState('closed');
  }

  sendJSON(type: string, fields: Record<string, unknown> = {}): void {
    const payload = JSON.stringify({ v: 1, type, ...fields });
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(payload);
    } else if (!this.closedByUser && !QUEUE_SKIP.has(type)) {
      // socket down (usually a reconnect window): queue in order and flush on
      // reopen. Dropping silently loses typed turns and — for a dropped
      // input.audio.commit — leaves the turn (and the listening UI) stuck.
      this.queuedSends.push(payload);
      if (this.queuedSends.length > MAX_QUEUED_SENDS) this.queuedSends.shift();
    }
  }

  sendMic(pcm: ArrayBuffer): void {
    this.sendBinary(TAG_MIC_PCM, pcm);
  }

  sendFrame(jpeg: ArrayBuffer, captureTsMs: number): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    const framed = new Uint8Array(5 + jpeg.byteLength);
    framed[0] = TAG_VIDEO_JPEG;
    new DataView(framed.buffer).setUint32(1, Math.max(0, Math.round(captureTsMs)) >>> 0, false);
    framed.set(new Uint8Array(jpeg), 5);
    this.ws.send(framed);
  }

  private sendBinary(tag: number, payload: ArrayBuffer): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    const framed = new Uint8Array(1 + payload.byteLength);
    framed[0] = tag;
    framed.set(new Uint8Array(payload), 1);
    this.ws.send(framed);
  }

  private open(): void {
    this.clearTimers();
    const url = this.maxSeq > 0 ? `${this.wsUrl}?last_seq=${this.maxSeq}` : this.wsUrl;
    this.setState(this.retries > 0 ? 'reconnecting' : 'connecting');
    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    this.ws = ws;

    ws.onopen = () => {
      this.retries = 0;
      this.setState('open');
      // deliver everything the reconnect window buffered, in send order
      const queued = this.queuedSends.splice(0);
      for (const payload of queued) ws.send(payload);
      this.pingTimer = setInterval(() => this.sendJSON('ping', { seq: ++this.pingSeq }), PING_INTERVAL_MS);
    };

    ws.onmessage = (msg: MessageEvent) => {
      if (typeof msg.data === 'string') {
        let ev: ServerEvent;
        try {
          ev = JSON.parse(msg.data);
        } catch {
          return;
        }
        if (typeof ev.seq === 'number' && ev.seq > this.maxSeq) this.maxSeq = ev.seq;
        if (ev.type === 'response.audio.delta') this.pendingAudioDescriptor = ev;
        this.handlers.onEvent(ev);
        return;
      }
      const bytes = new Uint8Array(msg.data as ArrayBuffer);
      if (bytes.length < 2 || bytes[0] !== TAG_TTS_PCM || !this.pendingAudioDescriptor) return;
      const descriptor = this.pendingAudioDescriptor;
      this.pendingAudioDescriptor = null;
      this.handlers.onAudio(bytes.slice(1).buffer, descriptor);
    };

    ws.onclose = (ev: CloseEvent) => {
      this.clearTimers();
      this.ws = null;
      if (this.closedByUser) return this.setState('closed');
      if (ev.code === 4000) return this.setState('superseded');
      if (ev.code === 4404) return this.setState('gone');
      if (ev.code === 1013) return this.setState('busy');
      const delay = Math.min(RECONNECT_MAX_MS, RECONNECT_BASE_MS * 2 ** this.retries++);
      this.setState('reconnecting');
      this.reconnectTimer = setTimeout(() => this.open(), delay);
    };
    // onerror always precedes a close event; onclose owns the retry logic
  }

  private clearTimers(): void {
    if (this.pingTimer) clearInterval(this.pingTimer);
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.pingTimer = null;
    this.reconnectTimer = null;
  }

  private setState(state: SocketState): void {
    if (this.state === state) return;
    this.state = state;
    this.handlers.onStateChange?.(state);
  }
}
