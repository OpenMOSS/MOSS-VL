// TTS PCM playback: gapless scheduling of interleaved Int16 chunks
// (backend_overhaul.md §F4). Sources are tracked per response_id so barge-in
// (`turn.speech_started` / response.done{interrupted|cancelled}) can stop one
// response's audio instantly. bufferedSeconds() feeds the server's §1C
// back-pressure gate via `playback.status`; level() drives the orb.

const SCHEDULE_LEAD_S = 0.06;

export class PcmPlayer {
  private ctx: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private analyserData: Uint8Array<ArrayBuffer> | null = null;
  private master: GainNode | null = null;
  private nextStartTime = 0;
  private sources = new Map<string, Set<AudioBufferSourceNode>>();

  private ensureContext(): AudioContext {
    if (!this.ctx) {
      this.ctx = new AudioContext();
      this.master = this.ctx.createGain();
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyserData = new Uint8Array(this.analyser.frequencyBinCount);
      this.master.connect(this.analyser);
      this.analyser.connect(this.ctx.destination);
      this.nextStartTime = 0;
    }
    if (this.ctx.state === 'suspended') void this.ctx.resume();
    return this.ctx;
  }

  /** Decode + schedule one interleaved PCM16 chunk right after the previous one. */
  playChunk(responseId: string, pcm: ArrayBuffer, sampleRate: number, channels: number): void {
    const frames = Math.floor(pcm.byteLength / 2 / channels);
    if (frames <= 0) return;
    const ctx = this.ensureContext();
    const buffer = ctx.createBuffer(channels, frames, sampleRate);
    const samples = new Int16Array(pcm, 0, frames * channels);
    for (let ch = 0; ch < channels; ch++) {
      const dest = buffer.getChannelData(ch);
      for (let i = 0; i < frames; i++) {
        dest[i] = samples[i * channels + ch] / 0x8000;
      }
    }

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.master!);
    const startAt = Math.max(ctx.currentTime + SCHEDULE_LEAD_S, this.nextStartTime);
    source.start(startAt);
    this.nextStartTime = startAt + buffer.duration;

    let set = this.sources.get(responseId);
    if (!set) {
      set = new Set();
      this.sources.set(responseId, set);
    }
    set.add(source);
    source.onended = () => {
      set!.delete(source);
      if (set!.size === 0) this.sources.delete(responseId);
    };
  }

  /** Barge-in: kill everything scheduled for one response. */
  stopResponse(responseId: string): void {
    const set = this.sources.get(responseId);
    if (!set) return;
    for (const source of set) {
      try {
        source.onended = null;
        source.stop();
      } catch {
        /* already ended */
      }
    }
    this.sources.delete(responseId);
    if (this.ctx) this.nextStartTime = this.ctx.currentTime;
  }

  stopAll(): void {
    for (const id of Array.from(this.sources.keys())) this.stopResponse(id);
  }

  /** Seconds of scheduled-but-unplayed audio — reported to the server (§1C). */
  bufferedSeconds(): number {
    if (!this.ctx) return 0;
    return Math.max(0, this.nextStartTime - this.ctx.currentTime);
  }

  get playing(): boolean {
    return this.sources.size > 0;
  }

  /** Instantaneous output level 0..~1.5 for the orb visualizer. */
  level(): number {
    if (!this.analyser || !this.analyserData || this.sources.size === 0) return 0;
    this.analyser.getByteFrequencyData(this.analyserData);
    let sum = 0;
    for (let i = 0; i < this.analyserData.length; i++) sum += this.analyserData[i];
    return (sum / this.analyserData.length / 255) * 1.6;
  }

  async close(): Promise<void> {
    this.stopAll();
    if (this.ctx) {
      await this.ctx.close().catch(() => undefined);
      this.ctx = null;
      this.master = null;
      this.analyser = null;
      this.analyserData = null;
    }
  }
}
