// Mic capture worklet: downsample the context rate (typically 48 kHz) to
// 16 kHz mono PCM16 and post 160 ms chunks (2560 samples / 5120 bytes) — the
// chunk size the backend ASR lane expects (backend_overhaul.md §F2).
// Linear-interpolation resampling is plenty for speech ASR.
class PcmWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.step = sampleRate / 16000; // `sampleRate` is a worklet-scope global
    this.buf = new Float32Array(0);
    this.pos = 0;
    this.out = new Int16Array(2560); // 160 ms @ 16 kHz
    this.outPos = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel || channel.length === 0) return true;

    const merged = new Float32Array(this.buf.length + channel.length);
    merged.set(this.buf);
    merged.set(channel, this.buf.length);
    this.buf = merged;

    while (this.pos + 1 < this.buf.length) {
      const i = Math.floor(this.pos);
      const frac = this.pos - i;
      const s = this.buf[i] * (1 - frac) + this.buf[i + 1] * frac;
      const v = Math.max(-1, Math.min(1, s));
      this.out[this.outPos++] = v < 0 ? v * 0x8000 : v * 0x7fff;
      if (this.outPos === this.out.length) {
        const chunk = this.out.slice(0);
        this.port.postMessage(chunk.buffer, [chunk.buffer]);
        this.outPos = 0;
      }
      this.pos += this.step;
    }

    const consumed = Math.floor(this.pos);
    this.buf = this.buf.slice(consumed);
    this.pos -= consumed;
    return true;
  }
}

registerProcessor('pcm-worklet', PcmWorklet);
