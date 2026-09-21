// MediaStream -> AudioWorklet -> PCM16 mono chunks at 16 kHz.
export interface MicCapture {
  setForwarding(on: boolean): void;
  stop(): Promise<void>;
}

export async function startMicCapture(
  stream: MediaStream,
  onChunk: (pcm: ArrayBuffer) => void,
  options: { signal?: AbortSignal; timeoutMs?: number } = {},
): Promise<MicCapture> {
  const { signal } = options;
  if (signal?.aborted) throw new DOMException('Microphone setup cancelled', 'AbortError');
  if (stream.getAudioTracks().length === 0) throw new Error('media stream has no audio track');
  const ctx = new AudioContext();
  let source: MediaStreamAudioSourceNode | null = null;
  let node: AudioWorkletNode | null = null;
  let sink: GainNode | null = null;
  let forwarding = true;
  const deadline = Date.now() + (options.timeoutMs ?? 10000);

  async function bounded(promise: Promise<void>) {
    let timer: ReturnType<typeof setTimeout> | undefined;
    let abort: (() => void) | undefined;
    try {
      await Promise.race([promise, new Promise<never>((_, reject) => {
        abort = () => reject(new DOMException('Microphone setup cancelled', 'AbortError'));
        signal?.addEventListener('abort', abort, { once: true });
        if (signal?.aborted) abort();
        timer = setTimeout(() => reject(new Error('Microphone setup timed out')), Math.max(0, deadline - Date.now()));
      })]);
    } finally {
      if (timer !== undefined) clearTimeout(timer);
      if (abort) signal?.removeEventListener('abort', abort);
    }
  }

  async function stop() {
    forwarding = false;
    if (node) node.port.onmessage = null;
    for (const part of [source, node, sink]) {
      try { part?.disconnect(); } catch { /* Already disconnected. */ }
    }
    await ctx.close().catch(() => undefined);
  }

  try {
    await bounded(ctx.audioWorklet.addModule(`${import.meta.env.BASE_URL}worklets/pcm-worklet.js`));
    source = ctx.createMediaStreamSource(stream);
    node = new AudioWorkletNode(ctx, 'pcm-worklet', {
      numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1, channelCountMode: 'explicit',
    });
    node.port.onmessage = (msg: MessageEvent<ArrayBuffer>) => {
      if (forwarding) onChunk(msg.data);
    };
    sink = ctx.createGain();
    sink.gain.value = 0;
    source.connect(node);
    node.connect(sink);
    sink.connect(ctx.destination);
    if (ctx.state === 'suspended') await bounded(ctx.resume());
    if (signal?.aborted) throw new DOMException('Microphone setup cancelled', 'AbortError');
    return { setForwarding(on: boolean) { forwarding = on; }, stop };
  } catch (error) {
    await stop();
    throw error;
  }
}
