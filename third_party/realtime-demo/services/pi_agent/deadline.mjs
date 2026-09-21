import { AsyncLocalStorage } from "node:async_hooks";

const requests = new AsyncLocalStorage();

export function requestSignal(timeoutMs = 5000) {
  const local = AbortSignal.timeout(timeoutMs);
  const parent = requests.getStore();
  return parent ? AbortSignal.any([parent, local]) : local;
}

export async function withDeadline(timeoutMs, action, externalSignal) {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error("invalid request timeout");
  const controller = new AbortController();
  const error = Object.assign(new Error("memory request deadline exceeded"), { status: 504 });
  const timer = setTimeout(() => controller.abort(error), timeoutMs);
  const signal = externalSignal ? AbortSignal.any([controller.signal, externalSignal]) : controller.signal;
  let onAbort;
  const cancelled = new Promise((_, reject) => {
    onAbort = () => reject(signal.reason || error);
    if (signal.aborted) onAbort();
    else signal.addEventListener("abort", onAbort, { once: true });
  });
  try {
    return await requests.run(signal, () => Promise.race([Promise.resolve().then(action), cancelled]));
  } finally {
    clearTimeout(timer);
    signal.removeEventListener("abort", onAbort);
  }
}
