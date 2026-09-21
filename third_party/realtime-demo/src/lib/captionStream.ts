// captionStream — sentence-windowing for the live subtitle.
//
// The streaming model emits a whole paragraph (many sentences) per turn, ending
// in `<|silence|>`. Showing the full paragraph on the orb reads poorly; instead
// the subtitle shows ONE sentence at a time and advances as each completes.
// The split rule is ported verbatim from the backend TTS segmenter
// (server/voice/segmenter.py:13) so the caption tracks the same clause the model
// would speak: end punctuation 。！？!?；; and newline, plus a bare `.` only when
// it is NOT between digits (so "3.14" never splits).
//
// —— 3-line hard cap ————————————————————————————————————————————————————————
// A sentence with no terminator can grow unbounded and stack the subtitle many
// lines high. Any single caption segment is therefore capped at ~3 rendered
// lines (.captions-overlay-container is max-width 640px at 20px type → ≈64
// latin units per line; CJK/fullwidth glyphs count double). When a segment
// overflows the budget it is force-closed: preferably at a soft punctuation
// mark inside the budget, else at whitespace, else cut hard mid-run. Chunks
// carved this way commit as their own transcript bubbles flagged `capped`, so
// the UI can mark them as machine-split rather than model-punctuated.

const SENTENCE_END_RE = /[。！？!?；;\n]|(?<!\d)\.(?!\d)/g;

export type CapKind = 'punct' | 'hard';

/** One committed slice of the reply stream. `text` is always an EXACT
 *  substring of the source buffer (callers do length arithmetic on it);
 *  display decorations like the continuation ellipsis are added at render. */
export interface CaptionSegment {
  text: string;
  /** Set when the 3-line cap force-closed this chunk: 'punct' = fell back to
   *  a soft punctuation/whitespace boundary, 'hard' = cut mid-run. */
  capped?: CapKind;
}

/** Width budget for one subtitle line, in latin-glyph units (CJK counts 2).
 *  640px column / 20px type ≈ 64 units; 56 leaves slack for the speaker tag
 *  and ragged wrap points. */
const LINE_UNITS = 56;
const MAX_CAPTION_LINES = 3;
const CAP_UNITS = LINE_UNITS * MAX_CAPTION_LINES;

/** CJK + fullwidth forms render ~2× the width of latin glyphs. */
const WIDE_CH_RE =
  /[ᄀ-ᅟ⺀-〾ぁ-㏿㐀-䶿一-鿿ꀀ-꓏가-힣豈-﫿︰-﹏＀-｠￠-￦]/;

/** Mid-sentence break candidates for the punctuation fallback (terminators
 *  never reach here — the sentence split already consumed them). Whitespace
 *  counts so an English run without commas still breaks between words. */
const SOFT_BOUNDARY_RE = /[，、,：:·—…()（）[\]【】《》「」『』“”‘’'"\s]/;

/** First carve of an overlong run: the chunk to force-close and how it was
 *  found, or null when `text` fits the 3-line budget. */
function capOne(text: string): { head: string; kind: CapKind } | null {
  let units = 0;
  let index = 0; // string index of the code point being weighed
  let softCut = -1; // index just after the last soft boundary inside budget
  for (const cp of text) {
    units += WIDE_CH_RE.test(cp) ? 2 : 1;
    if (units > CAP_UNITS) {
      if (softCut > 0) return { head: text.slice(0, softCut), kind: 'punct' };
      return { head: text.slice(0, Math.max(index, 1)), kind: 'hard' };
    }
    index += cp.length;
    if (SOFT_BOUNDARY_RE.test(cp)) softCut = index;
  }
  return null;
}

/** Split one sentence (or the open tail) into force-closed chunks plus an
 *  uncapped remainder (the remainder holds the real terminator, or is the
 *  still-growing tail). */
function capSegment(text: string): CaptionSegment[] {
  const out: CaptionSegment[] = [];
  let rest = text;
  for (let carve = capOne(rest); carve !== null; carve = capOne(rest)) {
    out.push({ text: carve.head, capped: carve.kind });
    rest = rest.slice(carve.head.length);
  }
  out.push({ text: rest });
  return out;
}

/** Segment the accumulated stream: `closed` = every committed segment in order
 *  (terminated sentences, each further split if overlong, plus full cap-chunks
 *  carved off the growing unterminated tail); `tail` = the open remainder. */
function segmentize(text: string): { closed: CaptionSegment[]; tail: string } {
  const closed: CaptionSegment[] = [];
  let last = 0;
  SENTENCE_END_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = SENTENCE_END_RE.exec(text)) !== null) {
    const end = m.index + m[0].length;
    closed.push(...capSegment(text.slice(last, end)));
    last = end;
  }
  const tailChunks = capSegment(text.slice(last));
  closed.push(...tailChunks.slice(0, -1));
  return { closed, tail: tailChunks[tailChunks.length - 1].text };
}

/** Split accumulated text into sentences, KEEPING each sentence's trailing
 *  punctuation. A trailing in-progress fragment (no closing punctuation yet)
 *  is the last element. (Terminator-only view — no cap applied.) */
export function splitSentences(text: string): string[] {
  const out: string[] = [];
  let last = 0;
  SENTENCE_END_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = SENTENCE_END_RE.exec(text)) !== null) {
    const end = m.index + m[0].length;
    out.push(text.slice(last, end));
    last = end;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

/** The sentence to display right now: the open tail while it types, else the
 *  just-closed segment (which lingers until the next one's first character
 *  arrives). Cap continuity is decorated here, display-only: a force-closed
 *  chunk shows a trailing ellipsis, its continuation a leading one. */
export function currentSentence(acc: string): string {
  if (!acc) return '';
  const { closed, tail } = segmentize(acc);
  const open = tail.trim();
  if (open) {
    const prev = closed[closed.length - 1];
    return prev?.capped ? `…${open}` : open;
  }
  for (let i = closed.length - 1; i >= 0; i--) {
    const s = closed[i].text.trim();
    if (s) return closed[i].capped ? `${s}…` : s;
  }
  return '';
}

/** All COMMITTED segments of `text` in order (terminator kept on real
 *  sentence ends; cap-carved chunks flagged); the open tail is EXCLUDED. The
 *  live transcript commits ONE BUBBLE PER ELEMENT the moment it closes, in
 *  step with the subtitle advancing — the in-progress clause lives in the
 *  caption alone. */
export function closedSegments(text: string): CaptionSegment[] {
  return segmentize(text).closed;
}

/** Back-compat string view of closedSegments (samp scheduler's length
 *  arithmetic relies on elements being exact source substrings). */
export function closedSentences(text: string): string[] {
  return closedSegments(text).map((s) => s.text);
}

/** The current sentence of `text` revealed up to `revealLen` characters. The
 *  real path reveals everything it has received (default), so the caption keeps
 *  pace with generation; the samp path passes a smaller `revealLen` it advances
 *  smoothly along the media timeline, so the sentence types out char-by-char
 *  instead of jumping chunk-by-chunk. */
export function windowedCaption(text: string, revealLen: number = text.length): string {
  const n = Math.max(0, Math.min(text.length, Math.floor(revealLen)));
  return currentSentence(text.slice(0, n));
}
