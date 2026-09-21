// Aurora Filament — the offline-chat thinking indicator.
//
// While the model thinks, ribbon filaments of the orb's theme colors ride the
// edge of the empty glass reply vessel over a faint full-perimeter base ring.
// Each ribbon is the study's three GAUSSIAN passes (halo σ10 / band σ4.2 /
// core σ1.6 — "gradient light, not solid strokes"): canvas 2D cannot blur a
// stroke while keeping its along-chord gradient, so each pass is approximated
// by nested concentric strokes whose alphas sample the gaussian profile
// (GAUSS_HW/GAUSS_DELTA). All strokes share ONE polyline and one gradient
// (trio colors + raised-cosine end feather as stops) — no per-segment seams.
//
// Compositing mirrors the study's composite_glow: passes accumulate
// additively in the buffer ('lighter' = the study's Σ over passes, blooms at
// overlaps); the THEME composite happens where the canvas element meets the
// page — dark: plain alpha over a near-black bg (≈ the study's screen), with
// the study's glow_gain 1.35 folded into passesDark alphas; light: CSS
// `mix-blend-mode: multiply` + `opacity: .85` on the canvas (the study's
// ink-on-paper branch, a·0.85 ceiling) so ribbons read across the white
// vessel fill and the page alike instead of washing out on the fill.
//
// Geometry is sourced from offsetWidth/offsetHeight (layout size — immune to
// the bubble's entrance transform, which poisons getBoundingClientRect) and
// re-checked every frame; the canvas box is set from JS in px, anchored to
// the BORDER box (absolute children offset from the padding box, so the 1px
// hairline is folded in — else the ring sits 1px down-right of centre).
//
// Every constant in P maps to the study's params so the two stay tunable in
// lockstep. Honors prefers-reduced-motion via opts (static two-hue edge tint,
// breathing handled in CSS); pauses while the tab is hidden.

type RGB = [number, number, number];

const TRIOS_DARK: Record<string, RGB[]> = {
  fluid: [[138, 75, 230], [74, 222, 128], [56, 189, 248]],
  cosmic: [[230, 95, 30], [244, 63, 94], [234, 179, 8]],
  quantum: [[14, 165, 233], [217, 70, 239], [20, 184, 166]],
};
// deeper hues so ribbons read as ink on the light theme
const TRIOS_LIGHT: Record<string, RGB[]> = {
  fluid: [[124, 45, 240], [13, 148, 90], [2, 122, 199]],
  cosmic: [[194, 65, 12], [200, 20, 65], [180, 105, 8]],
  quantum: [[2, 112, 178], [168, 34, 190], [13, 128, 118]],
};

// px of halo spill outside the vessel (≥ 2.5·σ_halo). Must stay within the
// chat scroller's side gutters (.chat-messages-container padding) or the
// spill clips at the flush bubble edges.
export const AURORA_MARGIN = 16;

// A σ-blurred stroke, approximated: nested concentric strokes with round
// caps/joins; half-widths in σ units, and per-stroke alpha deltas chosen so
// the additive ('lighter') stack reaches exp(−x²/2σ²) at each ring's
// midpoint. Six rings keep the steps under the antialiasing floor.
const GAUSS_HW = [2.5, 2.0, 1.5, 1.0, 0.6, 0.3];
const GAUSS_MID = GAUSS_HW.map((h, i) => (h + (GAUSS_HW[i + 1] ?? 0)) / 2);
const GAUSS_DELTA = GAUSS_MID.map((m, i) => {
  const g = Math.exp(-(m * m) / 2);
  const prev = i === 0 ? 0 : Math.exp(-(GAUSS_MID[i - 1] * GAUSS_MID[i - 1]) / 2);
  return g - prev;
});

const P = {
  spans: [0.26, 0.19, 0.12],          // fraction of perimeter per filament
  centers: [0.1, 0.47, 0.76],         // start-of-session positions
  speeds: [1 / 8, 1 / 11, 1 / 14],    // laps per second (incommensurate — never repeats)
  amps: [1.0, 0.85, 0.55],
  huePhase: [0.0, 0.38, 0.71],
  hueDrift: 0.06,                     // lap/s of gradient-stop drift (the "breathing")
  hueAlong: 0.55,                     // how much of the trio wheel one ribbon spans
  feather: 0.22,                      // raised-cosine feather over ribbon-end fraction
  segments: 56,
  // gaussian passes (halo / band / core). Sigmas are the study's pass_sigma
  // SCALED to this vessel: the study bubble is 132px tall, ours 64 — absolute
  // σ10 here reads as fog twice the designed relative width (σ·64/132 ≈
  // 4.8/2.0/0.8; core floored to 1.1 so the filament survives 1x screens).
  // Dark folds the study's glow_gain 1.35 into the alphas; light runs leaner
  // (multiply ink shows every skirt) — the ×0.85 ink ceiling lives on the
  // canvas element (CSS opacity, see index.css).
  passesDark: [
    { sigma: 5, alpha: 0.14 },
    { sigma: 2.2, alpha: 0.41 },
    { sigma: 1.1, alpha: 1.0 },
  ],
  passesLight: [
    { sigma: 5, alpha: 0.07 },
    { sigma: 2.2, alpha: 0.22 },
    { sigma: 1.1, alpha: 1.0 },
  ],
  // ribbons ride ON the hairline — the study strokes the bubble outline
  // itself; any inset shows as a second rail diverging from the edge
  edgeInset: 0,
  // faint always-on ring so the edge reads "surrounded" between ribbons.
  // Dark only: additive light disappears into the rim; in light-theme
  // multiply ink it reads as a drawn outline (a rail), so it is off there.
  baseRing: { width: 2, alphaDark: 0.10, alphaLight: 0 },
  enterAmp: 0.8,                      // amplitude ramps 0.8 → 1.0 over rampSeconds
  rampSeconds: 4,
  // static paint (settled halo + reduced-motion fallback): two-hue edge tint,
  // halo+band only. Centers are NOT fixed t values — they anchor to the
  // top-right and bottom-left corner apexes (computed per box in ensureSize),
  // so the glow peaks at the diagonal corners instead of mid-top/mid-bottom
  // (where fixed 0.25/0.75 land on a wide chat bubble).
  staticSpans: [0.5, 0.5],
  staticAmps: [0.55, 0.55],
  staticHues: [0.0, 0.55],
  settleMs: 400,                      // static arcs fade in — the ring STOPS, it doesn't snap
};

function trioColor(trio: RGB[], u: number): RGB {
  u = ((u % 1) + 1) % 1;
  const seg = Math.floor(u * 3) % 3;
  const frac = (u * 3) % 1;
  const k = 0.5 - 0.5 * Math.cos(frac * Math.PI);
  const a = trio[seg];
  const b = trio[(seg + 1) % 3];
  return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
}

/** Rounded-rect perimeter parametrised by arclength, t ∈ [0,1) clockwise from top-left. */
function perimeterFn(x0: number, y0: number, w: number, h: number, r: number) {
  r = Math.min(r, w / 2, h / 2);
  const sw = w - 2 * r;
  const sh = h - 2 * r;
  const arc = (Math.PI * r) / 2;
  const HP = Math.PI / 2;
  const segs: Array<[number, (u: number) => [number, number]]> = [
    [sw, (u) => [x0 + r + sw * u, y0]],
    [arc, (u) => [x0 + w - r + r * Math.sin(u * HP), y0 + r - r * Math.cos(u * HP)]],
    [sh, (u) => [x0 + w, y0 + r + sh * u]],
    [arc, (u) => [x0 + w - r + r * Math.cos(u * HP), y0 + h - r + r * Math.sin(u * HP)]],
    [sw, (u) => [x0 + w - r - sw * u, y0 + h]],
    [arc, (u) => [x0 + r - r * Math.sin(u * HP), y0 + h - r + r * Math.cos(u * HP)]],
    [sh, (u) => [x0, y0 + h - r - sh * u]],
    [arc, (u) => [x0 + r - r * Math.cos(u * HP), y0 + r - r * Math.sin(u * HP)]],
  ];
  const total = segs.reduce((s, [len]) => s + len, 0);
  return (t: number): [number, number] => {
    let d = (((t % 1) + 1) % 1) * total;
    for (const [len, fn] of segs) {
      if (d <= len || len <= 0) return fn(Math.min(1, d / Math.max(len, 1e-9)));
      d -= len;
    }
    return segs[segs.length - 1][1](1);
  };
}

export interface AuroraOptions {
  orbTheme: string;
  dark: boolean;
  reducedMotion: boolean;
  /** 'static' paints the two-arc reduced-motion tint once and never animates —
   *  the settled-reply halo (Option B research 2026-07-08): the newest reply
   *  keeps a quiet remnant of the thinking glow; history bubbles stay plain. */
  mode?: 'animated' | 'static';
}

interface Ribbon {
  center: number;
  span: number;
  amp: number;
  hue: number;
}

/** Attach the effect to `canvas`, tracking `host`'s layout box. Returns a cleanup fn. */
export function startAuroraFilament(
  canvas: HTMLCanvasElement,
  host: HTMLElement,
  opts: AuroraOptions,
): () => void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return () => undefined;
  const trio = (opts.dark ? TRIOS_DARK : TRIOS_LIGHT)[opts.orbTheme]
    ?? (opts.dark ? TRIOS_DARK : TRIOS_LIGHT).fluid;
  const passes = opts.dark ? P.passesDark : P.passesLight;
  const M = AURORA_MARGIN;
  // static paint = designed reduced-motion fallback OR the settled-reply halo
  const staticPaint = opts.reducedMotion || opts.mode === 'static';

  let raf: number | null = null;
  let point: (t: number) => [number, number] = () => [0, 0];
  // arclength positions of the top-right / bottom-left corner apexes
  // (t parametrises the perimeter clockwise from the start of the top edge)
  let cornerTs: [number, number] = [0.22, 0.72];
  let dpr = 1;
  let lastW = -1;
  let lastH = -1;
  const t0 = performance.now();

  /** Layout-sourced sizing; returns false while the host has no box yet. */
  const ensureSize = (): boolean => {
    const w = host.offsetWidth;
    const h = host.offsetHeight;
    if (w === 0 || h === 0) return false;
    const nextDpr = Math.min(2, window.devicePixelRatio || 1);
    if (w === lastW && h === lastH && nextDpr === dpr && canvas.width > 0) return true;
    lastW = w;
    lastH = h;
    dpr = nextDpr;
    // position + size the element from JS in px — no CSS % resolution
    // ambiguity. Anchor at the BORDER box (-M,-M): absolute children offset
    // from the padding box, so fold the hairline in via clientLeft/clientTop.
    canvas.style.left = `${-(M + host.clientLeft)}px`;
    canvas.style.top = `${-(M + host.clientTop)}px`;
    canvas.style.width = `${w + 2 * M}px`;
    canvas.style.height = `${h + 2 * M}px`;
    canvas.width = Math.round((w + 2 * M) * dpr);
    canvas.height = Math.round((h + 2 * M) * dpr);
    const radius = parseFloat(getComputedStyle(host).borderTopLeftRadius) || 20;
    const inset = P.edgeInset;
    const W = (w - 2 * inset) * dpr;
    const H = (h - 2 * inset) * dpr;
    const R0 = Math.max(4, radius - inset) * dpr;
    point = perimeterFn((M + inset) * dpr, (M + inset) * dpr, W, H, R0);
    // corner apexes in arclength (mirrors perimeterFn's segment order/clamp):
    // segments run top edge → TR arc → right → BR arc → bottom → BL arc →
    // left → TL arc, so TR apex = mid of the 2nd segment, BL = mid of the 6th
    const R = Math.min(R0, W / 2, H / 2);
    const sw = W - 2 * R;
    const sh = H - 2 * R;
    const arc = (Math.PI * R) / 2;
    const total = 2 * sw + 2 * sh + 4 * arc;
    cornerTs = [
      (sw + arc / 2) / total,                            // top-right
      (sw + arc + sh + arc + sw + arc / 2) / total,      // bottom-left
    ];
    return true;
  };

  const feather = (u: number) => {
    const fe = P.feather;
    if (u < fe) return 0.5 - 0.5 * Math.cos((u / fe) * Math.PI);
    if (u > 1 - fe) return 0.5 - 0.5 * Math.cos(((1 - u) / fe) * Math.PI);
    return 1;
  };

  const strokePolyline = (pts: Array<[number, number]>, style: string | CanvasGradient, width: number) => {
    ctx.strokeStyle = style;
    ctx.lineWidth = width * dpr;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();
  };

  const drawBaseRing = (timeS: number, master: number) => {
    if ((opts.dark ? P.baseRing.alphaDark : P.baseRing.alphaLight) <= 0) return;
    const n = 72;
    const pts: Array<[number, number]> = [];
    for (let i = 0; i <= n; i++) pts.push(point(i / n));
    const [r0, g0, b0] = trioColor(trio, P.hueDrift * timeS);
    const [r1, g1, b1] = trioColor(trio, P.hueDrift * timeS + 0.5);
    const grad = ctx.createLinearGradient(pts[0][0], pts[0][1], pts[Math.floor(n / 2)][0], pts[Math.floor(n / 2)][1]);
    const alpha = (opts.dark ? P.baseRing.alphaDark : P.baseRing.alphaLight) * master;
    grad.addColorStop(0, `rgba(${r0 | 0}, ${g0 | 0}, ${b0 | 0}, ${alpha.toFixed(4)})`);
    grad.addColorStop(1, `rgba(${r1 | 0}, ${g1 | 0}, ${b1 | 0}, ${alpha.toFixed(4)})`);
    strokePolyline(pts, grad, P.baseRing.width);
  };

  const drawRibbon = (ribbon: Ribbon, ribbonPasses: typeof passes, master: number) => {
    const n = P.segments;
    const pts: Array<[number, number]> = [];
    for (let i = 0; i <= n; i++) {
      pts.push(point(ribbon.center - ribbon.span / 2 + (ribbon.span * i) / n));
    }
    // one gradient per ribbon (hue drift + end feather in the stops);
    // pass strength and the gaussian deltas ride globalAlpha instead
    const grad = ctx.createLinearGradient(pts[0][0], pts[0][1], pts[n][0], pts[n][1]);
    for (let k = 0; k <= 8; k++) {
      const u = k / 8;
      const [r, g, b] = trioColor(trio, ribbon.hue + u * P.hueAlong);
      grad.addColorStop(u, `rgba(${r | 0}, ${g | 0}, ${b | 0}, ${feather(u).toFixed(4)})`);
    }
    for (const pass of ribbonPasses) {
      const peak = pass.alpha * ribbon.amp * master;
      for (let j = 0; j < GAUSS_HW.length; j++) {
        ctx.globalAlpha = peak * GAUSS_DELTA[j];
        strokePolyline(pts, grad, 2 * GAUSS_HW[j] * pass.sigma);
      }
    }
    ctx.globalAlpha = 1;
  };

  const drawFrame = (nowMs: number) => {
    if (!ensureSize()) return;
    const t = (nowMs - t0) / 1000;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // additive in-buffer accumulation = the study's Σ over passes (both
    // themes); the theme composite happens at the canvas ELEMENT (header)
    ctx.globalCompositeOperation = 'lighter';
    const master = P.enterAmp + (1 - P.enterAmp) * Math.min(1, t / P.rampSeconds);
    drawBaseRing(t, master);
    for (let i = 0; i < P.spans.length; i++) {
      drawRibbon(
        {
          center: (P.centers[i] + P.speeds[i] * t) % 1,
          span: P.spans[i],
          amp: P.amps[i],
          hue: (P.huePhase[i] + P.hueDrift * t) % 1,
        },
        passes,
        master,
      );
    }
  };

  const settleT0 = performance.now();

  const drawStatic = () => {
    if (!ensureSize()) return;
    // the ring STOPS rather than snaps: arcs fade in over settleMs
    const master = Math.min(1, (performance.now() - settleT0) / P.settleMs);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.globalCompositeOperation = 'lighter';
    drawBaseRing(0, master);
    for (let i = 0; i < cornerTs.length; i++) {
      drawRibbon(
        {
          center: cornerTs[i], // peak at the TR / BL corner apex
          span: P.staticSpans[i],
          amp: P.staticAmps[i],
          hue: P.staticHues[i],
        },
        passes.slice(0, 2), // halo + band only
        master,
      );
    }
  };

  const settleLoop = () => {
    drawStatic();
    if (performance.now() - settleT0 < P.settleMs) {
      raf = requestAnimationFrame(settleLoop);
    } else {
      raf = null;
    }
  };

  const loop = (now: number) => {
    drawFrame(now);
    raf = requestAnimationFrame(loop);
  };

  const onVisibility = () => {
    if (document.hidden) {
      if (raf !== null) cancelAnimationFrame(raf);
      raf = null;
    } else if (raf === null && !staticPaint) {
      raf = requestAnimationFrame(loop);
    }
  };

  // static bubbles resize as text streams in — repaint per layout change
  const observer = new ResizeObserver(() => {
    if (staticPaint) drawStatic();
  });
  observer.observe(host);

  if (staticPaint) {
    settleLoop();
  } else {
    raf = requestAnimationFrame(loop);
    document.addEventListener('visibilitychange', onVisibility);
  }

  return () => {
    if (raf !== null) cancelAnimationFrame(raf);
    document.removeEventListener('visibilitychange', onVisibility);
    observer.disconnect();
  };
}
