#!/usr/bin/env node
/**
 * crazyAI matmul hall of fame - the best invent kernels found so far.
 *
 * Pure Node, no dependencies: same approach and palette as flowchart.js /
 * invent_flowchart.js. A ranked bar chart of the top results by `value`
 * (speedup vs. a cache-blocked loop), then a story card per top-3 kernel:
 * the actual in-world story that produced it, the assumption it broke, and
 * what it became in C.
 *
 *   node assets/flowchart/matmul_showcase.js   # -> assets/flowchart/matmul_showcase.{svg,html,png}
 *   make matmul-showcase
 */

const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

// ---------------------------------------------------------------- palette (matches flowchart.js)
const C = {
  bg0: "#0f1022", bg1: "#1b1d3d",
  invent: "#ff7f50", measure: "#00ced1", mixed: "#c084fc", claude: "#ffbf00",
  text: "#f4f4ff", muted: "#a9abc9", line: "#8a8dbf",
  card: "#23264d", cardEdge: "#3b3f7a",
};

// ---------------------------------------------------------------- data (real archived runs)
const ranked = [
  { seed: 3, value: 16.012, top: true },
  { seed: 3004, value: 14.19, top: true },
  { seed: 42, value: 11.446, top: true },
  { seed: 4002, value: 10.431, top: false },
  { seed: 3005, value: 9.795, top: false },
];

const cards = [
  {
    seed: 3, value: "16.0", label: "value (\u00d7 blocked loop)", blend: "evolve",
    assumption: "the sum over k finishes before the next cell starts",
    hook: ["\u201cEvery boulder crosses in a covered cart \u2014", "unglanced, unswerving \u2014 ground on the quern", "and stacked, stone on stone, on the cairn.\u201d"],
    technique: ["Partial sums live only in registers (the \u201ccart\u201d);", "C is written exactly once, after the full k-sum."],
  },
  {
    seed: 3004, value: "14.2", label: "value (\u00d7 blocked loop)", blend: "graft",
    assumption: "a matrix lives in one memory",
    hook: ["\u201cSparrows carry each door\u2019s product into the", "granary where two fields\u2019 shadows cross \u2014", "flight, not arithmetic.\u201d"],
    technique: ["Two independent FMA accumulator chains per row;", "A and B are read in place, never repacked."],
  },
  {
    seed: 42, value: "121", label: "GFLOP/s @ n=1024 (first machine)", blend: "anneal",
    assumption: "n\u00b3 multiplications are needed",
    hook: ["\u201cCoral polyps eat coats hung along a pipe;", "the smoke of every product drifts into one fog", "that only gives up its embers when it has finished.\u201d"],
    technique: ["Full-k register accumulation, zero partial-C traffic;", "8\u00d724 AVX-512 tile \u2014 the project's headline result."],
  },
];

// ---------------------------------------------------------------- layout
const W = 1900, H = 900;
const padX = 60;

// ---------------------------------------------------------------- svg helpers
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const text = (x, y, s, o = {}) =>
  `<text x="${x}" y="${y}" font-size="${o.size || 14}" font-weight="${o.weight || 400}" fill="${o.fill || C.text}" ` +
  `text-anchor="${o.anchor || "start"}" font-family="${o.mono ? "'JetBrains Mono', 'Fira Code', Menlo, monospace" : "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"}" ` +
  `letter-spacing="${o.spacing || 0}" font-style="${o.italic ? "italic" : "normal"}" opacity="${o.opacity ?? 1}">${esc(s)}</text>`;

let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="${C.bg0}"/><stop offset="1" stop-color="${C.bg1}"/>
  </linearGradient>
  <linearGradient id="barTop" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="${C.invent}"/><stop offset="1" stop-color="#ffb199"/>
  </linearGradient>
  <filter id="shadow" x="-10%" y="-10%" width="130%" height="140%">
    <feDropShadow dx="0" dy="5" stdDeviation="7" flood-color="#000" flood-opacity="0.45"/>
  </filter>
  <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
    <path d="M40 0 L0 0 0 40" fill="none" stroke="#ffffff" stroke-opacity="0.04" stroke-width="1"/>
  </pattern>
</defs>
<rect width="100%" height="100%" fill="url(#bg)"/>
<rect width="100%" height="100%" fill="url(#grid)"/>
`;

svg += text(W / 2, 46, "crazyAI \u00d7 matrix multiplication", { size: 30, weight: 800, anchor: "middle", spacing: 0.4 });
svg += text(W / 2, 74, "the best kernels invent has found, ranked by value = speedup vs. a cache-blocked loop \u00d7 exactness", { size: 13.5, fill: C.muted, anchor: "middle" });

// ---------------------------------------------------------------- bar chart
{
  const chartTop = 106, rowH = 40, rowGap = 8;
  const labelW = 90, barX = padX + labelW, barMaxW = 1250, valX = barX + barMaxW + 24;
  const maxVal = Math.max(...ranked.map((r) => r.value));
  svg += text(padX, chartTop - 14, "TOP 5 ARCHIVED RUNS, BY VALUE", { size: 12, weight: 700, fill: C.muted, spacing: 0.8 });
  ranked.forEach((r, i) => {
    const y = chartTop + i * (rowH + rowGap);
    const w = (r.value / maxVal) * barMaxW;
    svg += text(barX - 14, y + rowH / 2 + 5, `seed ${r.seed}`, { size: 13.5, fill: C.text, anchor: "end", mono: true, weight: r.top ? 700 : 400 });
    svg += `<rect x="${barX}" y="${y}" width="${barMaxW}" height="${rowH}" rx="8" fill="${C.card}" stroke="${C.cardEdge}"/>`;
    svg += `<rect x="${barX}" y="${y}" width="${Math.max(w, 6)}" height="${rowH}" rx="8" fill="${r.top ? "url(#barTop)" : C.cardEdge}" opacity="${r.top ? 1 : 0.9}"/>`;
    svg += text(valX, y + rowH / 2 + 5, r.value.toFixed(2) + "\u00d7", { size: 14, fill: r.top ? C.invent : C.muted, weight: 700, mono: true });
  });
}

// ---------------------------------------------------------------- story cards
{
  const cardTop = 400, cardH = 430, gap = 34;
  const cardW = (W - 2 * padX - 2 * gap) / 3;
  cards.forEach((c, i) => {
    const x = padX + i * (cardW + gap), y = cardTop;
    svg += `<rect x="${x}" y="${y}" width="${cardW}" height="${cardH}" rx="18" fill="${C.card}" stroke="${C.cardEdge}" stroke-width="1.5" filter="url(#shadow)"/>`;
    svg += `<rect x="${x}" y="${y}" width="${cardW}" height="7" rx="3.5" fill="${C.invent}"/>`;
    svg += `<circle cx="${x + 34}" cy="${y + 46}" r="20" fill="${C.invent}"/>`;
    svg += text(x + 34, y + 53, "#" + (i + 1), { size: 17, weight: 800, fill: C.bg0, anchor: "middle" });
    svg += text(x + 66, y + 42, "seed " + c.seed, { size: 17, weight: 800 });
    svg += text(x + 66, y + 62, c.blend + " blend", { size: 12, fill: C.claude, weight: 600 });
    svg += text(x + cardW - 24, y + 48, c.value, { size: 30, weight: 800, fill: C.invent, anchor: "end" });
    svg += text(x + cardW - 24, y + 66, c.label, { size: 10, fill: C.muted, anchor: "end" });

    let ly = y + 108;
    svg += text(x + 26, ly, "THE STORY", { size: 10.5, weight: 700, fill: C.muted, spacing: 0.8 }); ly += 24;
    c.hook.forEach((l) => { svg += text(x + 26, ly, l, { size: 13, italic: true, fill: "#dfe1ff" }); ly += 20; });
    ly += 22;

    svg += text(x + 26, ly, "ASSUMPTION BROKEN", { size: 10.5, weight: 700, fill: C.muted, spacing: 0.8 }); ly += 22;
    // wrap assumption text manually at ~34 chars
    const words = c.assumption.split(" ");
    let line = "", lines = [];
    words.forEach((w) => { const t = line ? line + " " + w : w; if (t.length > 34) { lines.push(line); line = w; } else line = t; });
    if (line) lines.push(line);
    lines.forEach((l) => { svg += text(x + 26, ly, "\u201c" + l + "\u201d", { size: 12.5, fill: C.measure }); ly += 19; });
    ly += 18;

    svg += text(x + 26, ly, "WHAT IT BECAME", { size: 10.5, weight: 700, fill: C.muted, spacing: 0.8 }); ly += 22;
    c.technique.forEach((l) => { svg += text(x + 26, ly, l, { size: 12, fill: C.text }); ly += 18; });
  });
}

svg += text(W / 2, H - 34, "full stories, C source and honest caveats (n=1024, real OpenBLAS re-checks) in the README and archive/invent_<seed>_matmul/", { size: 11.5, fill: C.muted, anchor: "middle" });
svg += text(W / 2, H - 14, "crazyai invent-rank --target matmul", { size: 11, fill: C.line, anchor: "middle", mono: true });
svg += `</svg>`;

// ---------------------------------------------------------------- write + rasterise
const outDir = __dirname;
const svgPath = path.join(outDir, "matmul_showcase.svg");
const htmlPath = path.join(outDir, "matmul_showcase.html");
const pngPath = path.join(outDir, "matmul_showcase.png");
fs.writeFileSync(svgPath, svg);
fs.writeFileSync(htmlPath, `<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;background:${C.bg0}}svg{display:block}</style></head><body>${svg}</body></html>`);
console.log("wrote", svgPath, "and", htmlPath);

const scale = Number(process.env.SCALE || 2);
const candidates = [
  "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
  "msedge", "microsoft-edge",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
];
let done = false;
for (const bin of candidates) {
  try {
    execFileSync(bin, ["--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-sandbox",
      `--force-device-scale-factor=${scale}`, `--window-size=${W},${H}`, `--screenshot=${pngPath}`, `file://${htmlPath}`],
      { stdio: "ignore", timeout: 60000 });
    done = fs.existsSync(pngPath);
    if (done) { console.log("wrote", pngPath, `(${W * scale}x${H * scale}, via ${bin})`); break; }
  } catch (e) { /* try next */ }
}
if (!done) {
  for (const magickBin of ["magick", "convert"]) {
    try {
      execFileSync(magickBin, ["-density", String(96 * scale), svgPath, pngPath], { stdio: "ignore" });
      console.log("wrote", pngPath, `(via ${magickBin})`);
      done = true;
      break;
    } catch (e) { /* try next */ }
  }
}
if (!done) {
  console.error("could not rasterise: install a Chromium-family browser or ImageMagick; matmul_showcase.svg is still usable");
  process.exit(1);
}
