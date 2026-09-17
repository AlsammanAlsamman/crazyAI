#!/usr/bin/env node
/**
 * crazyAI invent pipeline flowchart.
 *
 * Pure Node, no dependencies: builds the diagram as SVG, writes
 * invent_pipeline.svg and invent_pipeline.html, then rasterises to
 * invent_pipeline.png with a headless Chromium-family browser (falls back
 * to ImageMagick `magick`/`convert` if none is found).
 *
 *   node assets/flowchart/invent_flowchart.js   # -> assets/flowchart/invent_pipeline.{svg,html,png}
 *   make invent-flowchart
 */

const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

// ---------------------------------------------------------------- palette (matches flowchart.js)
const C = {
  bg0: "#0f1022", bg1: "#1b1d3d",
  invent: "#ff7f50", inventDim: "#ff7f5033",   // RNG / math - deterministic, seeded, free
  measure: "#00ced1", measureDim: "#00ced133", // harness - the only step touching ground truth
  mixed: "#c084fc",                            // Claude + toolkit (tool-calling)
  claude: "#ffbf00",                           // Claude, no tools
  text: "#f4f4ff", muted: "#a9abc9", line: "#8a8dbf",
  card: "#23264d", cardEdge: "#3b3f7a",
};

// ---------------------------------------------------------------- data
const steps = [
  { n: 1, title: "SEED", who: "RNG", kind: "invent", file: "seed.json",
    lines: ["blend model · immersion depth", "1 of 8 matmul assumptions"] },
  { n: 2, title: "HARVEST", who: "Claude", kind: "claude", file: "harvest.yaml",
    lines: ["optional (--harvest N)", "N new corpus fragments"] },
  { n: 3, title: "WORLD", who: "RNG + math", kind: "invent", file: "world.json",
    lines: ["1 of 6 blend models", "over 4 corpus kinds"] },
  { n: 4, title: "IMMERSE", who: "Claude", kind: "claude", file: "ideas.md",
    lines: ["native of the blended", "world answers in-world"] },
  { n: 5, title: "BEND", who: "Claude + tools", kind: "mixed", file: "artifact.c",
    lines: ["maps the idea literally,", "writes + predicts the kernel"] },
  { n: 6, title: "MEASURE", who: "harness", kind: "measure", file: "measure.json",
    lines: ["gcc -O3, checked & timed", "vs naive + blocked loop"] },
  { n: 7, title: "ARCHIVE", who: "tool", kind: "invent", file: "run.json",
    lines: ["run.json + invent_index", "ranked by invent-rank"] },
];

// ---------------------------------------------------------------- layout
const W = 1900, H = 620;
const boxW = 220, boxH = 190, gap = 36;
const rowTop = 150;
const leftPad = (W - (7 * boxW + 6 * gap)) / 2;

const pos = {};
for (let i = 0; i < 7; i++) pos[i + 1] = { x: leftPad + i * (boxW + gap), y: rowTop };

const kindColor = (k) => (k === "invent" ? C.invent : k === "measure" ? C.measure : k === "claude" ? C.claude : C.mixed);

// ---------------------------------------------------------------- svg helpers
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const text = (x, y, s, o = {}) =>
  `<text x="${x}" y="${y}" font-size="${o.size || 14}" font-weight="${o.weight || 400}" fill="${o.fill || C.text}" ` +
  `text-anchor="${o.anchor || "start"}" font-family="${o.mono ? "'JetBrains Mono', 'Fira Code', Menlo, monospace" : "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"}" ` +
  `letter-spacing="${o.spacing || 0}" opacity="${o.opacity ?? 1}">${esc(s)}</text>`;

function card(s) {
  const { x, y } = pos[s.n];
  const col = kindColor(s.kind);
  let out = "";
  out += `<rect x="${x}" y="${y}" width="${boxW}" height="${boxH}" rx="16" fill="${C.card}" stroke="${C.cardEdge}" stroke-width="1.5" filter="url(#shadow)"/>`;
  out += `<rect x="${x}" y="${y}" width="${boxW}" height="6" rx="3" fill="${col}"/>`;
  out += `<circle cx="${x + 26}" cy="${y + 36}" r="15" fill="${col}"/>`;
  out += text(x + 26, y + 41, s.n, { size: 15, weight: 800, fill: C.bg0, anchor: "middle" });
  out += text(x + 50, y + 41, s.title, { size: 16.5, weight: 800, spacing: 0.3 });
  out += text(x + 18, y + 62, s.who, { size: 11.5, fill: col, weight: 600 });
  s.lines.forEach((l, i) => { out += text(x + 18, y + 86 + i * 17, l, { size: 11.5, fill: C.muted }); });
  const chipW = s.file.length * 6.6 + 18;
  out += `<rect x="${x + boxW - chipW - 12}" y="${y + boxH - 28}" width="${chipW}" height="18" rx="9" fill="${C.bg0}" stroke="${col}" stroke-opacity="0.6"/>`;
  out += text(x + boxW - chipW / 2 - 12, y + boxH - 15, s.file, { size: 10, mono: true, fill: col, anchor: "middle" });
  return out;
}

function arrow(d, o = {}) {
  const col = o.color || C.line;
  let out = `<path d="${d}" fill="none" stroke="${col}" stroke-width="${o.width || 2.5}" ` +
    `stroke-linecap="round" stroke-linejoin="round" marker-end="url(#head-${o.markerId || "line"})" ${o.dash ? `stroke-dasharray="${o.dash}"` : ""}/>`;
  if (o.label) {
    const [lx, ly] = o.labelAt;
    const w = o.label.length * 6.2 + 16;
    out += `<rect x="${lx - w / 2}" y="${ly - 11}" width="${w}" height="18" rx="9" fill="${C.bg0}" stroke="${col}" stroke-opacity="0.7"/>`;
    out += text(lx, ly + 3, o.label, { size: 10.5, fill: col, anchor: "middle", weight: 600 });
  }
  return out;
}

// ---------------------------------------------------------------- build
let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="${C.bg0}"/><stop offset="1" stop-color="${C.bg1}"/>
  </linearGradient>
  <filter id="shadow" x="-10%" y="-10%" width="130%" height="140%">
    <feDropShadow dx="0" dy="5" stdDeviation="7" flood-color="#000" flood-opacity="0.45"/>
  </filter>
  ${["line", "measure"].map((k) =>
    `<marker id="head-${k}" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto" markerUnits="userSpaceOnUse">
       <path d="M0,0 L9,4.5 L0,9 z" fill="${k === "line" ? C.line : C.measure}"/></marker>`).join("")}
  <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
    <path d="M40 0 L0 0 0 40" fill="none" stroke="#ffffff" stroke-opacity="0.04" stroke-width="1"/>
  </pattern>
</defs>
<rect width="100%" height="100%" fill="url(#bg)"/>
<rect width="100%" height="100%" fill="url(#grid)"/>
`;

svg += text(W / 2, 44, "crazyAI · the invent pipeline", { size: 28, weight: 800, anchor: "middle", spacing: 0.5 });
svg += text(W / 2, 70, "corpus \u2192 blend \u2192 immerse \u2192 bend \u2192 measure \u2192 archive, plus two opt-in loops back (dashed)", { size: 13, fill: C.muted, anchor: "middle" });

// forward arrows
for (let i = 1; i < 7; i++) {
  const a = pos[i], b = pos[i + 1];
  const y = a.y + boxH / 2;
  svg += arrow(`M${a.x + boxW + 3},${y} L${b.x - 5},${y}`);
}

// cards
steps.forEach((s) => { svg += card(s); });

// feedback loops beneath the row: ARCHIVE -> WORLD (evolve-corpus), ARCHIVE -> SEED (bias-from-history)
{
  const yBase = rowTop + boxH;
  const archiveCx = pos[7].x + boxW / 2;
  const worldCx = pos[3].x + boxW / 2;
  const seedCx = pos[1].x + boxW / 2;
  const yPromote = yBase + 46, ySeedLoop = yBase + 96;

  svg += arrow(`M${archiveCx - 20},${yBase + 4} L${archiveCx - 20},${yPromote} L${worldCx},${yPromote} L${worldCx},${yBase + 6}`,
    { color: C.measure, markerId: "measure", dash: "7 5", width: 2,
      label: "--evolve-corpus: new-best world \u2192 promoted into the corpus", labelAt: [(archiveCx + worldCx) / 2 + 40, yPromote - 14] });

  svg += arrow(`M${archiveCx + 20},${yBase + 4} L${archiveCx + 20},${ySeedLoop} L${seedCx},${ySeedLoop} L${seedCx},${yBase + 6}`,
    { color: C.measure, markerId: "measure", dash: "7 5", width: 2,
      label: "--bias-from-history: archived discovery \u2192 weights future draws", labelAt: [(archiveCx + seedCx) / 2 + 10, ySeedLoop + 20] });
}

// footer
svg += text(W / 2, H - 34, "every draw is seeded and logged \u00b7 both loops are opt-in and off by default \u00b7 archive/invent_<seed>_<target>/", { size: 12, fill: C.muted, anchor: "middle" });
svg += text(W / 2, H - 14, "crazyai invent --seed 42 --target matmul --provider claudecode", { size: 11.5, fill: C.line, anchor: "middle", mono: true });
svg += `</svg>`;

// ---------------------------------------------------------------- write + rasterise
const outDir = __dirname;
const svgPath = path.join(outDir, "invent_pipeline.svg");
const htmlPath = path.join(outDir, "invent_pipeline.html");
const pngPath = path.join(outDir, "invent_pipeline.png");
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
  console.error("could not rasterise: install a Chromium-family browser or ImageMagick; invent_pipeline.svg is still usable");
  process.exit(1);
}
