#!/usr/bin/env node
/**
 * crazyAI pipeline flowchart.
 *
 * Pure Node, no dependencies: builds the diagram as SVG, writes pipeline.svg
 * and pipeline.html, then rasterises to pipeline.png with headless Chrome
 * (falls back to ImageMagick `convert` if Chrome is not found).
 *
 *   node assets/flowchart/flowchart.js            # -> assets/flowchart/pipeline.{svg,html,png}
 *   make flowchart
 */

const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

// ---------------------------------------------------------------- palette
const C = {
  bg0: "#0f1022", bg1: "#1b1d3d",
  invent: "#ff7f50", inventDim: "#ff7f5033",
  measure: "#00ced1", measureDim: "#00ced133",
  mixed: "#c084fc",
  claude: "#ffbf00",
  text: "#f4f4ff", muted: "#a9abc9", line: "#8a8dbf",
  card: "#23264d", cardEdge: "#3b3f7a",
};

// ---------------------------------------------------------------- data
const steps = [
  { n: 1, title: "SEED", who: "RNG", kind: "invent", file: "seed.json",
    lines: ["draw domain, concept,", "one curated rule"] },
  { n: 2, title: "MUTATE", who: "RNG + Claude", kind: "invent", file: "mutation.json",
    lines: ["one of 7 operators: INVERT, REMOVE,", "EXTRAPOLATE, TRANSPOSE, COMPOSE,", "QUANTIFY, SUBSTITUTE · depth 1–5"] },
  { n: 3, title: "GENERATE", who: "Claude + toolkit", kind: "mixed", file: "artifact.md",
    lines: ["story · formula · plan · statmodel", "questions · debate", "every number from a tool call"] },
  { n: 4, title: "FORMALISE", who: "Claude + measure", kind: "measure", file: "formal.md",
    lines: ["equations, model specs,", "dimension & consistency checks", "pasted verbatim"] },
  { n: 5, title: "SELF-CHECK", who: "Claude + ground", kind: "measure", file: "key.json",
    lines: ["answer key: where the mutation", "enters, why impossible, what", "would make it possible"] },
  { n: 6, title: "CROSS-EXAMINE", who: "fresh Claude, no tools", kind: "measure", file: "verdicts.json",
    lines: ["N reviews, shuffled framings", "sees artifact only", "no shared context"] },
  { n: 7, title: "SCORE", who: "judge", kind: "measure", file: "score.json",
    lines: ["detection · acceptance · hedge", "false-flaw · confidence-when-wrong", "discovery = rigor × novelty × cost"] },
  { n: 8, title: "ARCHIVE", who: "tool", kind: "invent", file: "run.json",
    lines: ["run folder + index.jsonl", "rng log, timings, backend", "rank · report · compare"] },
];

// ---------------------------------------------------------------- layout
const W = 1700, H = 900;
const boxW = 270, boxH = 168, gap = 42;
const rowTop = 190, rowBottom = 560;
const leftPad = (W - (5 * boxW + 4 * gap)) / 2;

const pos = {};
for (let i = 0; i < 5; i++) pos[i + 1] = { x: leftPad + i * (boxW + gap), y: rowTop };
// bottom row: 8 7 6 under 1 2 3? No - 6 sits under 5, 7 under 4, 8 under 3, leaving room for the legend on the left
pos[6] = { x: pos[5].x, y: rowBottom };
pos[7] = { x: pos[4].x, y: rowBottom };
pos[8] = { x: pos[3].x, y: rowBottom };

const kindColor = (k) => (k === "invent" ? C.invent : k === "measure" ? C.measure : C.mixed);

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
  // number badge
  out += `<circle cx="${x + 30}" cy="${y + 40}" r="17" fill="${col}"/>`;
  out += text(x + 30, y + 46, s.n, { size: 18, weight: 800, fill: C.bg0, anchor: "middle" });
  out += text(x + 58, y + 46, s.title, { size: 19, weight: 800, spacing: 0.5 });
  out += text(x + 58, y + 66, s.who, { size: 12, fill: s.who.includes("Claude") ? C.claude : C.muted, weight: 600 });
  s.lines.forEach((l, i) => { out += text(x + 20, y + 94 + i * 18, l, { size: 12.5, fill: C.muted }); });
  // file chip
  const chipW = s.file.length * 7.4 + 22;
  out += `<rect x="${x + boxW - chipW - 14}" y="${y + boxH - 30}" width="${chipW}" height="20" rx="10" fill="${C.bg0}" stroke="${col}" stroke-opacity="0.6"/>`;
  out += text(x + boxW - chipW / 2 - 14, y + boxH - 16, s.file, { size: 11, mono: true, fill: col, anchor: "middle" });
  return out;
}

// arrow between two points with optional label; d is an SVG path
function arrow(d, o = {}) {
  const col = o.color || C.line;
  let out = `<path d="${d}" fill="none" stroke="${col}" stroke-width="${o.width || 3}" ` +
    `stroke-linecap="round" stroke-linejoin="round" marker-end="url(#head-${o.color === C.invent ? "invent" : o.color === C.claude ? "claude" : "line"})" ${o.dash ? `stroke-dasharray="${o.dash}"` : ""}/>`;
  if (o.label) {
    const [lx, ly] = o.labelAt;
    const w = o.label.length * 6.6 + 18;
    out += `<rect x="${lx - w / 2}" y="${ly - 12}" width="${w}" height="20" rx="10" fill="${C.bg0}" stroke="${col}" stroke-opacity="0.7"/>`;
    out += text(lx, ly + 2, o.label, { size: 11, fill: col, anchor: "middle", weight: 600 });
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
    <feDropShadow dx="0" dy="6" stdDeviation="8" flood-color="#000" flood-opacity="0.45"/>
  </filter>
  <filter id="glow"><feGaussianBlur stdDeviation="6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  ${["line", "invent", "claude"].map((k) =>
    `<marker id="head-${k}" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="userSpaceOnUse">
       <path d="M0,0 L10,5 L0,10 z" fill="${k === "line" ? C.line : k === "invent" ? C.invent : C.claude}"/></marker>`).join("")}
  <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
    <path d="M40 0 L0 0 0 40" fill="none" stroke="#ffffff" stroke-opacity="0.04" stroke-width="1"/>
  </pattern>
</defs>
<rect width="100%" height="100%" fill="url(#bg)"/>
<rect width="100%" height="100%" fill="url(#grid)"/>
`;

// title
svg += text(W / 2, 62, "crazyAI · the discovery pipeline", { size: 34, weight: 800, anchor: "middle", spacing: 0.5 });
svg += text(W / 2, 92, "one seed → one mutated rule → one artifact that is rigorous everywhere except in one place → measured by a session that has never seen it", { size: 14, fill: C.muted, anchor: "middle" });

// session brackets (behind cards)
const genX0 = pos[3].x - 18, genX1 = pos[5].x + boxW + 18;
svg += `<rect x="${genX0}" y="${rowTop - 46}" width="${genX1 - genX0}" height="${boxH + 74}" rx="22" fill="${C.mixed}" fill-opacity="0.06" stroke="${C.mixed}" stroke-opacity="0.45" stroke-dasharray="6 6"/>`;
svg += text(genX0 + 18, rowTop - 22, "GENERATOR SESSION  ·  invent ⇄ measure loop until the measure tools report exactly one flaw", { size: 12.5, fill: C.mixed, weight: 700, spacing: 0.4 });

const exX0 = pos[6].x - 18, exX1 = pos[6].x + boxW + 18;
svg += `<rect x="${exX0}" y="${rowBottom - 46}" width="${exX1 - exX0}" height="${boxH + 74}" rx="22" fill="${C.measure}" fill-opacity="0.06" stroke="${C.measure}" stroke-opacity="0.45" stroke-dasharray="6 6"/>`;
svg += text(exX0 + 18, rowBottom - 22, "FRESH SESSION  ·  no shared context", { size: 12.5, fill: C.measure, weight: 700, spacing: 0.4 });

// forward arrows 1→2→3→4→5
for (let i = 1; i < 5; i++) {
  const a = pos[i], b = pos[i + 1];
  const y = a.y + boxH / 2;
  svg += arrow(`M${a.x + boxW + 4},${y} L${b.x - 6},${y}`);
}
// 5 → 6 (down)
{
  const x = pos[5].x + boxW / 2;
  svg += arrow(`M${x},${pos[5].y + boxH + 4} L${x},${rowBottom - 52}`, { label: "artifact + formalisation only", labelAt: [x, (pos[5].y + boxH + rowBottom - 48) / 2] });
}
// 6 → 7 → 8 (leftwards)
for (const [a, b] of [[6, 7], [7, 8]]) {
  const y = rowBottom + boxH / 2;
  svg += arrow(`M${pos[a].x - 4},${y} L${pos[b].x + boxW + 6},${y}`);
}
// regenerate loop: 5 → back over the top → 3
{
  const x5 = pos[5].x + boxW / 2 + 70, x3 = pos[3].x + boxW / 2 - 70;
  const yTop = rowTop - 70;
  svg += arrow(`M${x5},${rowTop - 2} L${x5},${yTop} L${x3},${yTop} L${x3},${rowTop - 8}`,
    { color: C.invent, dash: "8 6", width: 2.5, label: "unintended flaw found → regenerate (≤ 2×)", labelAt: [(x5 + x3) / 2, yTop] });
}
// invent ⇄ measure mini-loop under 3-4
{
  const y = rowTop + boxH + 34;
  const x0 = pos[3].x + 40, x1 = pos[4].x + boxW - 40;
  svg += `<path d="M${x0},${y} C${x0 + 120},${y + 40} ${x1 - 120},${y + 40} ${x1},${y}" fill="none" stroke="${C.claude}" stroke-width="2" stroke-dasharray="4 5" opacity="0.8"/>`;
  svg += text((x0 + x1) / 2, y + 44, "invent → draft → measure → fix → measure …", { size: 11.5, fill: C.claude, anchor: "middle", weight: 600 });
}

// cards on top
steps.forEach((s) => { svg += card(s); });

// left-bottom: toolkit legend
{
  const x = leftPad, y = rowBottom - 20;
  const w = pos[8].x - 30 - x;
  svg += `<rect x="${x}" y="${y}" width="${w}" height="${boxH + 20}" rx="16" fill="${C.bg0}" fill-opacity="0.55" stroke="${C.cardEdge}"/>`;
  svg += text(x + 20, y + 32, "THE TOOLKIT  (55 tools, called by Claude through tool use)", { size: 13, weight: 800, spacing: 0.5 });
  svg += `<rect x="${x + 20}" y="${y + 48}" width="14" height="14" rx="4" fill="${C.invent}"/>`;
  svg += text(x + 42, y + 60, "INVENT · randomness & chaos", { size: 13, weight: 700, fill: C.invent });
  svg += text(x + 42, y + 80, "chaos · mutate · unconventional · transform · disguise", { size: 12, fill: C.muted });
  svg += text(x + 42, y + 97, "seeded draws only — the one place chaos is allowed in", { size: 11.5, fill: C.muted, opacity: 0.8 });
  svg += `<rect x="${x + 20}" y="${y + 114}" width="14" height="14" rx="4" fill="${C.measure}"/>`;
  svg += text(x + 42, y + 126, "MEASURE · truth & solid ground", { size: 13, weight: 700, fill: C.measure });
  svg += text(x + 42, y + 146, "symbolic · stats · logic · narrative · ground · novelty · archive", { size: 12, fill: C.muted });
  svg += text(x + 42, y + 163, "pure functions — they compute, the model reasons", { size: 11.5, fill: C.muted, opacity: 0.8 });
}

// footer
svg += text(W / 2, H - 46, "every random choice is seeded and logged · every step writes a file · same seed + same model = same artifact", { size: 13, fill: C.muted, anchor: "middle" });
svg += text(W / 2, H - 22, "archive/run_<seed>_<generator>/  ·  crazyai run --seed 42 --generator formula", { size: 12, fill: C.line, anchor: "middle", mono: true });
svg += `</svg>`;

// ---------------------------------------------------------------- write + rasterise
const outDir = __dirname;
const svgPath = path.join(outDir, "pipeline.svg");
const htmlPath = path.join(outDir, "pipeline.html");
const pngPath = path.join(outDir, "pipeline.png");
fs.writeFileSync(svgPath, svg);
fs.writeFileSync(htmlPath, `<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;background:${C.bg0}}svg{display:block}</style></head><body>${svg}</body></html>`);
console.log("wrote", svgPath, "and", htmlPath);

const scale = Number(process.env.SCALE || 2);
const chrome = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"];
let done = false;
for (const bin of chrome) {
  try {
    execFileSync(bin, ["--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-sandbox",
      `--force-device-scale-factor=${scale}`, `--window-size=${W},${H + 140}`, `--screenshot=${pngPath}`, `file://${htmlPath}`],
      { stdio: "ignore", timeout: 60000 });
    done = fs.existsSync(pngPath);
    if (done) {
      // the window includes browser chrome, so the capture is taller than the page: crop to W x H
      try { execFileSync("convert", [pngPath, "-crop", `${W * scale}x${H * scale}+0+0`, "+repage", pngPath], { stdio: "ignore" }); }
      catch (e) { /* ImageMagick missing: keep the uncropped capture */ }
    }
    if (done) { console.log("wrote", pngPath, `(${W * scale}x${H * scale}, via ${bin})`); break; }
  } catch (e) { /* try next */ }
}
if (!done) {
  try {
    execFileSync("convert", ["-density", String(96 * scale), svgPath, pngPath], { stdio: "ignore" });
    console.log("wrote", pngPath, "(via ImageMagick)");
  } catch (e) {
    console.error("could not rasterise: install Chrome or ImageMagick; pipeline.svg is still usable");
    process.exit(1);
  }
}
