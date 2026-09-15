"""Tiny SVG toolkit for the README figures: same palette as the flowchart, no plotting dependency.

Rasterisation: headless Chrome (google-chrome / chromium), ImageMagick fallback.
"""

from __future__ import annotations

import html
import math
import shutil
import subprocess
from pathlib import Path

C = {
    "bg0": "#0f1022", "bg1": "#1b1d3d", "card": "#23264d", "edge": "#3b3f7a",
    "invent": "#ff7f50", "measure": "#00ced1", "mixed": "#c084fc", "claude": "#ffbf00",
    "text": "#f4f4ff", "muted": "#a9abc9", "line": "#8a8dbf", "red": "#ff5470", "green": "#4ade80",
}
PALETTE = ["#00ced1", "#ff7f50", "#c084fc", "#ffbf00", "#4ade80", "#ff5470", "#60a5fa", "#f472b6"]
FONT = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'Fira Code', Menlo, monospace"


class SVG:
    def __init__(self, w: int, h: int, title: str = "", subtitle: str = ""):
        self.w, self.h = w, h
        self.parts: list[str] = []
        self.parts.append(f"""<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{C['bg0']}"/><stop offset="1" stop-color="{C['bg1']}"/></linearGradient>
<filter id="shadow" x="-10%" y="-10%" width="130%" height="140%"><feDropShadow dx="0" dy="5" stdDeviation="7" flood-color="#000" flood-opacity="0.45"/></filter>
<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0 L0 0 0 40" fill="none" stroke="#fff" stroke-opacity="0.04"/></pattern>
{''.join(f'<marker id="head-{k}" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L10,5 L0,10 z" fill="{v}"/></marker>' for k, v in C.items())}
</defs>
<rect width="100%" height="100%" fill="url(#bg)"/><rect width="100%" height="100%" fill="url(#grid)"/>""")
        if title:
            self.text(w / 2, 46, title, size=28, weight=800, anchor="middle")
        if subtitle:
            self.text(w / 2, 74, subtitle, size=14, fill=C["muted"], anchor="middle")

    # -- primitives ----------------------------------------------------------------
    def text(self, x, y, s, size=13, fill=None, anchor="start", weight=400, mono=False, opacity=1.0, rotate=None):
        tr = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill or C["text"]}" '
            f'text-anchor="{anchor}" font-family="{MONO if mono else FONT}" opacity="{opacity}"{tr}>{html.escape(str(s))}</text>')

    def rect(self, x, y, w, h, fill=None, stroke=None, rx=0, opacity=1.0, sw=1.0, dash=None, shadow=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        f = ' filter="url(#shadow)"' if shadow else ""
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill or "none"}" '
                          f'stroke="{stroke or "none"}" stroke-width="{sw}" opacity="{opacity}"{d}{f}/>')

    def line(self, x1, y1, x2, y2, stroke=None, sw=1.5, dash=None, opacity=1.0, arrow=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#head-{arrow})"' if arrow else ""
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke or C["line"]}" '
                          f'stroke-width="{sw}" stroke-linecap="round" opacity="{opacity}"{d}{m}/>')

    def path(self, d, stroke=None, sw=2, fill="none", dash=None, opacity=1.0, arrow=None):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#head-{arrow})"' if arrow else ""
        self.parts.append(f'<path d="{d}" fill="{fill}" stroke="{stroke or C["line"]}" stroke-width="{sw}" '
                          f'stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"{da}{m}/>')

    def circle(self, cx, cy, r, fill, stroke=None, opacity=1.0, sw=1.0):
        self.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.2f}" fill="{fill}" stroke="{stroke or "none"}" '
                          f'stroke-width="{sw}" opacity="{opacity}"/>')

    def card(self, x, y, w, h, title=None, color=None, subtitle=None):
        self.rect(x, y, w, h, fill=C["card"], stroke=C["edge"], rx=14, sw=1.5, shadow=True)
        if color:
            self.rect(x, y, w, 5, fill=color, rx=2.5)
        if title:
            self.text(x + 16, y + 30, title, size=15, weight=800)
        if subtitle:
            self.text(x + 16, y + 48, subtitle, size=11.5, fill=C["muted"])

    def chip(self, x, y, s, color, mono=True, anchor="start"):
        w = len(s) * 7.2 + 20
        x0 = x if anchor == "start" else x - w / 2 if anchor == "middle" else x - w
        self.rect(x0, y - 11, w, 20, fill=C["bg0"], stroke=color, rx=10, opacity=0.95)
        self.text(x0 + w / 2, y + 3, s, size=11, fill=color, anchor="middle", mono=mono, weight=600)
        return x0 + w

    def step_flow(self, x, y, w, items, color_of=None):
        """Row of small boxes with arrows: items = [(label, sub)]. Returns box height."""
        n = len(items)
        gap = 26
        bw = (w - gap * (n - 1)) / n
        bh = 56
        for i, (label, sub) in enumerate(items):
            bx = x + i * (bw + gap)
            col = color_of(i) if color_of else C["line"]
            self.rect(bx, y, bw, bh, fill=C["card"], stroke=col, rx=10, sw=1.5)
            self.text(bx + bw / 2, y + 23, label, size=12.5, weight=700, anchor="middle")
            self.text(bx + bw / 2, y + 42, sub, size=10.5, fill=C["muted"], anchor="middle", mono=True)
            if i < n - 1:
                self.line(bx + bw + 3, y + bh / 2, bx + bw + gap - 4, y + bh / 2, sw=2, arrow="line")
        return bh

    # -- axes -----------------------------------------------------------------------
    def axes(self, x, y, w, h, xlim, ylim, xlabel="", ylabel="", xticks=None, yticks=None, xfmt=str, yfmt=str, logx=False):
        ax = Axes(self, x, y, w, h, xlim, ylim, logx)
        self.rect(x, y, w, h, fill=C["bg0"], stroke=C["edge"], rx=6, opacity=0.6)
        for t in (xticks or []):
            px = ax.px(t)
            self.line(px, y, px, y + h, stroke="#fff", sw=0.6, opacity=0.08)
            self.text(px, y + h + 16, xfmt(t), size=10.5, fill=C["muted"], anchor="middle", mono=True)
        for t in (yticks or []):
            py = ax.py(t)
            self.line(x, py, x + w, py, stroke="#fff", sw=0.6, opacity=0.08)
            self.text(x - 8, py + 4, yfmt(t), size=10.5, fill=C["muted"], anchor="end", mono=True)
        if xlabel:
            self.text(x + w / 2, y + h + 34, xlabel, size=11.5, fill=C["muted"], anchor="middle")
        if ylabel:
            self.text(x - 44, y + h / 2, ylabel, size=11.5, fill=C["muted"], anchor="middle", rotate=-90)
        return ax

    def render(self) -> str:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
                + "".join(self.parts) + "</svg>")


class Axes:
    def __init__(self, svg: SVG, x, y, w, h, xlim, ylim, logx=False):
        self.s, self.x, self.y, self.w, self.h, self.xlim, self.ylim, self.logx = svg, x, y, w, h, xlim, ylim, logx

    def px(self, v):
        a, b = self.xlim
        if self.logx:
            a, b, v = math.log10(a), math.log10(b), math.log10(max(v, 1e-12))
        return self.x + (v - a) / (b - a) * self.w

    def py(self, v):
        a, b = self.ylim
        return self.y + self.h - (v - a) / (b - a) * self.h

    def polyline(self, xs, ys, stroke, sw=2.5, dash=None, opacity=1.0):
        d = "M" + " L".join(f"{self.px(x):.1f},{self.py(y):.1f}" for x, y in zip(xs, ys))
        self.s.path(d, stroke=stroke, sw=sw, dash=dash, opacity=opacity)

    def area(self, xs, y0s, y1s, fill, opacity=0.35):
        top = " L".join(f"{self.px(x):.1f},{self.py(y):.1f}" for x, y in zip(xs, y1s))
        bot = " L".join(f"{self.px(x):.1f},{self.py(y):.1f}" for x, y in zip(reversed(xs), reversed(y0s)))
        self.s.path(f"M{top} L{bot} Z", stroke="none", fill=fill, opacity=opacity)

    def scatter(self, xs, ys, colors, radii, opacity=0.9):
        for x, y, c, r in zip(xs, ys, colors, radii):
            self.s.circle(self.px(x), self.py(y), r, c, opacity=opacity)

    def bars(self, xs, heights, width, fill, opacity=0.9):
        for x, hgt in zip(xs, heights):
            x0, x1 = self.px(x - width / 2), self.px(x + width / 2)
            self.s.rect(x0, self.py(hgt), x1 - x0, self.py(0) - self.py(hgt), fill=fill, opacity=opacity, rx=2)


def rasterise(svg_text: str, out_png: Path, w: int, h: int, scale: int = 2) -> str:
    out_svg = out_png.with_suffix(".svg")
    out_html = out_png.with_suffix(".html")
    out_svg.write_text(svg_text)
    out_html.write_text(f'<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;background:{C["bg0"]}}}'
                        f'svg{{display:block}}</style></head><body>{svg_text}</body></html>')
    for binary in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        if shutil.which(binary):
            subprocess.run([binary, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-sandbox",
                            f"--force-device-scale-factor={scale}", f"--window-size={w},{h + 140}",
                            f"--screenshot={out_png}", f"file://{out_html}"], check=False, capture_output=True, timeout=90)
            if out_png.exists():
                if shutil.which("convert"):
                    subprocess.run(["convert", str(out_png), "-crop", f"{w*scale}x{h*scale}+0+0", "+repage", str(out_png)],
                                   check=False, capture_output=True)
                out_html.unlink(missing_ok=True)
                return binary
    if shutil.which("convert"):
        subprocess.run(["convert", "-density", str(96 * scale), str(out_svg), str(out_png)], check=False)
        out_html.unlink(missing_ok=True)
        return "convert"
    out_html.unlink(missing_ok=True)
    return "none"
