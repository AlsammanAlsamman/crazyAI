"""Aggregate statistics over the archive, Markdown report, radar chart (SVG, no plotting dependency)."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from crazyai.config import ARCHIVE_DIR

METRICS = ["detection_rate", "acceptance_rate", "false_flaw_rate", "hedge_rate", "confidence_when_wrong",
           "depth_of_detection", "discovery_value"]


def load_index(archive_dir: Path | str = ARCHIVE_DIR) -> list[dict[str, Any]]:
    p = Path(archive_dir) / "index.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    # keep the latest row per (seed, generator)
    latest: dict[tuple[int, str], dict[str, Any]] = {}
    for r in rows:
        latest[(r["seed"], r["generator"])] = r
    return list(latest.values())


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    return round(statistics.mean(xs), 3), round(statistics.pstdev(xs), 3) if len(xs) > 1 else 0.0


def aggregate(rows: list[dict[str, Any]], by: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(by))].append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        out[g] = {"n": len(rs)}
        for m in METRICS:
            mean, sd = _mean_sd([float(r["metrics"].get(m, 0)) for r in rs])
            out[g][m] = {"mean": mean, "sd": sd}
    return out


def rank(rows: list[dict[str, Any]], by: str = "discovery_value", top: int = 20) -> list[dict[str, Any]]:
    key = (lambda r: r["metrics"].get(by, 0)) if by in METRICS else (lambda r: r["key"].get(by, 0) or 0)
    return sorted(rows, key=key, reverse=True)[:top]


def radar_svg(values: dict[str, float], title: str = "", size: int = 360) -> str:
    keys = list(values)
    n = len(keys)
    cx = cy = size / 2
    r = size * 0.36
    def pt(i: int, frac: float) -> tuple[float, float]:
        a = -math.pi / 2 + 2 * math.pi * i / n
        return cx + r * frac * math.cos(a), cy + r * frac * math.sin(a)
    rings = "".join(
        f'<polygon points="{" ".join(f"{pt(i, f)[0]:.1f},{pt(i, f)[1]:.1f}" for i in range(n))}" '
        f'fill="none" stroke="#999" stroke-width="0.6"/>' for f in (0.25, 0.5, 0.75, 1.0))
    spokes = "".join(f'<line x1="{cx}" y1="{cy}" x2="{pt(i,1)[0]:.1f}" y2="{pt(i,1)[1]:.1f}" stroke="#bbb" stroke-width="0.6"/>'
                     for i in range(n))
    poly = " ".join(f"{pt(i, max(0.0, min(1.0, float(values[k]))))[0]:.1f},{pt(i, max(0.0, min(1.0, float(values[k]))))[1]:.1f}"
                    for i, k in enumerate(keys))
    labels = "".join(
        f'<text x="{pt(i,1.18)[0]:.1f}" y="{pt(i,1.18)[1]:.1f}" font-size="10" text-anchor="middle" '
        f'font-family="sans-serif" fill="#333">{k}</text>' for i, k in enumerate(keys))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
            f'<rect width="100%" height="100%" fill="white"/>{rings}{spokes}'
            f'<polygon points="{poly}" fill="#8A2BE2" fill-opacity="0.35" stroke="#4B0082" stroke-width="1.5"/>'
            f'<text x="{cx}" y="16" font-size="12" text-anchor="middle" font-family="sans-serif">{title}</text>{labels}</svg>')


def markdown_report(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "# crazyAI report\n\nThe archive is empty.\n"
    lines = ["# crazyAI report", "", f"Runs: {len(rows)}", ""]
    for by in ("generator", "operator", "domain"):
        agg = aggregate(rows, by)
        lines += [f"## By {by}", "", "| " + by + " | n | " + " | ".join(METRICS) + " |",
                  "|" + "---|" * (len(METRICS) + 2)]
        for g, a in agg.items():
            cells = [f"{a[m]['mean']:.2f} ± {a[m]['sd']:.2f}" for m in METRICS]
            lines.append(f"| {g} | {a['n']} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Top candidates for human review (by discovery value)", "",
              "| seed | generator | rule | operator | discovery | detection | acceptance |", "|---|---|---|---|---|---|---|"]
    for r in rank(rows, "discovery_value", 10):
        m = r["metrics"]
        lines.append(f"| {r['seed']} | {r['generator']} | {r['rule_id']} | {r['operator']} | "
                     f"{m['discovery_value']:.3f} | {m['detection_rate']:.2f} | {m['acceptance_rate']:.2f} |")
    lines.append("")
    return "\n".join(lines)


def compare(run_a: Path, run_b: Path) -> dict[str, Any]:
    a = json.loads((Path(run_a) / "run.json").read_text())
    b = json.loads((Path(run_b) / "run.json").read_text())
    diff = {m: {"a": a["metrics"].get(m), "b": b["metrics"].get(m),
                "delta": round(float(b["metrics"].get(m, 0)) - float(a["metrics"].get(m, 0)), 3)} for m in METRICS}
    return {"a": {k: a[k] for k in ("seed", "generator", "rule_id", "operator", "model")},
            "b": {k: b[k] for k in ("seed", "generator", "rule_id", "operator", "model")}, "metrics": diff}
