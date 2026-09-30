"""Report for `crazyai diagnose-return` (diagnosis kit, return-path probe).

Re-times every kernel REPEATS times, interleaved, in one sitting (the batch itself ran three
variants in parallel, so its own timings are noisy), then reports per variant:

- exact, wins (exact and > 1.05x), mean fresh speedup, per target
- idea survival (full / partial / none) and fallback rate, from the blind judge
- paired tests over the same sources: Wilcoxon signed-rank on fresh speedup, on survival
  (full=2, partial=1, none=0), and an exact sign test on fallback
- does survival predict speed? mean within-target speedup percentile by survival level

    python examples/12_diagnose_report.py [archive_dir] [repeats]      (repeats=0: skip re-timing)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

from scipy.stats import binomtest, rankdata, wilcoxon

from crazyai.targets import get_target
from crazyai.toolkit.registry import build_toolkit

ORDER = ["orig", "rp0_current", "rp1_no_known", "rp2_faithful"]
SURV = {"full": 2, "partial": 1, "none": 0}


def kernel_path(root: Path, archive: Path, variant: str, source: str) -> Path:
    return (archive / source / "artifact.c") if variant == "orig" else (root / variant / source / "artifact.c")


def load_rows(root: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    for variant in ORDER:
        for d in sorted((root / variant).glob("*")) if (root / variant).exists() else []:
            j = json.loads((d / "judge.json").read_text(encoding="utf-8")) if (d / "judge.json").exists() else {}
            rows[(variant, d.name)] = {"variant": variant, "source": d.name, "target": d.name.rsplit("_", 1)[1],
                                       "survival": j.get("survival"), "fallback": j.get("fallback"),
                                       "core_technique": j.get("core_technique", "")}
    return rows


def retime(rows: dict, root: Path, archive: Path, repeats: int) -> None:
    tk = build_toolkit(0)
    srcs = {k: kernel_path(root, archive, *k) for k in rows}
    srcs = {k: p.read_text(encoding="utf-8") for k, p in srcs.items() if p.exists()}
    runs: dict = {k: [] for k in srcs}
    status: dict = {}
    for _ in range(repeats):
        for k, code in srcs.items():
            r = tk.call(get_target(rows[k]["target"]).measure_tool, {"source": code, "budget": 0.3})
            status[k] = r.get("status", "ERROR")
            runs[k].append(r.get("value", 0.0) or 0.0)
    for k, row in rows.items():
        row["status"] = status.get(k, "NO_ARTIFACT")
        row["runs"] = [round(x, 3) for x in runs.get(k, [])]
        row["fresh"] = round(mean(runs[k]), 3) if runs.get(k) else 0.0
        if row["status"] != "exact":
            row["fresh"] = 0.0          # a wrong or crashed kernel scores 0, as everywhere in crazyAI


def main() -> None:
    archive = Path(sys.argv[1] if len(sys.argv) > 1 else "archive")
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    root = archive / "diagnose_return"
    rows = load_rows(root)
    cache = root / "remeasure.json"
    if repeats > 0:
        retime(rows, root, archive, repeats)
        cache.write_text(json.dumps([rows[k] for k in sorted(rows)], indent=1), encoding="utf-8")
    else:
        fresh = {(r["variant"], r["source"]): r for r in json.loads(cache.read_text(encoding="utf-8"))}
        for k, row in rows.items():
            row.update({f: fresh[k][f] for f in ("status", "runs", "fresh")} if k in fresh else {"status": None, "fresh": 0.0})

    variants = [v for v in ORDER if any(k[0] == v for k in rows)]
    targets = sorted({r["target"] for r in rows.values()})
    print("\n== fresh speedup: mean (exact / wins / n) per target ==")
    print(f"{'variant':<14}" + "".join(f"{t:>26}" for t in targets))
    for v in variants:
        cells = []
        for t in targets:
            g = [r for r in rows.values() if r["variant"] == v and r["target"] == t]
            ex = sum(r["status"] == "exact" for r in g)
            wins = sum(r["status"] == "exact" and r["fresh"] > 1.05 for r in g)
            cells.append(f"{mean(r['fresh'] for r in g):8.3f}x ({ex}/{wins}/{len(g)})" if g else "-")
        print(f"{v:<14}" + "".join(f"{c:>26}" for c in cells))

    print("\n== idea survival (blind judge) ==")
    for v in variants:
        g = [r for r in rows.values() if r["variant"] == v]
        c = {s: sum(r["survival"] == s for r in g) for s in SURV}
        fb = sum(r["fallback"] is True for r in g)
        print(f"{v:<14} full {c['full']:>2}  partial {c['partial']:>2}  none {c['none']:>2}   fallback {fb:>2}/{len(g)}")

    print("\n== paired tests over shared sources (two-sided) ==")
    pairs = [("orig", "rp0_current"), ("rp0_current", "rp1_no_known"), ("rp1_no_known", "rp2_faithful"),
             ("rp0_current", "rp2_faithful")]
    for a, b in pairs:
        if a not in variants or b not in variants:
            continue
        shared = sorted({s for (v, s) in rows if v == a} & {s for (v, s) in rows if v == b})
        xa = [rows[(a, s)]["fresh"] for s in shared]
        xb = [rows[(b, s)]["fresh"] for s in shared]
        sa = [SURV.get(rows[(a, s)]["survival"], 0) for s in shared]
        sb = [SURV.get(rows[(b, s)]["survival"], 0) for s in shared]
        up = sum(y > x for x, y in zip(xa, xb))
        line = f"{a} -> {b} (n={len(shared)}): speedup {mean(xa):.3f} -> {mean(xb):.3f}x, better on {up}/{len(shared)}"
        try:
            line += f", Wilcoxon p={wilcoxon(xa, xb).pvalue:.3f}"
        except ValueError:
            line += ", Wilcoxon n/a"
        try:
            line += f"; survival {mean(sa):.2f} -> {mean(sb):.2f}, p={wilcoxon(sa, sb).pvalue:.3f}"
        except ValueError:
            line += f"; survival {mean(sa):.2f} -> {mean(sb):.2f}, p n/a (no differences)"
        fa = [rows[(a, s)]["fallback"] is True for s in shared]
        fbb = [rows[(b, s)]["fallback"] is True for s in shared]
        gone, new = sum(x and not y for x, y in zip(fa, fbb)), sum(y and not x for x, y in zip(fa, fbb))
        p = binomtest(gone, gone + new).pvalue if gone + new else 1.0
        line += f"; fallback {sum(fa)} -> {sum(fbb)} (sign p={p:.3f})"
        print(line)

    print("\n== does survival predict speed? mean within-target speedup percentile ==")
    pct = {}
    for t in targets:
        g = [k for k, r in rows.items() if r["target"] == t]
        ranks = rankdata([rows[k]["fresh"] for k in g])
        for k, rk in zip(g, ranks):
            pct[k] = (rk - 1) / max(len(g) - 1, 1)
    for s in SURV:
        g = [pct[k] for k, r in rows.items() if r["survival"] == s]
        if g:
            print(f"survival={s:<8} n={len(g):>2}  mean percentile {mean(g):.2f}")
    for fb in (True, False):
        g = [pct[k] for k, r in rows.items() if r["fallback"] is fb]
        if g:
            print(f"fallback={str(fb):<8} n={len(g):>2}  mean percentile {mean(g):.2f}")
    out = root / "report.json"
    out.write_text(json.dumps([rows[k] for k in sorted(rows)], indent=1), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
