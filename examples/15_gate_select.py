"""Gate + selection on hash: does gating the candidates keep the native's idea without losing speed?

For each hash source, two arms each made K candidates with `crazyai diagnose-return`:

    plain   c0_today__k1..kK   today's normal translate prompt
    gate    i3_gate__k1..kK    the same prompt, judged, and retried on a fallback

Every candidate is re-timed REPEATS times (interleaved, idle machine) and run through the
SMHasher-style quality check (examples/11_hash_quality.py). Per (source, arm) the winner is the
fastest candidate that is exact AND passes the quality check. Reported: the winner's speed, whether
the winner kept the native's idea (blind judge: not a fallback), and paired tests between the arms.

    python examples/15_gate_select.py [archive_dir] [repeats] [K]
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from statistics import mean

from scipy.stats import binomtest, wilcoxon

from crazyai.targets import get_target
from crazyai.toolkit.registry import build_toolkit

SOURCES = [f"invent_{s}_hash" for s in (7201, 7202, 7203, 7204, 7221, 7222, 7223, 7224, 7225)]
ARMS = {"plain": "c0_today", "gate": "i3_gate"}

_spec = importlib.util.spec_from_file_location("hq", Path(__file__).with_name("11_hash_quality.py"))
hq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hq)


def quality(path: Path) -> dict:
    try:
        r = hq.check(path)
    except Exception as exc:  # noqa: BLE001 - e.g. the binary is quarantined by endpoint security
        return {"pass": None, "error": str(exc)[-200:]}
    return {"pass": r.get("pass"), "worst_bias": r.get("worst_bias"), "twobit_coll64": r.get("twobit_coll64")}


def main() -> None:
    archive = Path(sys.argv[1] if len(sys.argv) > 1 else "archive")
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    root = archive / "diagnose_return"
    tk = build_toolkit(0)
    tool = get_target("hash").measure_tool

    cands = []
    for src in SOURCES:
        for arm, base in ARMS.items():
            for i in range(1, k + 1):
                d = root / f"{base}__k{i}" / src
                if not (d / "artifact.c").exists():
                    continue
                j = json.loads((d / "judge.json").read_text(encoding="utf-8")) if (d / "judge.json").exists() else {}
                cands.append({"source": src, "arm": arm, "k": i, "dir": d, "fallback": j.get("fallback"),
                              "survival": j.get("survival"), "core": j.get("core_technique", ""), "runs": []})
    code = {id(c): (c["dir"] / "artifact.c").read_text(encoding="utf-8") for c in cands}
    for _ in range(repeats):
        for c in cands:
            r = tk.call(tool, {"source": code[id(c)], "budget": 0.3})
            c["status"] = r.get("status", "ERROR")
            c["runs"].append(r.get("value", 0.0) or 0.0)
    for c in cands:
        c["fresh"] = round(mean(c["runs"]), 3) if c["runs"] and c["status"] == "exact" else 0.0
        c["quality"] = quality(c["dir"] / "artifact.c") if c["status"] == "exact" else {"pass": False}
        c["ok"] = c["status"] == "exact" and c["quality"].get("pass") is True

    win = {}
    print(f"{'source':<20}" + "".join(f"{a + ' winner':>34}" for a in ARMS))
    for src in SOURCES:
        cells = []
        for arm in ARMS:
            pool = [c for c in cands if c["source"] == src and c["arm"] == arm and c["ok"]]
            w = max(pool, key=lambda c: c["fresh"]) if pool else None
            win[(src, arm)] = w
            cells.append(f"{w['fresh']:8.2f}x k{w['k']} {'KEPT IDEA' if w['fallback'] is False else 'fallback':>9} ({len(pool)} ok)"
                         if w else "none passed")
        print(f"{src:<20}" + "".join(f"{c:>34}" for c in cells))

    print()
    for arm in ARMS:
        ws = [win[(s, arm)] for s in SOURCES]
        n_ok = sum(w is not None for w in ws)
        kept = sum(w is not None and w["fallback"] is False for w in ws)
        allc = [c for c in cands if c["arm"] == arm]
        print(f"{arm:<6} winners {n_ok}/{len(SOURCES)}  mean winner speed {mean(w['fresh'] if w else 0 for w in ws):.2f}x  "
              f"winner kept the idea {kept}/{len(SOURCES)}  |  candidates: {len(allc)}, exact+quality {sum(c['ok'] for c in allc)}, "
              f"non-fallback {sum(c['fallback'] is False for c in allc)}, blocked {sum(c['quality'].get('pass') is None for c in allc)}")
    a = [win[(s, 'plain')]['fresh'] if win[(s, 'plain')] else 0.0 for s in SOURCES]
    b = [win[(s, 'gate')]['fresh'] if win[(s, 'gate')] else 0.0 for s in SOURCES]
    try:
        print(f"winner speed plain -> gate: {mean(a):.2f} -> {mean(b):.2f}x, gate faster on {sum(y > x for x, y in zip(a, b))}/{len(a)}, "
              f"Wilcoxon p={wilcoxon(a, b).pvalue:.3f}")
    except ValueError:
        print("winner speed: no differences")
    ka = [bool(win[(s, 'plain')] and win[(s, 'plain')]['fallback'] is False) for s in SOURCES]
    kb = [bool(win[(s, 'gate')] and win[(s, 'gate')]['fallback'] is False) for s in SOURCES]
    gain, loss = sum(y and not x for x, y in zip(ka, kb)), sum(x and not y for x, y in zip(ka, kb))
    print(f"winner kept the idea plain -> gate: {sum(ka)} -> {sum(kb)} (gate gains {gain}, loses {loss}, "
          f"sign p={binomtest(gain, gain + loss).pvalue if gain + loss else 1.0:.3f})")

    out = root / "gate_select.json"
    out.write_text(json.dumps([{k_: (str(v) if k_ == "dir" else v) for k_, v in c.items()} for c in cands], indent=1),
                   encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
