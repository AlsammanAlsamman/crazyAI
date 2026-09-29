"""Fair re-measurement for `invent-disguise-all` against the original `invent-disguise` pick.

Picking the best of three noisy timings flatters the winner (winner's curse). So the choice is
made from the pipeline's own first measurement, and the comparison uses fresh timings: the
engineer's original kernel and all three solution kernels, re-timed REPEATS times each,
interleaved, on the same machine in the same sitting.

    python examples/10_remeasure_disguise_all.py [archive_dir] [repeats]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import mean

from crazyai.targets import get_target
from crazyai.toolkit.registry import build_toolkit

_PICK = re.compile(r"solution\s*(\d)", re.I)


def engineer_pick(artifact_md: str) -> int | None:
    """Which disguised solution the original single-call engineer said it built (first mention in APPROACH)."""
    approach = artifact_md.split("ARTIFACT")[0]
    m = _PICK.search(approach)
    return int(m.group(1)) if m else None


def main() -> None:
    archive = Path(sys.argv[1] if len(sys.argv) > 1 else "archive")
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    tk = build_toolkit(0)
    rows = []
    for run in sorted(archive.glob("disguise_all_*_*/run.json")):
        d = run.parent
        s = json.loads(run.read_text(encoding="utf-8"))
        tgt = get_target(s["target"])
        orig = archive / f"disguise_{s['seed']}_{s['target']}"
        kernels = {"orig": orig / "artifact.c"} | {f"sol{k}": d / f"sol_{k}" / "artifact.c" for k in (1, 2, 3)}
        kernels = {k: p.read_text(encoding="utf-8") for k, p in kernels.items() if p.exists()}
        times: dict[str, list[float]] = {k: [] for k in kernels}
        status: dict[str, str] = {}
        for _ in range(repeats):
            for k, src in kernels.items():
                r = tk.call(tgt.measure_tool, {"source": src, "budget": 0.3})
                status[k] = r.get("status", "ERROR")
                times[k].append(r.get("value", 0.0) or 0.0)
        row = {"seed": s["seed"], "target": s["target"], "picked_by_benchmark": s["best"],
               "engineer_pick": engineer_pick((orig / "artifact.md").read_text(encoding="utf-8"))
               if (orig / "artifact.md").exists() else None,
               "status": status, "mean": {k: round(mean(v), 3) for k, v in times.items()},
               "runs": {k: [round(x, 3) for x in v] for k, v in times.items()}}
        row["chosen_mean"] = row["mean"].get(f"sol{s['best']}")
        row["orig_mean"] = row["mean"].get("orig")
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("seed", "target", "engineer_pick", "picked_by_benchmark", "status",
                                              "mean")}), flush=True)
    out = archive / "disguise_all_remeasure.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\ntarget      original pick   benchmark pick   (fresh means, exact-only wins)")
    for t in sorted({r["target"] for r in rows}):
        g = [r for r in rows if r["target"] == t]
        o = [r["orig_mean"] if r["status"].get("orig") == "exact" else 0.0 for r in g]
        c = [r["chosen_mean"] if r["status"].get(f"sol{r['picked_by_benchmark']}") == "exact" else 0.0 for r in g]
        print(f"{t:<11} {mean(o):>8.3f}x ({sum(x > 1.05 for x in o)}/{len(g)})   "
              f"{mean(c):>8.3f}x ({sum(x > 1.05 for x in c)}/{len(g)})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
