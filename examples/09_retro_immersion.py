"""Retroactive check: does immersion in the `immerse` text predict the real speedup?

Diagnosis-kit step 0, no API calls. For every archived `invent_*` run, score the
native's answer (`ideas.md`) with `crazyai.diagnose.immersion_metrics` against its
world (`world.md`), then correlate each metric with the measured speedup.

Speedups are not comparable across targets (wht reaches 180x, nim never passes
0.1x), so the outcome is the run's percentile rank *within its own target*, and
the p-value comes from a permutation test that only shuffles within a target.

Also scored: `survival`, the share of the native's world-specific words that
reappear in the engineer's write-up (`artifact.md`) - a direct measure of how
much of the in-world idea the bend step keeps.

    python examples/09_retro_immersion.py [archive_dir]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from crazyai.diagnose import TECH_TERMS, _content, immersion_metrics, words

N_PERM = 10000


def survival(ideas: str, world: str, artifact: str) -> float:
    world_vocab = set(_content(words(world)))
    carried = {w for w in _content(words(ideas)) if w in world_vocab and w not in TECH_TERMS}
    if not carried:
        return 0.0
    art = set(words(artifact))
    return len(carried & art) / len(carried)


def load(archive: Path) -> list[dict]:
    rows = []
    for d in sorted(archive.glob("invent_*")):
        need = [d / f for f in ("ideas.md", "world.md", "measure.json", "run.json")]
        if not all(p.exists() for p in need):
            continue
        m = json.loads((d / "measure.json").read_text(encoding="utf-8"))
        r = json.loads((d / "run.json").read_text(encoding="utf-8"))
        ideas = (d / "ideas.md").read_text(encoding="utf-8")
        world = (d / "world.md").read_text(encoding="utf-8")
        art = (d / "artifact.md").read_text(encoding="utf-8") if (d / "artifact.md").exists() else ""
        row = {"run": d.name, "target": m["target"], "status": m["status"], "value": m.get("value") or 0.0,
               "depth": r.get("depth"), "blend": r.get("blend_model"),
               "imagination_score": m.get("imagination", {}).get("score")}
        row.update(immersion_metrics(ideas, world))
        row["survival"] = round(survival(ideas, world, art), 4)
        rows.append(row)
    by_t = defaultdict(list)
    for row in rows:
        by_t[row["target"]].append(row)
    for group in by_t.values():
        ranks = rankdata([g["value"] for g in group])
        for g, rk in zip(group, ranks):
            g["pct"] = (rk - 1) / (len(group) - 1) if len(group) > 1 else 0.5
    return rows


def stratified_perm_p(x: np.ndarray, y: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> float:
    obs = abs(spearmanr(x, y).statistic)
    idx = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    hits = 0
    for _ in range(N_PERM):
        yp = y.copy()
        for ix in idx:
            yp[ix] = y[rng.permutation(ix)]
        hits += abs(spearmanr(x, yp).statistic) >= obs
    return (hits + 1) / (N_PERM + 1)


def main() -> None:
    archive = Path(sys.argv[1] if len(sys.argv) > 1 else "archive")
    rows = load(archive)
    y = np.array([r["pct"] for r in rows])
    strata = np.array([r["target"] for r in rows])
    rng = np.random.default_rng(0)
    metrics = ["immersion", "world_rate", "first_person", "tech_rate", "meta_rate", "drift", "survival",
               "words", "imagination_score", "depth"]
    print(f"{len(rows)} invent runs, {len(set(strata))} targets\n")
    print(f"{'metric':<18}{'mean':>9}{'rho':>8}{'perm p':>9}")
    for k in metrics:
        x = np.array([np.nan if r[k] is None else r[k] for r in rows], dtype=float)
        ok = ~np.isnan(x)
        if np.ptp(x[ok]) == 0:
            print(f"{k:<18}{np.nanmean(x):>9.4f}{'constant - no variance':>26}")
            continue
        rho = spearmanr(x[ok], y[ok]).statistic
        p = stratified_perm_p(x[ok], y[ok], strata[ok], rng)
        print(f"{k:<18}{np.nanmean(x):>9.4f}{rho:>8.3f}{p:>9.4f}")
    print("\nper target, rho(immersion, speedup) and rho(survival, speedup):")
    for t in sorted(set(strata)):
        g = [r for r in rows if r["target"] == t]
        if len(g) < 5:
            continue
        v = [r["value"] for r in g]
        im, sv = spearmanr([r["immersion"] for r in g], v), spearmanr([r["survival"] for r in g], v)
        print(f"  {t:<10} n={len(g):<3} immersion {im.statistic:>6.3f} (p={im.pvalue:.3f})"
              f"   survival {sv.statistic:>6.3f} (p={sv.pvalue:.3f})")
    out = archive / "retro_immersion.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"\nper-run scores -> {out}")


if __name__ == "__main__":
    main()
