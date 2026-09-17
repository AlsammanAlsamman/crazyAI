"""crazyai command line.

    crazyai list                       domains, operators, generators, tools
    crazyai tools [--kind invent]      the toolkit
    crazyai seed --seed 42             step 1 only
    crazyai run --seed 42 --generator formula [--provider mock] [--runs 5]
    crazyai batch --n 20 --start 1 --generator all
    crazyai rank [--by discovery_value] [--top 20]
    crazyai report [--out report.md] [--svg profile.svg]
    crazyai compare archive/run_1_formula archive/run_2_formula

    crazyai blend --seed 42 [--model compare]          blend the imagination corpus, offline
    crazyai invent --seed 42 --target matmul [--blend anneal] [--harvest 5] [--provider mock]
    crazyai invent-rank [--target matmul] [--top 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from crazyai.config import ARCHIVE_DIR, DEFAULT_EFFORT, DEFAULT_MODEL, GENERATORS, OPERATORS


def _provider(args):
    from crazyai.providers import get_provider

    if args.provider == "mock":
        return get_provider("mock", detect=getattr(args, "mock_detect", False))
    return get_provider("anthropic", model=args.model, effort=args.effort, fallbacks=not args.no_fallbacks)


def cmd_list(args) -> int:
    from crazyai.domains import load_domains
    from crazyai.generators import list_generators
    from crazyai.toolkit.registry import build_toolkit

    print("DOMAINS")
    for d in load_domains().values():
        print(f"  {d.name:14s} {len(d.rules()):3d} rules  - {d.description}")
    print("\nOPERATORS\n  " + ", ".join(OPERATORS))
    print("\nGENERATORS")
    for g in list_generators():
        print(f"  {g.name:10s} {g.title}")
    tk = build_toolkit(0)
    print(f"\nTOOLS: {len(tk.names(kind='invent'))} invent, {len(tk.names(kind='measure'))} measure  (crazyai tools)")
    return 0


def cmd_tools(args) -> int:
    from crazyai.toolkit.registry import build_toolkit

    tk = build_toolkit(0)
    for name in tk.names(kind=args.kind or None):
        spec = tk.all_specs()[name]
        req = ", ".join(spec.input_schema.get("required", []))
        print(f"[{spec.kind:7s}] {name}({req})\n           {spec.description}")
    return 0


def cmd_seed(args) -> int:
    from crazyai.toolkit.registry import build_toolkit

    tk = build_toolkit(args.seed)
    s = tk.call("chaos_draw_seed", {"domain": args.domain or ""})
    op = tk.call("chaos_draw_operator")
    print(json.dumps({"seed": args.seed, "draw": s, "operator": op["operator"], "rng_log": tk.rng.log}, indent=2))
    return 0


def cmd_run(args) -> int:
    from crazyai.pipeline.run import Run

    provider = _provider(args)
    gens = GENERATORS if args.generator == "all" else [args.generator]
    for g in gens:
        run = Run(seed=args.seed, generator=g, domain=args.domain or "", runs=args.runs,
                  examiner_tools=args.examiner_tools, with_formal=not args.no_formal, force=args.force,
                  archive_dir=Path(args.archive))
        summary = run.execute(provider)
        print(json.dumps(summary["metrics"], indent=2))
    return 0


def cmd_batch(args) -> int:
    from crazyai.pipeline.run import Run

    provider = _provider(args)
    gens = GENERATORS if args.generator == "all" else [args.generator]
    for i in range(args.n):
        seed = args.start + i
        for g in gens:
            Run(seed=seed, generator=g, runs=args.runs, force=args.force, archive_dir=Path(args.archive)).execute(provider)
    return 0


def cmd_rank(args) -> int:
    from crazyai.pipeline.report import load_index, rank

    rows = rank(load_index(args.archive), args.by, args.top)
    for r in rows:
        m = r["metrics"]
        print(f"seed={r['seed']:<6} {r['generator']:9s} {r['rule_id']:24s} {r['operator']:11s} "
              f"discovery={m['discovery_value']:.3f} detect={m['detection_rate']:.2f} accept={m['acceptance_rate']:.2f}")
    if not rows:
        print("archive is empty")
    return 0


def cmd_report(args) -> int:
    from crazyai.pipeline.report import METRICS, aggregate, load_index, markdown_report, radar_svg

    rows = load_index(args.archive)
    md = markdown_report(rows)
    if args.out:
        Path(args.out).write_text(md)
        print(f"wrote {args.out}")
    else:
        print(md)
    if args.svg and rows:
        agg = aggregate(rows, "generator")
        overall = {m: sum(a[m]["mean"] * a["n"] for a in agg.values()) / len(rows) for m in METRICS}
        Path(args.svg).write_text(radar_svg(overall, title="crazyAI profile"))
        print(f"wrote {args.svg}")
    return 0


def cmd_blend(args) -> int:
    from crazyai.toolkit.invent.blend import MODELS
    from crazyai.toolkit.registry import build_toolkit

    tk = build_toolkit(args.seed)
    if args.model == "compare":
        r = tk.call("blend_compare", {})
        for row in r["ranking"]:
            print(f"{row['model']:8s} score={row['score']:.3f}  imagination={row['imagination']:.2f} readable={row['readable']:.2f}  "
                  f"surprise={row['surprise']:.2f} mixing={row['mixing']:.2f} originality={row['originality']:.2f} coherence={row['coherence']:.2f}")
        print(f"\n--- best: {r['best']} ---\n{r['texts'][r['best']]}")
    else:
        r = tk.call(f"blend_{args.model}", {})
        print(json.dumps(r["score"], indent=2))
        print("\n" + r["text"])
        print("\nbuilt from: " + ", ".join(f"{f['kind']}:{f['source']}" for f in r["fragments"]))
    return 0


def cmd_invent(args) -> int:
    from crazyai.pipeline.invent import Invent

    provider = _provider(args)
    for i in range(args.n):
        run = Invent(seed=args.seed + i, target=args.target, blend=args.blend, harvest=args.harvest,
                     force=args.force, archive_dir=Path(args.archive))
        summary = run.execute(provider)
        print(json.dumps({k: summary[k] for k in ("seed", "target", "blend_model", "imagination_score", "status",
                                                   "value", "prediction", "calibration", "discovery")}, indent=2))
    return 0


def cmd_invent_rank(args) -> int:
    from crazyai.pipeline.invent import load_invent_index

    rows = [r for r in load_invent_index(args.archive) if not args.target or r["target"] == args.target]
    rows.sort(key=lambda r: -(r.get("discovery") or 0))
    for r in rows[: args.top]:
        print(f"seed={r['seed']:<6} {r['target']:9s} blend={r['blend_model']:8s} imagination={r['imagination_score']:.2f} "
              f"status={r['status']:9s} value={r['value'] if r['value'] is not None else '-':<8} "
              f"pred={r['prediction'] if r['prediction'] is not None else '-':<6} calib={r['calibration'] if r['calibration'] is not None else '-':<6} "
              f"discovery={r['discovery']:.3f}  focus='{r['assumption_focus']}'")
    if not rows:
        print("no invent runs archived")
    return 0


def cmd_compare(args) -> int:
    from crazyai.pipeline.report import compare

    print(json.dumps(compare(Path(args.a), Path(args.b)), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crazyai", description="The impossible, disguised as possible and true.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_provider(sp):
        sp.add_argument("--provider", choices=["anthropic", "mock"], default="anthropic")
        sp.add_argument("--model", default=DEFAULT_MODEL)
        sp.add_argument("--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high", "xhigh", "max"])
        sp.add_argument("--no-fallbacks", action="store_true", help="disable server-side refusal fallbacks")
        sp.add_argument("--mock-detect", action="store_true", help="mock provider: the examiner finds the flaw")
        sp.add_argument("--archive", default=str(ARCHIVE_DIR))
        sp.add_argument("--runs", type=int, default=5, help="cross-examination repetitions")
        sp.add_argument("--force", action="store_true", help="redo steps even if files exist")

    sub.add_parser("list").set_defaults(fn=cmd_list)
    t = sub.add_parser("tools")
    t.add_argument("--kind", choices=["invent", "measure"])
    t.set_defaults(fn=cmd_tools)
    s = sub.add_parser("seed")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--domain", default="")
    s.set_defaults(fn=cmd_seed)
    r = sub.add_parser("run")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--generator", default="formula", choices=GENERATORS + ["all"])
    r.add_argument("--domain", default="")
    r.add_argument("--examiner-tools", action="store_true", help="give the cross-examiner the measure tools")
    r.add_argument("--no-formal", action="store_true", help="show the examiner only the artifact, not the formalisation")
    add_provider(r)
    r.set_defaults(fn=cmd_run)
    b = sub.add_parser("batch")
    b.add_argument("--n", type=int, default=10)
    b.add_argument("--start", type=int, default=1)
    b.add_argument("--generator", default="all", choices=GENERATORS + ["all"])
    add_provider(b)
    b.set_defaults(fn=cmd_batch)
    k = sub.add_parser("rank")
    k.add_argument("--by", default="discovery_value")
    k.add_argument("--top", type=int, default=20)
    k.add_argument("--archive", default=str(ARCHIVE_DIR))
    k.set_defaults(fn=cmd_rank)
    rp = sub.add_parser("report")
    rp.add_argument("--out", default="")
    rp.add_argument("--svg", default="")
    rp.add_argument("--archive", default=str(ARCHIVE_DIR))
    rp.set_defaults(fn=cmd_report)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(fn=cmd_compare)

    from crazyai.targets import TARGETS
    from crazyai.toolkit.invent.blend import MODELS
    bl = sub.add_parser("blend", help="blend the imagination corpus with one model, or compare all")
    bl.add_argument("--seed", type=int, required=True)
    bl.add_argument("--model", default="compare", choices=MODELS + ["compare"])
    bl.set_defaults(fn=cmd_blend)
    iv = sub.add_parser("invent", help="corpus -> blend -> immerse -> bend -> measure")
    iv.add_argument("--seed", type=int, required=True)
    iv.add_argument("--n", type=int, default=1, help="number of consecutive seeds")
    iv.add_argument("--target", default="matmul", choices=TARGETS)
    iv.add_argument("--blend", default="", choices=MODELS + ["compare", ""], help="blend model (default: seeded draw)")
    iv.add_argument("--harvest", type=int, default=0, help="fragments the AI adds to the corpus first")
    add_provider(iv)
    iv.set_defaults(fn=cmd_invent)
    ir = sub.add_parser("invent-rank")
    ir.add_argument("--target", default="")
    ir.add_argument("--top", type=int, default=20)
    ir.add_argument("--archive", default=str(ARCHIVE_DIR))
    ir.set_defaults(fn=cmd_invent_rank)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
