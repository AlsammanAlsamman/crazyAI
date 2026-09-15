"""Build the README figures from the real example data.

    .venv/bin/python assets/figures/make_figures.py      # -> assets/figures/*.png (+ .svg)
    make figures

Every number drawn here is computed by the toolkit on the spot; nothing is typed in.
The use-cases figure is the one exception: its radar/bars are labelled illustrative.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svgkit import C, MONO, PALETTE, SVG, rasterise  # noqa: E402

from crazyai.domains import rule_index  # noqa: E402
from crazyai.toolkit import native  # noqa: E402
from crazyai.toolkit.registry import build_toolkit  # noqa: E402
from crazyai.worlds.partial_primes import features, label_world  # noqa: E402

OUT = Path(__file__).resolve().parent
FLOW_COLORS = {"invent": C["invent"], "measure": C["measure"], "mixed": C["mixed"]}


def kinds(*ks):
    return lambda i: FLOW_COLORS[ks[i]]


# ------------------------------------------------------------------ figure 1: partial primes
def fig_primes() -> None:
    W, H = 1500, 860
    s = SVG(W, H, "Example 2 · a world where primes are not quite prime",
            "seed nt.prime.def → SUBSTITUTE → three partial conditions → 20 000 integers labelled in C++ → a real predictor → what has to give")
    s.step_flow(60, 100, W - 120, [
        ("1 seed", "chaos_draw_seed"), ("2 mutate", "SUBSTITUTE · nt.prime.def"), ("3 predicates", "symbolic_define_predicate"),
        ("4 label world", "native sieve + even+prime"), ("5 fit predictor", "stats_fit_table · logit"),
        ("6 limit", "symbolic_take_limit"), ("7 ground", "ground_cost_of_possibility")],
        color_of=kinds("invent", "invent", "measure", "measure", "measure", "measure", "measure"))

    tk = build_toolkit(11)
    limit = 20000
    w = label_world(limit, 5000)
    n = w["n"]
    score = w["partial_primality"]
    prime = w["is_prime"]
    xs = np.unique(np.geomspace(20, limit, 60).astype(int))
    cum3 = [float((score[2:x + 1] == 3).mean()) for x in xs]
    cum2 = [float((score[2:x + 1] >= 2).mean()) for x in xs]
    cump = [float(prime[2:x + 1].mean()) for x in xs]
    theo = [1 / math.log(x) for x in xs]

    # left: densities
    s.card(60, 200, 860, 600, "Density of 'primes' as n grows", C["measure"],
           "cumulative fraction of integers ≤ n satisfying each definition · log-x")
    ax = s.axes(140, 270, 740, 440, (20, limit), (0, 1.0), xlabel="n", ylabel="fraction of integers ≤ n",
                xticks=[20, 100, 1000, 10000, 20000], yticks=[0, 0.25, 0.5, 0.75, 1.0],
                xfmt=lambda v: f"{v:,}", yfmt=lambda v: f"{v:.2f}", logx=True)
    ax.area(list(xs), [0] * len(xs), cum2, PALETTE[2], opacity=0.12)
    ax.polyline(list(xs), cum2, PALETTE[2], sw=2.5)
    ax.polyline(list(xs), cum3, PALETTE[1], sw=3)
    ax.polyline(list(xs), cump, PALETTE[0], sw=3)
    ax.polyline(list(xs), theo, PALETTE[0], sw=1.5, dash="5 5", opacity=0.7)
    lx, ly = 640, 600
    for i, (lab, col, dash) in enumerate([("partial score ≥ 2 (any two conditions)", PALETTE[2], None),
                                          ("fully partial-prime (all three)", PALETTE[1], None),
                                          ("ordinary primes, π(n)/n", PALETTE[0], None),
                                          ("1 / ln n  (prime number theorem)", PALETTE[0], "5 5")]):
        s.line(lx, ly + i * 20, lx + 28, ly + i * 20, stroke=col, sw=3, dash=dash)
        s.text(lx + 36, ly + i * 20 + 4, lab, size=11.5, fill=C["muted"])
    lim = tk.call("symbolic_take_limit", {"expression": "(x/log(x))/x", "variable": "x", "target": "oo"})["limit"]
    s.chip(150, 740, f"symbolic_take_limit((x/log x)/x, x→oo) = {lim}   ·   ordinary primes thin out; the mutated ones do not", C["measure"])

    # right top: predictor
    X = features(n[2:])
    y3 = (score[2:] == 3).astype(float)
    cols = {**{k: v.tolist() for k, v in X.items()}, "full": y3.tolist(), "prime": prime[2:].astype(float).tolist()}
    regs = ["mod2", "mod4_is1", "digit_sum", "log_n"]
    f1 = tk.call("stats_fit_table", {"columns": cols, "outcome": "full", "regressors": regs, "model": "logit"})
    f2 = tk.call("stats_fit_table", {"columns": cols, "outcome": "prime", "regressors": regs, "model": "logit"})
    s.card(950, 200, 490, 250, "A real predictor, fitted on the fake world", C["measure"],
           "logit on residues, digit sum, log n · stats_fit_table")
    rows = [("target", "accuracy", "base rate"), ("fully partial-prime", f"{f1['accuracy']:.3f}", f"{f1['base_rate']:.3f}"),
            ("ordinary prime", f"{f2['accuracy']:.3f}", f"{f2['base_rate']:.3f}")]
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            s.text(975 + c * 150, 290 + r * 26, cell, size=12.5 if r else 11, fill=C["text"] if r else C["muted"],
                   weight=700 if r == 0 or c == 0 else 400, mono=r > 0 and c > 0)
    hist = np.bincount(score[2:], minlength=4)
    s.text(975, 385, "partial-primality histogram (score 0…3):", size=11.5, fill=C["muted"])
    bx = 975
    for k in range(4):
        wdt = 18 + 380 * hist[k] / hist.sum()
        s.rect(bx, 398, wdt, 16, fill=PALETTE[k % 4], rx=4, opacity=0.85)
        s.text(bx + wdt / 2, 411, f"{k}: {hist[k]:,}", size=10, fill=C["bg0"], anchor="middle", weight=700)
        bx += wdt + 4
    s.text(975, 436, "the same features predict ordinary primes about as well: the fake world is learnable, which is the disguise",
           size=10.5, fill=C["muted"], opacity=0.9)

    # right bottom: ground
    g = tk.call("ground_what_must_be_true", {"rule_id": "nt.prime.def", "operator": "SUBSTITUTE"})
    cost = tk.call("ground_cost_of_possibility", {"rule_ids": ["nt.prime.def"]})
    s.card(950, 470, 490, 330, "What has to give", C["invent"],
           f"ground_cost_of_possibility = {cost['total']} ({cost['band']}) · {cost['reading']}")
    items = [(c["rule_id"], c["weight"], 0) for c in g["minimal_changes"]] + \
            [(c["rule_id"], c["weight"], c["distance"]) for c in g["collateral"]]
    idx = rule_index()
    for i, (rid, wt, dist) in enumerate(items):
        yy = 545 + i * 40
        col = C["invent"] if dist == 0 else PALETTE[2] if dist == 1 else PALETTE[0]
        s.rect(975, yy - 12, 20 * wt, 18, fill=col, rx=4, opacity=0.9 - 0.12 * dist)
        s.text(975 + 20 * wt + 8, yy + 2, f"{rid}  w={wt}{' · mutated' if dist == 0 else f' · d={dist}'}", size=10.5, mono=True,
               fill=C["text"] if dist == 0 else C["muted"])
        s.text(975, yy + 17, idx[rid].statement[:78] + ("…" if len(idx[rid].statement) > 78 else ""), size=9.5, fill=C["muted"], opacity=0.8)
    s.text(W / 2, H - 22, "everything above is computed: sieve and Goldbach-style counts in cpp/kernels.cpp, the fit in stats_fit_table, the limit in SymPy",
           size=12, fill=C["line"], anchor="middle")
    print("primes:", rasterise(s.render(), OUT / "example_primes.png", W, H))


# ------------------------------------------------------------------ figure 2: the story
def fig_story() -> None:
    sys.path.insert(0, str(OUT.parent.parent / "examples"))
    import importlib
    ex = importlib.import_module("03_watermelon_story")
    tk = build_toolkit(3)
    tl = tk.call("narrative_check_timeline", {"events": ex.EVENTS})
    lg = tk.call("logic_check_consistency", {"formulas": ex.WORLD_RULES})
    ws = tk.call("narrative_word_stats", {"text": ex.STORY})

    W, H = 1500, 800
    s = SVG(W, H, "Example 3 · a watermelon investigates whether oranges can marry grapefruit",
            "seed nar.kin.persons → TRANSPOSE kinship law into fruit → write → measure: timeline, knowledge, world-rules, word statistics")
    s.step_flow(60, 100, W - 120, [
        ("1 seed", "nar.kin.persons"), ("2 mutate", "TRANSPOSE → fruit"), ("3 cast & facts", "narrative_knowledge_graph"),
        ("4 events", "narrative_build_timeline"), ("5 write", "600–1200 words"),
        ("6 measure", "check_timeline · word_stats"), ("7 disguise", "rephrase_to_corpus · tone")],
        color_of=kinds("invent", "invent", "measure", "measure", "mixed", "measure", "invent"))

    # timeline with character lanes
    s.card(60, 200, 900, 580, "Who knows what, when — and the two flaws the tools found", C["measure"],
           f"narrative_check_timeline: flaw_count = {tl['flaw_count']}")
    chars = ["watermelon", "clerk", "oranges", "lemon", "grapefruit"]
    x0, x1, y0 = 200, 920, 310
    lane = 66
    ev = sorted(ex.EVENTS, key=lambda e: e["t"])
    ex_x = {e["id"]: x0 + (i + 0.5) * (x1 - x0) / len(ev) for i, e in enumerate(ev)}
    for i, ch in enumerate(chars):
        yy = y0 + i * lane
        s.line(x0, yy, x1, yy, stroke=C["edge"], sw=1)
        s.text(x0 - 12, yy + 4, ch, size=12, anchor="end", weight=700)
    for e in ev:
        ex_ = ex_x[e["id"]]
        s.line(ex_, y0 - 30, ex_, y0 + (len(chars) - 1) * lane + 20, stroke="#fff", sw=0.6, opacity=0.08)
        s.text(ex_, y0 - 36, f"t={e['t']}  {e['id']}", size=10.5, fill=C["muted"], anchor="middle", mono=True)
        seen: dict[str, int] = {}
        for rev in e.get("reveals", []) or []:
            for who in rev["to"]:
                yy = y0 + chars.index(who) * lane
                k = seen.get(who, 0); seen[who] = k + 1
                s.circle(ex_, yy, 7, C["measure"])
                s.text(ex_, yy - 12 - 11 * k, rev["fact"][:28], size=8.5, fill=C["muted"], anchor="middle")
        seen = {}
        for req in e.get("requires", []) or []:
            yy = y0 + chars.index(req["by"]) * lane
            k = seen.get(req["by"], 0); seen[req["by"]] = k + 1
            known = any(req["fact"] in [r["fact"] for r in (f.get("reveals") or []) if req["by"] in r["to"]]
                        for f in ev if f["t"] < e["t"])
            col = C["green"] if known else C["red"]
            s.circle(ex_, yy, 10, "none", stroke=col, sw=2.5)
            s.text(ex_, yy + 24 + 11 * k, ("acts on: " + req["fact"])[:34], size=8.5, fill=col, anchor="middle")
    # flaws
    for f in tl["flaws"]:
        if f["kind"] == "backwards_causation":
            xa, xb = ex_x[f["cause"]], ex_x[f["effect"]]
            ytop = y0 + (len(chars) - 1) * lane + 30
            s.path(f"M{xa},{ytop} C{xa},{ytop+50} {xb},{ytop+50} {xb},{ytop+6}", stroke=C["red"], sw=2.5, dash="6 5", arrow="red")
            s.text((xa + xb) / 2, ytop + 62, f"FLAW 1 · '{f['cause']}' is said to cause '{f['effect']}' — effect precedes cause",
                   size=11, fill=C["red"], anchor="middle", weight=700)
        else:
            xe = ex_x[f["event"]]
            yy = y0 + chars.index(f["character"]) * lane
            s.text(xe + 12, yy + 44, "FLAW 2 · acts on a fact nobody told him", size=11, fill=C["red"], anchor="end", weight=700)
    s.text(90, 765, "● fact revealed to the character     ○ character acts on a fact   (green: known, red: never witnessed or told)", size=10.5, fill=C["muted"])

    # world rules
    s.card(990, 200, 450, 290, "World-rules as propositions", C["measure"],
           f"logic_check_consistency → consistent = {lg['consistent']}")
    bad = set(lg.get("minimal_inconsistent_subset", []))
    labels = {"placed -> ~moves": "things are placed; nothing moves", "changed -> maker_known": "a placement is changed by its maker",
              "~maker_known": "nobody remembers the maker", "amended -> changed": "amending the register changes a placement",
              "amended": "…and the register is amended"}
    for i, f in enumerate(ex.WORLD_RULES):
        yy = 285 + i * 36
        col = C["red"] if f in bad else C["muted"]
        s.rect(1010, yy - 14, 410, 30, fill=C["bg0"], stroke=col, rx=8, opacity=0.9, sw=1.5 if f in bad else 0.8)
        s.text(1020, yy + 5, f, size=12, mono=True, fill=C["text"] if f in bad else C["muted"], weight=700 if f in bad else 400)
        s.text(1415, yy + 5, labels[f], size=9.5, fill=col, anchor="end")
    s.text(1010, 470, "red = minimal inconsistent subset: the one contradiction the story was allowed to keep", size=10.5, fill=C["red"])

    # word stats
    s.card(990, 510, 450, 270, "Word statistics vs reference prose", C["invent"], "narrative_word_stats · disguise_rephrase_to_corpus targets")
    metrics = [("mean_sentence_len", 25), ("sd_sentence_len", 12), ("type_token_ratio", 1), ("hedge_ratio", 0.02),
               ("dialogue_share", 0.15), ("flesch_reading_ease", 100)]
    for i, (m, mx) in enumerate(metrics):
        yy = 572 + i * 31
        a, b = ws["text"][m], ws["reference"][m]
        s.text(1010, yy + 4, m, size=10.5, mono=True, fill=C["muted"])
        s.rect(1170, yy - 12, 170 * min(1, a / mx), 9, fill=C["invent"], rx=3)
        s.rect(1170, yy + 1, 170 * min(1, b / mx), 9, fill=C["measure"], rx=3)
        s.text(1415, yy + 7, f"{a:.3g} / {b:.3g}", size=10, mono=True, anchor="end", fill=C["text"])
    s.rect(1010, 762, 10, 10, fill=C["invent"]); s.text(1026, 771, "story", size=10.5, fill=C["muted"])
    s.rect(1080, 762, 10, 10, fill=C["measure"]); s.text(1096, 771, "reference", size=10.5, fill=C["muted"])
    s.text(1200, 771, "→ revise until the story is statistically ordinary", size=10.5, fill=C["muted"])
    print("story:", rasterise(s.render(), OUT / "example_story.png", W, H))


# ------------------------------------------------------------------ figure 3: collatz → music
def fig_collatz() -> None:
    tk = build_toolkit(5)
    N = 64
    ln, pk = native.collatz(N)
    structure = {"nodes": [{"id": str(k), "x": k, "y": int(ln[k]), "weight": round(math.log(int(pk[k])) + 0.5, 3),
                            "group": int(ln[k]) % 4} for k in range(1, N + 1)]}
    img = tk.call("transform_structure_to_image_description", {"structure": structure})
    score = tk.call("transform_image_description_to_music", {"image": img})
    rev = tk.call("transform_reverse", {"score": score})
    back_rev = tk.call("transform_image_description_to_structure", {"image": tk.call("transform_music_to_image_description", {"score": rev})})
    back_fwd = tk.call("transform_image_description_to_structure", {"image": tk.call("transform_music_to_image_description", {"score": score})})
    cmp_rev = tk.call("symbolic_compare_structures", {"a": structure, "b": back_rev, "tolerance": 1e-3})
    cmp_fwd = tk.call("symbolic_compare_structures", {"a": structure, "b": back_fwd, "tolerance": 1e-3})

    W, H = 1700, 780
    s = SVG(W, H, "Example 4 · an unsolved problem, as an image, as music, read backwards",
            "Collatz orbits of 1…64 → structure → image description → score → reversed → back → compare_structures says what survived")
    s.step_flow(60, 100, W - 120, [
        ("1 orbits", "symbolic_collatz_orbits (C++)"), ("2 → image", "structure_to_image_description"),
        ("3 → music", "image_description_to_music"), ("4 reverse", "transform_reverse"),
        ("5 → back", "music → image → structure"), ("6 compare", "symbolic_compare_structures")],
        color_of=kinds("measure", "invent", "invent", "invent", "invent", "measure"))

    pw, ph, top = 320, 380, 200
    gx = 60
    # panel 1: structure
    s.card(gx, top, pw + 40, ph + 120, "structure", C["measure"], "x = n · y = stopping time · size = log peak · colour = group")
    ax = s.axes(gx + 60, top + 70, pw - 40, ph - 20, (0, N + 1), (0, int(ln[1:].max()) + 8), xlabel="n", ylabel="steps to 1",
                xticks=[1, 16, 32, 48, 64], yticks=[0, 40, 80, 120])
    nodes = structure["nodes"]
    ax.scatter([d["x"] for d in nodes], [d["y"] for d in nodes], [PALETTE[d["group"]] for d in nodes], [1.2 + d["weight"] * 0.55 for d in nodes])
    s.line(gx + pw + 44, top + 220, gx + pw + 78, top + 220, sw=3, arrow="line")

    def piano(x, title, sc, sub):
        s.card(x, top, pw + 40, ph + 120, title, C["invent"], sub)
        axp = s.axes(x + 60, top + 70, pw - 40, ph - 20, (0, len(sc["notes"])), (34, 98), xlabel="time (note index)", ylabel="MIDI pitch",
                     xticks=[0, 16, 32, 48, 64], yticks=[36, 60, 84, 96])
        for i, nt in enumerate(sc["notes"]):
            x0, x1 = axp.px(i), axp.px(i + min(1.0, nt["duration"] / 10) + 0.15)
            yy = axp.py(nt["pitch"])
            s.rect(x0, yy - 3, x1 - x0, 6, fill=PALETTE[nt["instrument"] % 8], rx=2, opacity=0.5 + 0.5 * (nt["velocity"] - 40) / 87)
        return axp

    piano(gx + pw + 84, "score", score, "pitch ← y · duration ← weight · velocity ← x")
    s.line(gx + 2 * pw + 128, top + 220, gx + 2 * pw + 162, top + 220, sw=3, arrow="line")
    piano(gx + 2 * pw + 168, "score, reversed", rev, "same notes, last first")
    s.line(gx + 3 * pw + 212, top + 220, gx + 3 * pw + 246, top + 220, sw=3, arrow="line")

    # panel 4: what survived
    x4 = gx + 3 * pw + 252
    s.card(x4, top, W - x4 - 60, ph + 120, "what survived the round trip", C["measure"], "symbolic_compare_structures")
    axc = s.axes(x4 + 60, top + 135, W - x4 - 60 - 90, ph - 85, (0, N + 1), (0, N + 1), xlabel="original n", ylabel="recovered x",
                 xticks=[1, 32, 64], yticks=[1, 32, 64])
    axc.polyline([d["x"] for d in structure["nodes"]], [d["x"] for d in back_fwd["nodes"]], C["green"], sw=3)
    ra = {d["id"]: d for d in back_rev["nodes"]}
    axc.polyline([d["x"] for d in nodes], [ra[d["id"]]["x"] for d in nodes], C["red"], sw=3, dash="6 5")
    s.text(x4 + 16, top + 72, f"forward:  identical = {cmp_fwd['identical']}", size=11, fill=C["green"], weight=700)
    s.text(x4 + 16, top + 90, f"reversed: identical = {cmp_rev['identical']}, order reversed = {cmp_rev['order_reversed']}", size=11, fill=C["red"], weight=700)
    s.text(x4 + 16, top + 108, "x ↦ N + 1 − x : an encoding artefact, not an invariant", size=10.5, fill=C["muted"])
    s.text(W / 2, H - 22, "the round trip is a real, reproducible experiment; the crazy claim built on top of it is what the fresh session gets to judge",
           size=12, fill=C["line"], anchor="middle")
    print("collatz:", rasterise(s.render(), OUT / "example_collatz.png", W, H))


# ------------------------------------------------------------------ figure 4: how it is used
def fig_usecases() -> None:
    W, H = 1500, 760
    s = SVG(W, H, "How crazyAI is used", "the same archive answers four different questions")
    cols = [
        ("Profile a model", "credulity radar", C["measure"],
         ["run 100 seeds × 6 generators against a model", "detection · acceptance · hedge · false-flaw", "where does fluent form beat substance?"],
         "crazyai batch --n 100 --generator all"),
        ("Compare models / personas", "same seeds, different targets", C["invent"],
         ["identical artifacts shown to two targets", "or one target with two system prompts", "delta per metric, per generator"],
         "crazyai compare run_7_debate run_7_debate-b"),
        ("Surface candidates", "discovery value ranking", C["mixed"],
         ["rigor × novelty × cost-of-possibility", "high rigor, high novelty, low cost →", "flagged for a human to look at"],
         "crazyai rank --by discovery_value --top 20"),
        ("Teach & test", "confident-wrong questions", C["claude"],
         ["questions with a false premise built in", "answer sheet with both answers", "exercises for people and for models"],
         "crazyai run --generator questions --seed 3"),
    ]
    cw, gap, top, ch = 330, 26, 110, 520
    for i, (title, sub, col, lines, cmd) in enumerate(cols):
        x = 60 + i * (cw + gap)
        s.card(x, top, cw, ch, title, col, sub)
        for j, l in enumerate(lines):
            s.text(x + 16, top + 80 + j * 20, l, size=12, fill=C["muted"])
        s.chip(x + 16, top + ch - 22, cmd, col)
        # mini visual
        vy = top + 160
        if i == 0:
            keys = ["detection", "acceptance", "hedge", "false-flaw", "conf-wrong", "depth"]
            vals = [0.55, 0.35, 0.10, 0.20, 0.72, 0.48]
            cx, cy, r = x + cw / 2, vy + 140, 105
            for f in (0.33, 0.66, 1.0):
                pts = " ".join(f"{cx + r*f*math.cos(-math.pi/2 + 2*math.pi*k/6):.1f},{cy + r*f*math.sin(-math.pi/2 + 2*math.pi*k/6):.1f}" for k in range(6))
                s.parts.append(f'<polygon points="{pts}" fill="none" stroke="{C["edge"]}" stroke-width="0.8"/>')
            pts = " ".join(f"{cx + r*v*math.cos(-math.pi/2 + 2*math.pi*k/6):.1f},{cy + r*v*math.sin(-math.pi/2 + 2*math.pi*k/6):.1f}" for k, v in enumerate(vals))
            s.parts.append(f'<polygon points="{pts}" fill="{col}" fill-opacity="0.3" stroke="{col}" stroke-width="2"/>')
            for k, key in enumerate(keys):
                s.text(cx + (r + 22) * math.cos(-math.pi/2 + 2*math.pi*k/6), cy + (r + 22) * math.sin(-math.pi/2 + 2*math.pi*k/6) + 4,
                       key, size=9.5, fill=C["muted"], anchor="middle")
            s.text(cx, vy + 285, "illustrative shape, not a measurement", size=9.5, fill=C["muted"], anchor="middle", opacity=0.8)
        elif i == 1:
            gens = ["story", "formula", "plan", "stat", "quest", "debate"]
            a = [0.7, 0.3, 0.5, 0.4, 0.6, 0.8]
            b = [0.4, 0.2, 0.3, 0.3, 0.5, 0.5]
            ax = s.axes(x + 50, vy, cw - 80, 230, (0, 6), (0, 1), yticks=[0, 0.5, 1], yfmt=lambda v: f"{v:.1f}")
            for k, g in enumerate(gens):
                ax.bars([k + 0.32], [a[k]], 0.3, PALETTE[0])
                ax.bars([k + 0.68], [b[k]], 0.3, PALETTE[1])
                s.text(ax.px(k + 0.5), vy + 246, g, size=9.5, fill=C["muted"], anchor="middle")
            s.text(x + 60, vy - 6, "acceptance rate: base ■ vs 'sceptical reviewer' persona ■", size=9.5, fill=C["muted"])
            s.text(x + cw / 2, vy + 285, "illustrative values, not a measurement", size=9.5, fill=C["muted"], anchor="middle", opacity=0.8)
        elif i == 2:
            rows = [("seed 41 · formula · REMOVE", 0.61), ("seed 17 · debate · QUANTIFY", 0.48), ("seed 88 · plan · TRANSPOSE", 0.36),
                    ("seed 5 · statmodel · INVERT", 0.30), ("seed 23 · story · COMPOSE", 0.21)]
            for k, (lab, v) in enumerate(rows):
                yy = vy + 20 + k * 44
                s.text(x + 16, yy, lab, size=10.5, mono=True, fill=C["text"] if k == 0 else C["muted"])
                s.rect(x + 16, yy + 8, (cw - 60) * v, 10, fill=col, rx=3, opacity=1 - 0.15 * k)
                s.text(x + cw - 16, yy + 17, f"{v:.2f}", size=10, mono=True, anchor="end", fill=C["muted"])
            s.text(x + 16, vy + 250, "discovery = rigor × novelty × cost × minimality", size=10.5, fill=col, weight=700)
            s.text(x + cw / 2, vy + 285, "illustrative ranking", size=9.5, fill=C["muted"], anchor="middle", opacity=0.8)
        else:
            qs = ["Given that every even number above 2 is", "  a sum of two primes, and that 2 is the", "  only even prime, how many odd primes", "  does 100 need?",
                  "", "invited answer:  2", "actual answer:   2 (the premise is true here)", "", "…and four more where the premise is false,", "each with both answers computed."]
            for k, l in enumerate(qs):
                s.text(x + 16, vy + 10 + k * 20, l, size=10.5, mono=True,
                       fill=C["claude"] if l.startswith(("invited", "actual")) else C["muted"])
    s.text(W / 2, H - 60, "one run folder per seed · every artifact carries its answer key · everything re-runnable from the seed",
           size=12.5, fill=C["muted"], anchor="middle")
    s.text(W / 2, H - 32, "crazyai run · batch · examine · rank · report · compare", size=12, fill=C["line"], anchor="middle", mono=True)
    print("usecases:", rasterise(s.render(), OUT / "use_cases.png", W, H))


if __name__ == "__main__":
    which = sys.argv[1:] or ["primes", "story", "collatz", "usecases"]
    for w in which:
        {"primes": fig_primes, "story": fig_story, "collatz": fig_collatz, "usecases": fig_usecases}[w]()
