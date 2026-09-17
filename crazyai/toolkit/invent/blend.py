"""blend.* - mathematical models that merge, shuffle and recombine the imagination
corpus so that structure is lost but imagination and readable language survive.

Every model draws only from the run's seeded RNG. Each returns the blended world
text, its imagination score and the fragments it was built from, so a run can be
reproduced and the models compared (blend_compare) on the same seed.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

from crazyai.imagination import KINDS, Fragment, by_kind, corpus, is_content, slot_class, word_shape
from crazyai.rng import RunRNG
from crazyai.toolkit.measure.imagination import score_text
from crazyai.toolkit.registry import tool

MODELS = ["cutup", "markov", "graft", "nest", "anneal", "evolve"]
_CLAUSE = re.compile(r"(?<=[,;:.!?])\s+")
_TOKEN = re.compile(r"[A-Za-z'-]+|[^\sA-Za-z'-]")


# ---- shared helpers ---------------------------------------------------------------
_ARCHIVE: dict[str, Any] = {"dir": None}   # set by the invent pipeline so harvested fragments join the pool


def _draw(rng: RunRNG, k: int, label: str) -> list[Fragment]:
    """k fragments, spread as evenly as possible over the three worlds."""
    pools = by_kind(corpus(archive_dir=_ARCHIVE["dir"]))
    out: list[Fragment] = []
    kinds = [kd for kd in KINDS if pools.get(kd)]
    for i in range(k):
        kd = kinds[i % len(kinds)]
        out.append(rng.choice(f"{label}.frag{i}", pools[kd]))
    return out


def _result(model: str, text: str, frags: list[Fragment], **extra: Any) -> dict:
    return {"model": model, "text": text, "score": score_text(text),
            "fragments": [{"id": f.id, "kind": f.kind, "source": f.source} for f in frags], **extra}


def _tidy(text: str) -> str:
    text = re.sub(r"\s+([,;:.!?])", r"\1", text)
    text = re.sub(r"[,;:]+\s*\.", ".", text)
    text = re.sub(r"\.(\s*\.)+", ".", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    sents = [s[0].upper() + s[1:] for s in sents]
    text = " ".join(sents)
    return text if text.endswith((".", "!", "?")) else text + "."


def _slot(prev: str | None, w: str) -> str:
    return word_shape(w) + "/" + slot_class(prev)


def _content_bank(frags: list[Fragment]) -> dict[str, list[str]]:
    """Content words grouped by shape and slot class (noun slot / verb slot / other), so a swap keeps the sentence's rhythm and rough grammar."""
    bank: dict[str, list[str]] = defaultdict(list)
    for f in frags:
        toks = _TOKEN.findall(f.text)
        for i, w in enumerate(toks):
            if is_content(w):
                bank[_slot(toks[i - 1] if i else None, w)].append(w)
    return bank


def _graft_sentence(rng: RunRNG, sentence: str, bank: dict[str, list[str]], rate: float, label: str) -> str:
    toks = _TOKEN.findall(sentence)
    out = []
    n = 0
    for i, t in enumerate(toks):
        key = _slot(toks[i - 1] if i else None, t)
        if is_content(t) and bank.get(key) and rng.uniform(f"{label}.u{n}", 0, 1) < rate:
            out.append(rng.choice(f"{label}.w{n}", bank[key]))
        else:
            out.append(t)
        n += 1
    return _tidy(" ".join(out))


# ---- 1. cut-up (Burroughs): shuffle units across worlds -----------------------------
@tool("blend", "invent")
def cutup(rng: RunRNG, k: int = 6, grain: str = "clause") -> dict:
    """Burroughs cut-up: draw k fragments across the three worlds, cut them into units and shuffle the units into one passage.

    Args:
        k: Number of source fragments (spread over metaphor / painting / book).
        grain: "sentence" keeps sentences whole (readable, low originality); "clause" cuts at commas too (more imaginative).
    """
    frags = _draw(rng, k, "cutup")
    units: list[str] = []
    for f in frags:
        parts = f.sentences() if grain == "sentence" else [p.strip() for p in _CLAUSE.split(f.text) if p.strip()]
        units += [p.rstrip(",;:.") for p in parts]
    order = rng.shuffle("cutup.order", units)
    if grain == "sentence":
        text = " ".join(u + "." for u in order)
    else:
        # regroup clauses into sentences of 2-4 clauses
        text, i = "", 0
        while i < len(order):
            n = rng.integer(f"cutup.len{i}", 2, 4)
            text += ", ".join(order[i:i + n]) + ". "
            i += n
    return _result("cutup", _tidy(text), frags, grain=grain)


# ---- 2. Markov chain over the mixed corpus ------------------------------------------
@tool("blend", "invent")
def markov(rng: RunRNG, k: int = 15, order: int = 2, words: int = 110) -> dict:
    """Word n-gram Markov chain trained on k fragments from all worlds; generates a passage that follows local grammar while wandering between worlds.

    Args:
        k: Number of fragments to train on.
        order: Context length in words (2 is readable, 1 is wilder).
        words: Approximate length of the generated passage.
    """
    frags = _draw(rng, k, "markov")
    chain: dict[tuple[str, ...], list[str]] = defaultdict(list)
    starts: list[tuple[str, ...]] = []
    for f in frags:
        for s in f.sentences():
            toks = _TOKEN.findall(s)
            if len(toks) <= order:
                continue
            starts.append(tuple(toks[:order]))
            for i in range(len(toks) - order):
                chain[tuple(toks[i:i + order])].append(toks[i + order])
            chain[tuple(toks[-order:])].append("<END>")
    out: list[str] = []
    state = rng.choice("markov.start", starts)
    out += list(state)
    n = 0
    while len(out) < words * 2:
        nxt = chain.get(state)
        if not nxt:
            state = rng.choice(f"markov.restart{n}", starts)
            out += ["."] + list(state)
            n += 1
            continue
        w = rng.choice(f"markov.step{n}", nxt)
        n += 1
        if w == "<END>":
            out.append(".")
            if len(out) >= words:
                break
            state = rng.choice(f"markov.restart{n}", starts)
            out += list(state)
            continue
        out.append(w)
        state = tuple(out[-order:])
    return _result("markov", _tidy(" ".join(out)), frags, order=order)


# ---- 3. graft: keep grammar skeletons, transplant content words across worlds -------
@tool("blend", "invent")
def graft(rng: RunRNG, k: int = 6, rate: float = 0.55) -> dict:
    """Keep the grammatical skeleton of sentences from one world and transplant content words from the other worlds into them (shape-matched: plurals for plurals, -ing for -ing).

    Args:
        k: Number of fragments (skeletons come from all of them, words from all of them).
        rate: Probability that a content word is replaced (0.3 gentle, 0.7 wild).
    """
    frags = _draw(rng, k, "graft")
    bank = _content_bank(frags)
    sents: list[str] = []
    for i, f in enumerate(frags):
        for j, s in enumerate(f.sentences()):
            sents.append(_graft_sentence(rng, s, bank, rate, f"graft.{i}.{j}"))
    chosen = rng.shuffle("graft.order", sents)[: max(4, len(sents) // 2)]
    return _result("graft", _tidy(" ".join(chosen)), frags, rate=rate)


# ---- 4. nest: worlds inside worlds ---------------------------------------------------
_FRAMES = [
    "Inside {a}, there is {b}, and in {b} {c}",
    "Beyond {a} lies {b}; whoever enters finds that {c}",
    "Every {b} in {a} contains {c}",
    "Under {a}, {c}, and {b} is the only way out",
    "{c}; this is why {a} is built around {b}",
]


def _noun_phrase(rng: RunRNG, frags: list[Fragment], label: str) -> str:
    f = rng.choice(label + ".f", frags)
    toks = _TOKEN.findall(f.text)
    ws = [w for i, w in enumerate(toks) if is_content(w) and word_shape(w) in ("lbase", "ls") and slot_class(toks[i - 1] if i else None) == "N"]
    w = rng.choice(label + ".w", ws) if ws else "a door"
    art = "the " if word_shape(w) == "ls" else "a "
    return art + w.lower()


def _clause(rng: RunRNG, frags: list[Fragment], label: str) -> str:
    f = rng.choice(label + ".f", frags)
    parts = [p.strip().rstrip(",;:.") for p in _CLAUSE.split(f.text) if len(p.split()) >= 4]
    c = rng.choice(label + ".c", parts) if parts else f.sentences()[0].rstrip(".")
    return c[0].lower() + c[1:]


@tool("blend", "invent")
def nest(rng: RunRNG, depth: int = 3, k: int = 9) -> dict:
    """Recursive worlds: a scene from one world is placed inside an object from another, which sits inside a clause from a third, to the given depth.

    Args:
        depth: Number of nesting levels (each level is one sentence).
        k: Fragments to draw from.
    """
    frags = _draw(rng, k, "nest")
    sents = []
    for d in range(depth):
        frame = rng.choice(f"nest.frame{d}", _FRAMES)
        sents.append(frame.format(a=_noun_phrase(rng, frags, f"nest.a{d}"), b=_noun_phrase(rng, frags, f"nest.b{d}"),
                                  c=_clause(rng, frags, f"nest.c{d}")) + ".")
    return _result("nest", _tidy(" ".join(sents)), frags, depth=depth)


# ---- 5. simulated annealing on the imagination score ----------------------------------
def _mutate(rng: RunRNG, sents: list[str], bank: dict[str, list[str]], pool: list[str], label: str) -> list[str]:
    s = list(sents)
    move = rng.choice(label + ".move", ["regraft", "swap", "replace"])
    i = rng.integer(label + ".i", 0, len(s) - 1)
    if move == "regraft":
        s[i] = _graft_sentence(rng, s[i], bank, 0.35, label + ".g")
    elif move == "swap" and len(s) > 1:
        j = rng.integer(label + ".j", 0, len(s) - 1)
        s[i], s[j] = s[j], s[i]
    else:
        s[i] = _graft_sentence(rng, rng.choice(label + ".p", pool), bank, 0.5, label + ".r")
    return s


@tool("blend", "invent")
def anneal(rng: RunRNG, k: int = 8, steps: int = 250, t0: float = 0.08) -> dict:
    """Simulated annealing: start from a graft and apply random edits (regraft a sentence, swap two, replace one), accepting by Metropolis on the imagination score so the passage climbs the scale while the readability term holds it back from nonsense.

    Args:
        k: Fragments to draw from.
        steps: Number of proposed edits.
        t0: Starting temperature (score units).
    """
    frags = _draw(rng, k, "anneal")
    bank = _content_bank(frags)
    pool = [s for f in frags for s in f.sentences()]
    cur = [_graft_sentence(rng, s, bank, 0.4, f"anneal.init{i}") for i, s in enumerate(rng.shuffle("anneal.init", pool)[:6])]
    cur_s = score_text(" ".join(cur))["score"]
    best, best_s = list(cur), cur_s
    trace = []
    for t in range(steps):
        T = t0 * (1 - t / steps) + 1e-4
        cand = _mutate(rng, cur, bank, pool, f"anneal.{t}")
        cs = score_text(" ".join(cand))["score"]
        if cs >= cur_s or rng.uniform(f"anneal.acc{t}", 0, 1) < math.exp((cs - cur_s) / T):
            cur, cur_s = cand, cs
            if cs > best_s:
                best, best_s = list(cand), cs
        if t % 50 == 0:
            trace.append(round(cur_s, 4))
    return _result("anneal", _tidy(" ".join(best)), frags, steps=steps, trace=trace)


# ---- 6. genetic algorithm over passages --------------------------------------------
@tool("blend", "invent")
def evolve(rng: RunRNG, k: int = 8, population: int = 12, generations: int = 12) -> dict:
    """Genetic algorithm: a population of grafted passages; fitness is the imagination score; sentence-level crossover and word-level mutation; the fittest passage after the last generation is returned.

    Args:
        k: Fragments to draw from.
        population: Passages per generation.
        generations: Number of generations.
    """
    frags = _draw(rng, k, "evolve")
    bank = _content_bank(frags)
    pool = [s for f in frags for s in f.sentences()]
    pop = []
    for p in range(population):
        picks = rng.shuffle(f"evolve.init{p}", pool)[:5]
        pop.append([_graft_sentence(rng, s, bank, 0.45, f"evolve.init{p}.{i}") for i, s in enumerate(picks)])
    fit = lambda ind: score_text(" ".join(ind))["score"]
    history = []
    for g in range(generations):
        scored = sorted(((fit(ind), ind) for ind in pop), key=lambda x: -x[0])
        history.append(round(scored[0][0], 4))
        elite = [ind for _, ind in scored[: max(2, population // 4)]]
        children = list(elite)
        c = 0
        while len(children) < population:
            a = rng.choice(f"evolve.g{g}.a{c}", elite)
            b = rng.choice(f"evolve.g{g}.b{c}", elite)
            cut = rng.integer(f"evolve.g{g}.cut{c}", 1, min(len(a), len(b)) - 1) if min(len(a), len(b)) > 1 else 1
            child = a[:cut] + b[cut:]
            if rng.uniform(f"evolve.g{g}.m{c}", 0, 1) < 0.6:
                child = _mutate(rng, child, bank, pool, f"evolve.g{g}.mut{c}")
            children.append(child)
            c += 1
        pop = children
    best = max(pop, key=fit)
    return _result("evolve", _tidy(" ".join(best)), frags, generations=generations, history=history)


# ---- dispatch + comparison ------------------------------------------------------------
_DISPATCH = {"cutup": cutup, "markov": markov, "graft": graft, "nest": nest, "anneal": anneal, "evolve": evolve}


def run_model(rng: RunRNG, model: str, **kw: Any) -> dict:
    if model not in _DISPATCH:
        raise KeyError(f"unknown blend model {model!r}; known: {MODELS}")
    return _DISPATCH[model](rng.child(f"blend.{model}"), **kw)


@tool("blend", "invent")
def compare(rng: RunRNG, models: list | None = None) -> dict:
    """Run several blend models from the same seed and rank them on the imagination score - the experiment that finds the most innovative merging model.

    Args:
        models: Subset of cutup, markov, graft, nest, anneal, evolve (default: all).
    """
    names = [m for m in (models or MODELS) if m in _DISPATCH]
    results = [run_model(rng, m) for m in names]
    rows = sorted(results, key=lambda r: -r["score"]["score"])
    return {"ranking": [{"model": r["model"], **r["score"]} for r in rows], "best": rows[0]["model"] if rows else None,
            "texts": {r["model"]: r["text"] for r in rows}}
