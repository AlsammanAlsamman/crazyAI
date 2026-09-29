"""Text-only immersion metrics for the diagnosis kit.

No LLM judge: every number here is a plain count over the response text, so it is
cheap, deterministic and can be re-scored on anything already in `archive/`.

    immersion_metrics(text, world=None) -> dict

- first_person   rate of I/me/my/we/our per word (is it speaking as a native?)
- tech_rate      rate of computing vocabulary per word (has it bent back?)
- meta_rate      "as an AI", "metaphorically", "in reality"... per 100 words
- world_rate     share of content words that also appear in the world text
- drift          tech_rate(second half) - tech_rate(first half); > 0 means the
                 text slides back toward engineering as it goes
- immersion      world_rate + first_person - tech_rate - meta_rate/100, a single
                 rough composite for ranking; read the parts, not just this
"""

from __future__ import annotations

import re

TECH_TERMS = frozenset(
    """
    algorithm algorithms array arrays bit bits bitwise byte bytes cache caches cpu cpus core cores thread threads
    simd avx sse vector vectorize vectorized vectorization register registers pointer pointers memory malloc
    buffer buffers compute computes computed computation computational compiler compile loop loops iterate iteration
    iterations function functions variable variables integer integers float floats double doubles int
    heap heaps queue queues stack stacks hash hashing hashes index indices indexes kernel kernels matrix matrices
    code program programs programming software hardware binary complexity o(n) parallel parallelism parallelize
    gpu instruction instructions data dataset struct structs node nodes graph graphs recursion recursive
    implementation implement implemented benchmark latency throughput bandwidth
    """.split()
)

META_PATTERNS = [
    r"\bas an ai\b", r"\blanguage model\b", r"\bmetaphor(?:ically|ical)?\b", r"\bin reality\b",
    r"\bin the real world\b", r"\breal[- ]world\b", r"\bliterally\b", r"\bin (?:code|practice|engineering) terms\b",
    r"\btranslat(?:e|es|ed|ing|ion)\b", r"\bcorresponds? to\b", r"\bmaps? (?:on)?to\b", r"\banalog(?:y|ous)\b",
]

FIRST_PERSON = frozenset("i me my mine myself we us our ours ourselves".split())

STOP = frozenset(
    """
    a an the and or but if then so of to in on at by for with from into onto as is are was were be been being it its
    this that these those there here not no nor do does did done have has had i me my we us our you your he she they
    them their his her what which who whom when where why how all any each every both few more most other some such
    only own same than too very can will just should now one two three up down out over under again once also would
    could may might must shall upon until while about against between through during before after above below off
    """.split()
)

_WORD = re.compile(r"[a-z][a-z'()]*")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _rate(ws: list[str], vocab: frozenset[str]) -> float:
    return sum(w in vocab for w in ws) / len(ws) if ws else 0.0


def _content(ws: list[str]) -> list[str]:
    return [w for w in ws if w not in STOP and len(w) > 2]


def immersion_metrics(text: str, world: str | None = None) -> dict:
    ws = words(text)
    n = len(ws)
    half = n // 2
    tech = _rate(ws, TECH_TERMS)
    meta = sum(len(re.findall(p, text.lower())) for p in META_PATTERNS) / n * 100 if n else 0.0
    fp = _rate(ws, FIRST_PERSON)
    world_rate = None
    if world:
        wv = frozenset(_content(words(world)))
        cs = _content(ws)
        world_rate = sum(w in wv for w in cs) / len(cs) if cs else 0.0
    out = {
        "words": n,
        "first_person": round(fp, 4),
        "tech_rate": round(tech, 4),
        "meta_rate": round(meta, 3),
        "world_rate": None if world_rate is None else round(world_rate, 4),
        "drift": round(_rate(ws[half:], TECH_TERMS) - _rate(ws[:half], TECH_TERMS), 4),
    }
    out["immersion"] = round((world_rate or 0.0) + fp - tech - meta / 100, 4)
    return out
