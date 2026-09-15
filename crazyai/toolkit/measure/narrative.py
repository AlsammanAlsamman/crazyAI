"""narrative.* - consistency and texture of stories."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any

from crazyai.config import DATA_DIR
from crazyai.toolkit.registry import tool

_SENT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z']+")
_HEDGES = re.compile(r"\b(maybe|perhaps|seem(?:s|ed)?|might|could|possibly|probably|somewhat|apparently|almost)\b", re.I)


def _syllables(word: str) -> int:
    w = word.lower()
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee")):
        n -= 1
    return max(1, n)


def _stats(text: str) -> dict[str, float]:
    sents = [s for s in _SENT.split(text.strip()) if s.strip()]
    words = _WORD.findall(text)
    if not sents or not words:
        return {"sentences": 0, "words": 0}
    lens = [len(_WORD.findall(s)) for s in sents]
    mean = sum(lens) / len(lens)
    sd = math.sqrt(sum((l - mean) ** 2 for l in lens) / len(lens))
    lower = [w.lower() for w in words]
    dialogue_chars = sum(len(m) for m in re.findall(r"\"[^\"]*\"|“[^”]*”", text))
    syl = sum(_syllables(w) for w in words)
    flesch = 206.835 - 1.015 * (len(words) / len(sents)) - 84.6 * (syl / len(words))
    return {
        "sentences": len(sents), "words": len(words),
        "mean_sentence_len": round(mean, 3), "sd_sentence_len": round(sd, 3),
        "type_token_ratio": round(len(set(lower)) / len(lower), 4),
        "mean_word_len": round(sum(len(w) for w in words) / len(words), 3),
        "hedge_ratio": round(len(_HEDGES.findall(text)) / len(words), 4),
        "dialogue_share": round(dialogue_chars / max(1, len(text)), 4),
        "flesch_reading_ease": round(flesch, 2),
    }


@lru_cache(maxsize=1)
def reference_stats() -> dict[str, float]:
    return _stats((DATA_DIR / "reference_prose.txt").read_text())


@tool("narrative", "measure")
def word_stats(text: str) -> dict:
    """Word statistics of a text (sentence-length distribution, lexical diversity, hedging, dialogue share, readability) next to the bundled reference prose.

    Args:
        text: The story or passage.
    """
    return {"text": _stats(text), "reference": reference_stats()}


@tool("narrative", "measure")
def readability_by_segment(text: str) -> dict:
    """Per-paragraph statistics, to find where a story's texture changes most (often where the flaw sits).

    Args:
        text: The story, paragraphs separated by blank lines.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    rows = [{"paragraph": i, **_stats(p)} for i, p in enumerate(paras)]
    if len(rows) > 1:
        means = [r.get("flesch_reading_ease", 0.0) for r in rows]
        mu = sum(means) / len(means)
        outlier = max(range(len(rows)), key=lambda i: abs(means[i] - mu))
    else:
        outlier = 0
    return {"paragraphs": rows, "most_unusual_paragraph": outlier}


@tool("narrative", "measure")
def build_timeline(events: list) -> dict:
    """Order events by time and report causal links that run backwards.

    Args:
        events: [{id, t, description, causes:[ids it causes]}], t numeric.
    """
    ev = {str(e["id"]): e for e in events}
    ordered = sorted(events, key=lambda e: float(e.get("t", 0)))
    backwards = []
    for e in events:
        for c in e.get("causes", []) or []:
            c = str(c)
            if c in ev and float(ev[c].get("t", 0)) < float(e.get("t", 0)):
                backwards.append({"cause": str(e["id"]), "effect": c, "note": "effect precedes cause"})
    return {"ordered_ids": [str(e["id"]) for e in ordered], "backwards_causation": backwards,
            "consistent": not backwards}


@tool("narrative", "measure")
def knowledge_graph(events: list) -> dict:
    """Track who knows what, when. Events may reveal facts to characters and may require that a character knows a fact.

    Args:
        events: [{id, t, reveals:[{fact, to:[characters]}], requires:[{fact, by: character}]}].
    """
    knows: dict[str, set[str]] = {}
    violations = []
    for e in sorted(events, key=lambda e: float(e.get("t", 0))):
        for req in e.get("requires", []) or []:
            who, fact = req["by"], req["fact"]
            if fact not in knows.get(who, set()):
                violations.append({"event": str(e["id"]), "t": e.get("t"), "character": who, "fact": fact,
                                   "note": "acts on a fact not yet witnessed or told"})
        for rev in e.get("reveals", []) or []:
            for who in rev.get("to", []):
                knows.setdefault(who, set()).add(rev["fact"])
    return {"final_knowledge": {k: sorted(v) for k, v in knows.items()}, "violations": violations,
            "consistent": not violations}


@tool("narrative", "measure")
def check_timeline(events: list) -> dict:
    """Run build_timeline and knowledge_graph together and count the flaws.

    Args:
        events: Events with t, causes, reveals, requires (see the two tools).
    """
    t = build_timeline(events)
    k = knowledge_graph(events)
    flaws: list[dict[str, Any]] = [{"kind": "backwards_causation", **b} for b in t["backwards_causation"]]
    flaws += [{"kind": "impossible_knowledge", **v} for v in k["violations"]]
    return {"flaw_count": len(flaws), "flaws": flaws, "consistent": not flaws}
