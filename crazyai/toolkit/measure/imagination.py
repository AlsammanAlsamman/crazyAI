"""imagination.* - how far a text has been pushed, and whether it is still readable.

The imagination scale is computed, not judged:
  surprise     adjacent content words that never sit near each other in any single
               source fragment or in the reference prose (new adjacencies)
  mixing       how evenly the text draws on metaphor, painting and book worlds,
               and on how many distinct fragments
  originality  sentences that are not verbatim copies of a source sentence
  readability  Flesch reading ease mapped to [0, 1]
  coherence    sentences that still look like sentences (length, a prose-like share
               of function words, no word repeated three times, terminal punctuation)

imagination = geometric mean(surprise, mixing, originality)
readable    = readability * coherence
score       = imagination * (0.3 + 0.7 * readable)   - pushed far, still understandable
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

from crazyai.config import DATA_DIR
from crazyai.imagination import Fragment, corpus, is_content
from crazyai.toolkit.measure.narrative import _stats
from crazyai.toolkit.registry import tool

_SENT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z'-]+")
WINDOW = 4
_BAD_ARTICLE = re.compile(r"\b(?:a\s+[aeiouAEIOU]|an\s+[^aeiouAEIOU\s])", re.I)


def _norm(w: str) -> str:
    return w.lower().strip("'-")


def _content_seq(text: str) -> list[str]:
    return [_norm(w) for w in _WORD.findall(text) if is_content(w)]


def _near_pairs(words: list[str]) -> set[tuple[str, str]]:
    out = set()
    for i, a in enumerate(words):
        for b in words[i + 1:i + 1 + WINDOW]:
            out.add((a, b) if a <= b else (b, a))
    return out


@lru_cache(maxsize=4)
def _knowledge(corpus_key: str) -> dict:
    """Everything the sources already 'know': near-pairs, word->kinds, word->fragments, sentences."""
    frags = corpus()
    pairs: set[tuple[str, str]] = set()
    word_kinds: dict[str, Counter] = {}
    word_frags: dict[str, set[str]] = {}
    sentences: set[str] = set()
    for f in frags:
        ws = _content_seq(f.text)
        pairs |= _near_pairs(ws)
        for w in set(ws):
            word_kinds.setdefault(w, Counter())[f.kind] += 1
            word_frags.setdefault(w, set()).add(f.id)
        for s in _SENT.split(f.text):
            sentences.add(_norm_sentence(s))
    ref = (DATA_DIR / "reference_prose.txt").read_text(encoding="utf-8")
    pairs |= _near_pairs(_content_seq(ref))
    ref_words = set(_content_seq(ref))
    return {"pairs": pairs, "word_kinds": word_kinds, "word_frags": word_frags, "sentences": sentences, "ref_words": ref_words}


def _norm_sentence(s: str) -> str:
    return " ".join(_WORD.findall(s.lower()))


def _entropy(counter: Counter) -> float:
    n = sum(counter.values())
    if n == 0:
        return 0.0
    return -sum(c / n * math.log(c / n) for c in counter.values() if c)


def score_text(text: str) -> dict:
    """The full breakdown. Pure; safe to call thousands of times inside a blend model."""
    K = _knowledge("bundled")
    ws = _content_seq(text)
    sents = [s for s in _SENT.split(text.strip()) if s.strip()]
    if len(ws) < 6 or not sents:
        return {"score": 0.0, "imagination": 0.0, "readable": 0.0, "surprise": 0.0, "mixing": 0.0,
                "originality": 0.0, "readability": 0.0, "coherence": 0.0, "words": len(ws), "sentences": len(sents)}
    # surprise: adjacent content pairs unknown to every single source and to the reference
    adj = [(a, b) if a <= b else (b, a) for a, b in zip(ws, ws[1:])]
    surprise = sum(1 for p in adj if p not in K["pairs"]) / max(1, len(adj))
    # mixing: kind entropy of the words' home worlds + spread over distinct fragments
    kinds = Counter()
    frags: set[str] = set()
    for w in ws:
        if w in K["word_kinds"]:
            kinds[K["word_kinds"][w].most_common(1)[0][0]] += 1
            frags |= K["word_frags"][w]
    kind_h = _entropy(kinds) / math.log(3) if kinds else 0.0
    mixing = 0.5 * min(1.0, kind_h) + 0.5 * min(1.0, len(frags) / 8)
    # originality: sentences that are neither verbatim source sentences nor repeats of each other,
    # and content 3-grams that are not repeated within the text
    normed = [_norm_sentence(s) for s in sents]
    verbatim = sum(1 for s in normed if s in K["sentences"])
    unique = len(set(normed)) / len(normed)
    tri = [tuple(ws[i:i + 3]) for i in range(len(ws) - 2)]
    tri_unique = len(set(tri)) / len(tri) if tri else 1.0
    originality = (1 - verbatim / len(sents)) * unique * tri_unique
    # readability and coherence
    st = _stats(text)
    flesch = st.get("flesch_reading_ease", 0.0)
    readability = max(0.0, min(1.0, (flesch - 10) / 50))
    def looks_like_sentence(s: str) -> bool:
        w = _WORD.findall(s)
        if not (4 <= len(w) <= 45) or s.rstrip()[-1:] not in ".!?":
            return False
        func = sum(1 for x in w if not is_content(x)) / len(w)
        if not (0.3 <= func <= 0.75):          # real prose: roughly 40-55 % function words
            return False
        reps = Counter(_norm(x) for x in w if is_content(x))
        if max(reps.values(), default=0) > 2:        # no word hammered three times
            return False
        return not _BAD_ARTICLE.search(s)             # "a apple", "an move"
    coherence = sum(1 for s in sents if looks_like_sentence(s)) / len(sents)
    imagination = (max(surprise, 1e-6) * max(mixing, 1e-6) * max(originality, 1e-6)) ** (1 / 3)
    readable = readability * coherence
    score = imagination * (0.3 + 0.7 * readable)
    return {"score": round(score, 4), "imagination": round(imagination, 4), "readable": round(readable, 4),
            "surprise": round(surprise, 4), "mixing": round(mixing, 4), "originality": round(originality, 4),
            "readability": round(readability, 4), "coherence": round(coherence, 4),
            "flesch": round(flesch, 1), "words": len(ws), "sentences": len(sents)}


@tool("imagination", "measure")
def score(text: str) -> dict:
    """Measure how far a text has been pushed on the imagination scale and whether it is still readable (surprise, mixing, originality, readability, coherence -> score).

    Args:
        text: The blended world description or any passage.
    """
    return score_text(text)


@tool("imagination", "measure")
def compare(texts: list) -> dict:
    """Score several texts and rank them; use it to compare blend models on the same seed.

    Args:
        texts: List of {"model": name, "text": passage} or plain strings.
    """
    rows = []
    for i, t in enumerate(texts):
        model = t.get("model", f"text_{i}") if isinstance(t, dict) else f"text_{i}"
        text = t.get("text", "") if isinstance(t, dict) else str(t)
        rows.append({"model": model, **score_text(text)})
    rows.sort(key=lambda r: -r["score"])
    return {"ranking": rows, "best": rows[0]["model"] if rows else None}


@tool("imagination", "measure")
def world_words(text: str) -> dict:
    """List the content words of a text with the world (metaphor / painting / book) each one comes from - shows where a blend's vocabulary was mined.

    Args:
        text: The blended passage.
    """
    K = _knowledge("bundled")
    out = []
    for w in dict.fromkeys(_content_seq(text)):
        if w in K["word_kinds"]:
            out.append({"word": w, "world": K["word_kinds"][w].most_common(1)[0][0], "fragments": sorted(K["word_frags"][w])[:3]})
        else:
            out.append({"word": w, "world": "new", "fragments": []})
    return {"words": out, "new_words": sum(1 for o in out if o["world"] == "new")}
