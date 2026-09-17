"""`crazyai harvest-corpus`: grow a bundled imagination corpus, copyright-safely.

Reuses the same prompt (`invent_prompts.HARVEST_SYSTEM`/`harvest_prompt`) and
response-parsing as `crazyai invent --harvest`, but at standing-corpus scale:
batched calls (lower truncation risk than one huge request), a quality filter
(drop anything short, near-duplicate, or unsurprising against the *existing*
corpus), and a write to a *candidates* file - never straight into a shipped
`crazyai/data/imagination/*.yaml` - so a human skims before anything ships
permanently with the package.

The harvest system prompt already instructs the model to paraphrase from its
own knowledge and never quote copyrighted text; this module adds nothing that
changes that contract, only volume and a quality gate.
"""

from __future__ import annotations

import json
import re
from typing import Any

import yaml

from crazyai.imagination import corpus
from crazyai.pipeline import invent_prompts as P
from crazyai.providers.base import Provider
from crazyai.toolkit.measure.imagination import score_text
from crazyai.toolkit.registry import build_toolkit

_FRAGS = re.compile(r"\[.*\]", re.S)
_WORD = re.compile(r"[A-Za-z'-]+")


def parse_fragments(text: str) -> list[dict[str, Any]]:
    """Extract a JSON array of {kind, source, text} fragments from a harvest response.

    Shared by `Invent.step_harvest` (one run's small top-up) and `build_corpus`
    (a standing-corpus batch) so the parsing behaves identically either way.
    """
    m = _FRAGS.search(text)
    if not m:
        return []
    try:
        return [f for f in json.loads(m.group(0)) if isinstance(f, dict) and f.get("text")]
    except json.JSONDecodeError:
        return []


# Per-kind nudges: HARVEST_SYSTEM/harvest_prompt describe every kind generically, which
# is not enough steering when a kind has no bundled examples yet for the model to pattern-match against.
_KIND_HINTS = {
    "poem": (
        " Spread across cultures and eras, not just English-language poetry: include classical Arabic poetry "
        "(e.g. the Mu'allaqat, Antara ibn Shaddad), Persian (Rumi, Hafez), other non-Western traditions, and "
        "Western poets (Blake, Rilke, Neruda, Dickinson) - whichever are the most imaginative you know, in any "
        "language, described in your own words rather than any translator's exact lines."
    ),
}


def harvest_batch(provider: Provider, kind: str, n: int, avoid: list[str]) -> list[dict[str, Any]]:
    """One harvest call for a single kind, up to n fragments, avoiding the given sources."""
    toolkit = build_toolkit(0)
    prompt = P.harvest_prompt(n, [kind], avoid) + _KIND_HINTS.get(kind, "")
    res = provider.agent(P.HARVEST_SYSTEM, prompt, toolkit, [], 2)
    return parse_fragments(res.text)


def _norm(s: str) -> str:
    return " ".join(_WORD.findall(s.lower()))


def build_corpus(provider: Provider, kind: str, n: int, batch_size: int = 15,
                 min_surprise: float = 0.5, log=print) -> list[dict[str, Any]]:
    """Harvest up to n fragments of one kind in batches, quality-filtered against the existing corpus.

    Rejects: fewer than 8 words, a kind mismatch, a near-duplicate source or
    normalized text (against the existing corpus and against fragments
    already accepted in this call), and anything scoring below `min_surprise`
    on the existing `surprise` term (unsurprising vs. what the corpus already
    knows). Returns accepted fragments; does not write anything to disk.
    """
    existing = corpus()
    seen_sources = {f.source.strip().lower() for f in existing}
    seen_texts = {_norm(f.text) for f in existing}
    accepted: list[dict[str, Any]] = []

    while len(accepted) < n:
        batch_n = min(batch_size, n - len(accepted))
        avoid = sorted(seen_sources)[:40]
        raw = harvest_batch(provider, kind, batch_n, avoid)
        if not raw:
            log(f"harvest-corpus: empty/unparseable batch for kind={kind}, stopping early "
               f"({len(accepted)}/{n} collected)")
            break
        added_this_batch = 0
        for f in raw:
            text = str(f.get("text", "")).strip()
            source = str(f.get("source", "")).strip()
            if len(_WORD.findall(text)) < 8:
                continue
            if f.get("kind", kind) != kind:
                continue
            norm = _norm(text)
            if not source or source.lower() in seen_sources or norm in seen_texts:
                continue
            surprise = score_text(text).get("surprise", 0.0)
            if surprise < min_surprise:
                continue
            accepted.append({"kind": kind, "source": source, "text": text, "surprise": round(surprise, 4)})
            seen_sources.add(source.lower())
            seen_texts.add(norm)
            added_this_batch += 1
        log(f"harvest-corpus: batch of {len(raw)} -> {added_this_batch} accepted "
           f"({len(accepted)}/{n} total, kind={kind})")
        if added_this_batch == 0:
            log("harvest-corpus: a full batch added nothing new, stopping early")
            break
    return accepted


def save_candidates(fragments: list[dict[str, Any]], kind: str, out_path) -> None:
    clean = [{"id": f"{kind}.candidate.{i}", "kind": f["kind"], "source": f["source"], "text": f["text"]}
             for i, f in enumerate(fragments)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump({"kind": kind, "fragments": clean}, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
