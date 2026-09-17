"""The imagination corpus: fragments of human metaphor, painting and story-world.

Three bundled corpora (crazyai/data/imagination/*.yaml) plus anything the AI
has harvested into the archive (archive/imagination/*.yaml). Fragments are the
raw material the blend models cut up and recombine; nothing here is random.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from crazyai.config import ARCHIVE_DIR, DATA_DIR

IMAGINATION_DIR = DATA_DIR / "imagination"
HARVEST_DIR = ARCHIVE_DIR / "imagination"
PROMOTED_DIR = ARCHIVE_DIR / "imagination_promoted"
KINDS = ["metaphor", "painting", "book", "poem"]

_SENT = re.compile(r"(?<=[.!?;:])\s+")
_WORD = re.compile(r"[A-Za-z'-]+")

# function words keep the grammar; everything else is "content" and may be grafted
FUNCTION_WORDS = set("""
a an the and or but nor so yet for of in on at to from by with without into onto over under
above below between among through across along around before after during until while as
is are was were be been being am do does did done has have had having will would shall should
can could may might must ought not no nor never always ever also only just even still yet
this that these those it its they them their there here where when why how what which who whom
whose i you he she we me him her us my your his our one ones some any each every all both few
more most much many such very too so than then if unless because since though although whether
up down out off again once about like near far every anything nothing something everything
someone anyone nobody everyone own same other another else nowhere anywhere somewhere
""".split())


@dataclass(frozen=True)
class Fragment:
    id: str
    kind: str
    source: str
    text: str

    def sentences(self) -> list[str]:
        return [s.strip() for s in _SENT.split(self.text.strip()) if s.strip()]

    def words(self) -> list[str]:
        return _WORD.findall(self.text)


def _load_file(path: Path) -> list[Fragment]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    kind = raw.get("kind", path.stem)
    return [Fragment(f["id"], f.get("kind", kind), f.get("source", ""), f["text"].strip())
            for f in raw.get("fragments", []) if f.get("text")]


@lru_cache(maxsize=1)
def bundled() -> list[Fragment]:
    out: list[Fragment] = []
    for p in sorted(IMAGINATION_DIR.glob("*.yaml")):
        out.extend(_load_file(p))
    return out


def harvest_dir(archive_dir: Path | str | None = None) -> Path:
    return Path(archive_dir) / "imagination" if archive_dir else HARVEST_DIR


def harvested(archive_dir: Path | str | None = None) -> list[Fragment]:
    d = harvest_dir(archive_dir)
    if not d.exists():
        return []
    out: list[Fragment] = []
    for p in sorted(d.glob("*.yaml")):
        out.extend(_load_file(p))
    return out


def promoted_dir(archive_dir: Path | str | None = None) -> Path:
    return Path(archive_dir) / "imagination_promoted" if archive_dir else PROMOTED_DIR


def promoted(archive_dir: Path | str | None = None) -> list[Fragment]:
    """Fragments promoted from a run whose `discovery` beat every prior archived run for its target.

    Separate from `harvested()` deliberately - `corpus()` merges harvest fragments
    unconditionally, but promoted fragments must stay opt-in (`include_promoted`),
    or `--evolve-corpus` would silently affect every other run too.
    """
    d = promoted_dir(archive_dir)
    if not d.exists():
        return []
    out: list[Fragment] = []
    for p in sorted(d.glob("*.yaml")):
        out.extend(_load_file(p))
    return out


def save_promoted(name: str, fragments: list[dict], archive_dir: Path | str | None = None) -> Path:
    """Write a promoted (outcome-selected) fragment into <archive>/imagination_promoted/. Mirrors save_harvest."""
    d = promoted_dir(archive_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.yaml"
    clean = []
    for i, f in enumerate(fragments):
        text = str(f.get("text", "")).strip()
        if len(_WORD.findall(text)) < 8:
            continue
        clean.append({"id": f.get("id") or f"{name}.{i}", "kind": f.get("kind", "book") if f.get("kind") in KINDS else "book",
                      "source": str(f.get("source", "")), "text": text})
    path.write_text(yaml.safe_dump({"kind": "promoted", "fragments": clean}, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")
    return path


def corpus(include_harvest: bool = True, archive_dir: Path | str | None = None,
          include_promoted: bool = False) -> list[Fragment]:
    frags = list(bundled())
    if include_harvest:
        seen = {f.id for f in frags}
        frags += [f for f in harvested(archive_dir) if f.id not in seen]
    if include_promoted:
        seen = {f.id for f in frags}
        frags += [f for f in promoted(archive_dir) if f.id not in seen]
    return frags


def by_kind(frags: list[Fragment]) -> dict[str, list[Fragment]]:
    out: dict[str, list[Fragment]] = {k: [] for k in KINDS}
    for f in frags:
        out.setdefault(f.kind, []).append(f)
    return out


def save_harvest(name: str, fragments: list[dict], archive_dir: Path | str | None = None) -> Path:
    """Write AI-harvested fragments into <archive>/imagination/. Returns the file written."""
    d = harvest_dir(archive_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.yaml"
    clean = []
    for i, f in enumerate(fragments):
        text = str(f.get("text", "")).strip()
        if len(_WORD.findall(text)) < 8:
            continue
        clean.append({"id": f.get("id") or f"{name}.{i}", "kind": f.get("kind", "book") if f.get("kind") in KINDS else "book",
                      "source": str(f.get("source", "")), "text": text})
    path.write_text(yaml.safe_dump({"kind": "harvest", "fragments": clean}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def is_content(word: str) -> bool:
    w = word.lower().strip("'-")
    return len(w) > 2 and w not in FUNCTION_WORDS


DETERMINERS = set("a an the every each some any this that these those its his her their our my your no one another such".split())
PREPOSITIONS = set("of in on at into onto from with by over under through across between among along around behind beyond inside above below".split())
VERB_CUES = set("is are was were be been being can could will would may might must shall should to not never always who which and then still".split())


def slot_class(prev: str | None) -> str:
    """Rough part of speech from the preceding token: N after a determiner/preposition, V after an auxiliary, else O."""
    w = (prev or "").lower()
    if w in DETERMINERS or w in PREPOSITIONS:
        return "N"
    if w in VERB_CUES:
        return "V"
    return "O"


def word_shape(word: str) -> str:
    """A coarse grammatical shape so grafted words keep the sentence readable."""
    w = word.lower()
    cap = "C" if word[:1].isupper() else "l"
    if w.endswith("ing"):
        return cap + "ing"
    if w.endswith("ed"):
        return cap + "ed"
    if w.endswith("ly"):
        return cap + "ly"
    if w.endswith("s") and not w.endswith("ss"):
        return cap + "s"
    return cap + "base"
