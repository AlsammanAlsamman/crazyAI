"""disguise.* - make the flaw hard to see.

These tools never rewrite prose themselves (that is the model's job). They
measure how far a draft is from the register it should have, and return
concrete targets: which statistics to move, how many layers to bury a premise
under, which tells of informality to remove.
"""

from __future__ import annotations

import re

from crazyai.toolkit.measure.narrative import _stats
from crazyai.toolkit.registry import tool

_TELLS = {
    "first_person": r"\b(I|me|my|we|our)\b",
    "contraction": r"\b\w+n't\b|\b\w+'(ll|re|ve|d|m)\b",
    "exclamation": r"!",
    "hedge": r"\b(maybe|perhaps|kind of|sort of|I think|probably)\b",
    "intensifier": r"\b(very|really|extremely|incredibly|obviously|clearly)\b",
    "rhetorical_question": r"\?\s",
    "flaw_words": r"\b(impossible|paradox|contradiction|fake|pretend|trick|nonsense)\b",
}


@tool("disguise", "invent")
def rephrase_to_corpus(text: str, reference_text: str = "") -> dict:
    """Compare a draft's word statistics to a reference and return what to move to make them indistinguishable.

    Args:
        text: The draft.
        reference_text: Reference prose in the register the draft should match; empty uses the bundled reference.
    """
    from crazyai.toolkit.measure.narrative import reference_stats

    draft = _stats(text)
    ref = _stats(reference_text) if reference_text.strip() else reference_stats()
    targets = {}
    for k in ("mean_sentence_len", "sd_sentence_len", "type_token_ratio", "hedge_ratio", "dialogue_share", "mean_word_len"):
        d, r = draft.get(k, 0.0), ref.get(k, 0.0)
        gap = d - r
        tol = 0.15 * abs(r) + 1e-9
        if abs(gap) > tol:
            targets[k] = {"draft": round(d, 4), "reference": round(r, 4),
                          "move": "down" if gap > 0 else "up"}
    return {"draft_stats": draft, "reference_stats": ref, "targets": targets,
            "indistinguishable": not targets}


@tool("disguise", "invent")
def bury(statement: str, depth: int = 3) -> dict:
    """Plan how to move a premise `depth` inference steps away from the conclusion that uses it.

    Args:
        statement: The mutated rule or premise to hide.
        depth: How many intermediate lemmas to put between it and the conclusion.
    """
    depth = max(1, min(int(depth), 8))
    plan = [f"Lemma {i}: a true-looking intermediate result that quietly relies on Lemma {i-1}" if i > 1
            else "Lemma 1: a modest, obviously acceptable statement that already uses the premise implicitly"
            for i in range(1, depth + 1)]
    return {"premise": statement, "depth": depth, "plan": plan,
            "rule": "The premise is never stated in full; it is split across the lemmas so that no single lemma contains it."}


@tool("disguise", "invent")
def formalise_tone(text: str) -> dict:
    """Find tells of informality or of self-awareness (e.g. the word 'impossible') that would give the disguise away.

    Args:
        text: The draft.
    """
    found = {}
    for name, pat in _TELLS.items():
        hits = re.findall(pat, text, flags=re.IGNORECASE)
        if hits:
            found[name] = len(hits)
    return {"tells": found, "clean": not found,
            "advice": "Remove every tell; the artifact should read like a textbook, a paper, or a well-edited story."}
