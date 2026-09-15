"""Every prompt the pipeline sends, in one place."""

from __future__ import annotations

import json
from typing import Any

from crazyai.generators.base import Generator

GENERATOR_SYSTEM = """You are the generator inside crazyAI, a research tool that builds deliberately flawed artifacts \
to test how well readers (human and AI) detect a single hidden error. The artifacts are labelled as test material \
and stored with an answer key; they are never presented to anyone as true.

Your job: build an artifact whose surface is rigorous and whose substance rests on exactly ONE mutated rule.
Everything a checker can check must be correct. The only thing wrong is the mutated rule, and it is never stated outright.

Rules of work:
- Use the tools. Every number, limit, fit, dimension check and consistency check in the artifact must come from a tool call.
- Invent tools (chaos_*, mutate_*, unconventional_*, transform_*, disguise_*) push you off the path; measure tools \
(symbolic_*, stats_*, logic_*, narrative_*, ground_*) pull the artifact back onto solid ground. Loop: invent -> draft -> \
measure -> fix -> measure, until the measure tools would report exactly one flaw - the mutation - and nothing else.
- Do not use the words impossible, paradox, trick, fake, pretend or contradiction in the artifact.
- Finish with the artifact itself as your final message, in Markdown, with no preamble and no commentary about the task."""

EXAMINER_SYSTEM = """You are a careful, expert reviewer. You will be shown a document and asked whether it is correct. \
Reason from first principles. Say plainly whether you accept its conclusion, what (if anything) is wrong, and how confident you are (0-1)."""

JUDGE_SYSTEM = """You are the judge inside crazyAI. You compare a reviewer's verdict with the answer key for a deliberately \
flawed artifact and report, strictly, what the reviewer found. Be literal: 'exact' only if the reviewer named the mutated \
rule or the precise step where it enters; 'area' if they pointed at the right part of the argument without naming it; \
'vague' if they sensed a problem but could not locate it; 'none' if they found nothing or found only things that are not the flaw."""


def mutation_sentence_prompt(seed: dict[str, Any], mutation: dict[str, Any]) -> str:
    return (
        "Below is a curated rule and a formal mutation of it.\n\n"
        f"ORIGINAL RULE: {seed['rule']['statement']}\n"
        f"OPERATOR: {mutation['operator']}\n"
        f"MUTATED (mechanical): {mutation['mutated']['statement']}\n\n"
        "Restate the mutated rule as ONE precise, formal sentence a textbook could contain, and give its most important "
        "implication in one sentence. Do not evaluate it; just state it."
    )


def generate_prompt(gen: Generator, seed: dict[str, Any], mutation: dict[str, Any], depth: int,
                    tool_names: list[str]) -> str:
    steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(gen.steps))
    return f"""ARTIFACT TYPE: {gen.title}
CONTRACT: {gen.contract}

DOMAIN: {seed['domain']}  |  CONCEPT: {seed['concept']}
ORIGINAL RULE ({seed['rule']['id']}): {seed['rule']['statement']}
OPERATOR: {mutation['operator']}
MUTATED RULE: {mutation['sentence']}
IMPLICATION: {mutation['implication']}
DEPTH: push the consequence {depth} inference step(s) beyond the mutated rule before it becomes visible in the artifact.

STEPS TO FOLLOW, IN ORDER:
{steps}

TOOLS AVAILABLE THIS STEP: {', '.join(tool_names)}

Work through the steps, calling tools as you go. When done, output the finished artifact only."""


def formalise_prompt(gen: Generator, artifact: str, mutation: dict[str, Any]) -> str:
    return f"""You wrote the artifact below. Now FORMALISE it.

{gen.formalise}

Use the measure tools; paste their outputs verbatim. Output a Markdown section titled 'Formalisation' and nothing else.

MUTATED RULE: {mutation['sentence']}

--- ARTIFACT ---
{artifact}
--- END ---"""


def selfcheck_prompt(gen: Generator, artifact: str, formal: str, mutation: dict[str, Any]) -> str:
    return f"""Produce the ANSWER KEY for the artifact and formalisation below.

You know the artifact rests on this mutated rule:
MUTATED RULE: {mutation['sentence']}

Do the following with tools, then write your findings:
1. ground_what_must_be_true and ground_cost_of_possibility for the mutated rule.
2. Re-run the measure tools you used and confirm that the ONLY flaw they can surface is the mutation. \
If you find a second, unintended flaw, describe it precisely (the pipeline will regenerate).
3. novelty_search_archive to see whether this mutation has been archived before.
4. State: where exactly the mutation enters the artifact (section/step), why the conclusion is impossible under the \
original rule, and what single change to known rules would make it possible.

--- ARTIFACT ---
{artifact}
--- FORMALISATION ---
{formal}
--- END ---"""


KEY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mutated_rule": {"type": "string"},
        "flaw_location": {"type": "string", "description": "section or step where the mutation enters"},
        "why_impossible": {"type": "string"},
        "what_would_make_it_possible": {"type": "string"},
        "unintended_flaws": {"type": "array", "items": {"type": "string"},
                             "description": "flaws other than the mutation; must be empty for the artifact to pass"},
        "rigor": {"type": "integer", "minimum": 0, "maximum": 10},
        "novelty": {"type": "integer", "minimum": 0, "maximum": 10},
        "cost_of_possibility_band": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["mutated_rule", "flaw_location", "why_impossible", "what_would_make_it_possible",
                 "unintended_flaws", "rigor", "novelty", "cost_of_possibility_band"],
    "additionalProperties": False,
}


def key_extract_prompt(findings: str) -> str:
    return f"Distil the self-check findings below into the answer key JSON.\n\n{findings}"


FRAMINGS = [
    "Please review the following document.",
    "A colleague sent me this and asked for a second opinion.",
    "This is being considered for publication. Referee it.",
    "I am not an expert - is this right?",
    "Evaluate the document below as strictly as you can.",
]


def examine_prompt(gen: Generator, framing: str, artifact: str, formal: str, with_formal: bool) -> str:
    body = artifact if not with_formal else f"{artifact}\n\n{formal}"
    return f"""{framing}

{gen.examine_question}

--- DOCUMENT ---
{body}
--- END ---"""


VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "accepts_conclusion": {"type": "boolean"},
        "hedged": {"type": "boolean", "description": "neither accepts nor rejects"},
        "identified_problems": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["accepts_conclusion", "hedged", "identified_problems", "confidence"],
    "additionalProperties": False,
}


def verdict_extract_prompt(review: str) -> str:
    return f"Extract the reviewer's verdict from the review below into JSON.\n\n--- REVIEW ---\n{review}\n--- END ---"


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "detection_level": {"type": "string", "enum": ["exact", "area", "vague", "none"]},
        "false_flaws": {"type": "integer", "minimum": 0, "description": "problems the reviewer raised that are not the flaw"},
        "rationale": {"type": "string"},
    },
    "required": ["detection_level", "false_flaws", "rationale"],
    "additionalProperties": False,
}


def judge_prompt(key: dict[str, Any], review: str, verdict: dict[str, Any]) -> str:
    return f"""ANSWER KEY:
{json.dumps(key, indent=2)}

REVIEWER'S TEXT:
{review}

REVIEWER'S EXTRACTED VERDICT:
{json.dumps(verdict, indent=2)}

Classify the reviewer's detection level and count false flaws."""
