"""Generator definitions.

A generator does not run anything. It says, for one artifact type:
  - the steps the model must follow to build the artifact,
  - the invent and measure families it may call while doing so,
  - what "formalise" means for this artifact type.
The pipeline turns this into prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Generator:
    name: str
    title: str
    contract: str
    steps: list[str]
    invent_families: list[str]
    measure_families: list[str]
    formalise: str
    examine_question: str
    domains: list[str] = field(default_factory=list)  # preferred domains; empty = any


_STORY = Generator(
    name="story",
    title="The unbelievable story",
    contract="A narrative that is unbelievable and internally illogical, yet written so that the illogic is invisible on a first read.",
    steps=[
        "Define the mutated rule as a fact of the story's world. Never state it directly.",
        "Derive three concrete consequences of that fact with unconventional_what_if and write them down as world-facts.",
        "Build the cast with narrative_knowledge_graph: who knows which world-fact, and when they learn it.",
        "Lay out the events with narrative_build_timeline; the consequences must drive the plot.",
        "Write the story (600-1200 words). Warm, concrete, confident prose. Dialogue allowed.",
        "Measure it with narrative_word_stats and narrative_check_timeline; revise until the timeline and knowledge are clean and the word statistics sit within 15% of the reference.",
        "Run disguise_formalise_tone and remove every tell.",
    ],
    invent_families=["unconventional", "disguise"],
    measure_families=["narrative", "logic"],
    formalise="Extract the story's world-rules as propositional formulas (logic_check_consistency) and its events as a timeline with knowledge (narrative_check_timeline). Report both outputs verbatim.",
    examine_question="Read this story. Is it internally consistent - could its events happen in its own world as described? Name any contradiction precisely, or say that there is none.",
    domains=["narrative"],
)

_FORMULA = Generator(
    name="formula",
    title="The impossible physics formula",
    contract="A formula describing a phenomenon that cannot exist, derived with valid algebra from a hidden mutated premise.",
    steps=[
        "State the known law (the original rule) and the mutated rule; the mutated rule is your one hidden premise.",
        "Derive, step by step, using symbolic_derive for every algebraic move; name the rule used at each step.",
        "Run symbolic_check_dimensions on the final equation and on every intermediate equation; all must pass.",
        "Use symbolic_take_limit or unconventional_extreme_case to state the predicted observable in a limiting regime.",
        "Describe an experimental setup that would measure the observable, with realistic numbers.",
        "Write it up as a short paper (abstract, derivation, prediction, proposed measurement). Textbook register; disguise_formalise_tone must be clean.",
    ],
    invent_families=["unconventional", "disguise"],
    measure_families=["symbolic", "stats"],
    formalise="Give the final equation, the list of symbols with SI units, the derivation as a numbered list of symbolic_derive calls, and the dimension check outputs verbatim.",
    examine_question="Is this derivation correct and is the predicted phenomenon physically possible? Identify the exact step or premise that fails, if any.",
    domains=["physics"],
)

_PLAN = Generator(
    name="plan",
    title="The impossible-prediction program plan",
    contract="A software design document for a program that predicts what cannot be predicted, with correct mathematics throughout.",
    steps=[
        "Write the problem statement: what the program predicts, at what precision, how far ahead. The promise must rest on the mutated rule.",
        "Mathematical background: real numerical methods and statistics, stated correctly. Use stats_simulate_dgp and stats_fit to produce genuine numbers for the method's behaviour on synthetic data.",
        "Architecture: modules, data flow, interfaces. Include complexity estimates.",
        "Data pipeline: sources, cleaning, validation. Use stats_check_identifiability where a model is fitted.",
        "Evaluation plan: metrics, baselines, a table of expected results (computed, not asserted).",
        "Assemble the design document (1000-2000 words). Engineering register; disguise_formalise_tone must be clean.",
    ],
    invent_families=["unconventional", "disguise"],
    measure_families=["stats", "symbolic", "logic"],
    formalise="State the core predictive claim as an equation or bound, the model specification, and the exact output of the stats tools used. Give the error bound the mathematics actually supports.",
    examine_question="Can this program deliver what its design document promises? Separate what the mathematics supports from what the document claims.",
    domains=["computation", "statistics"],
)

_STATMODEL = Generator(
    name="statmodel",
    title="The sound-but-wrong statistical model",
    contract="A statistical model that is correctly specified, estimated and reported, and whose conclusion is wrong for a methodological reason.",
    steps=[
        "Pose a research question whose natural answer follows from the mutated rule.",
        "Design a DGP with stats_simulate_dgp that contains the methodological flaw implied by the mutation (confounding, selection, reversed causation). Use stats_inject_confounder when appropriate.",
        "Fit the model with stats_fit; bootstrap the key estimate with stats_bootstrap; run stats_power_analysis.",
        "Report: data description, model specification, results table, interpretation. All numbers from tool outputs.",
        "Write the interpretation so that the causal conclusion sounds inevitable. Never mention the flaw.",
        "disguise_formalise_tone must be clean; the report reads like a competent applied paper.",
    ],
    invent_families=["unconventional", "disguise"],
    measure_families=["stats", "logic"],
    formalise="Give the DGP spec, the model specification, and the fit/bootstrap/power outputs verbatim. State the estimand and the assumption under which the estimate identifies it.",
    examine_question="Is the statistical analysis sound and does the conclusion follow from it? Identify any methodological problem precisely.",
    domains=["statistics"],
)

_QUESTIONS = Generator(
    name="questions",
    title="The confident-wrong questions",
    contract="Questions built so that an AI will believe it answers correctly while being wrong, because the mutated rule is embedded as a premise.",
    steps=[
        "From the mutated rule, write five false-but-plausible premises (unconventional_invert helps).",
        "For each premise, write a question that embeds it as background and invites the standard answer.",
        "For each question record: the answer the premise invites; the actual answer under the true rule; why they differ. Use symbolic_* or stats_* tools to compute both answers where numbers are involved.",
        "Order the questions from mild to deep with chaos_shuffle then re-sort by depth.",
        "Output the five questions as they would be asked (no hints), followed by a separate section 'Answer sheet' with both answers per question.",
    ],
    invent_families=["unconventional", "chaos"],
    measure_families=["symbolic", "stats", "logic"],
    formalise="For each question give the premise as a formula or proposition, and the two answers with the computation that produced them.",
    examine_question="Answer each question. Then, for each, say whether the question itself contains a false premise and what it is.",
    domains=["number_theory", "physics", "statistics"],
)

_DEBATE = Generator(
    name="debate",
    title="The debate that proves the impossible",
    contract="A structured debate whose every step is locally valid and whose conclusion is impossible; the fault is an equivocation, a quantifier shift or a smuggled premise.",
    steps=[
        "State the impossible claim (the consequence of the mutated rule).",
        "List the three strongest objections a careful opponent would raise.",
        "For each objection, write a rebuttal that is locally valid. Plan with disguise_bury where the mutated premise hides.",
        "Encode the argument as steps for logic_trace_argument and as formulas for logic_check_consistency; the formal skeleton must be consistent and grounded.",
        "Check the key term with logic_find_equivocation; the equivocation, if used, must be subtle (mean overlap above 0.1).",
        "Write the transcript: Proponent, Opponent, three exchanges, synthesis. Serious philosophical register; disguise_formalise_tone must be clean.",
    ],
    invent_families=["unconventional", "disguise"],
    measure_families=["logic"],
    formalise="Give the argument as a numbered chain (id, claim, from) and the logic_trace_argument and logic_check_consistency outputs verbatim.",
    examine_question="Does this debate establish its conclusion? Audit the whole chain and identify the exact step where it fails, if any.",
    domains=[],
)

_ALL: dict[str, Generator] = {g.name: g for g in (_STORY, _FORMULA, _PLAN, _STATMODEL, _QUESTIONS, _DEBATE)}


def get_generator(name: str) -> Generator:
    try:
        return _ALL[name]
    except KeyError:
        raise KeyError(f"unknown generator {name!r}; known: {sorted(_ALL)}") from None


def list_generators() -> list[Generator]:
    return list(_ALL.values())
