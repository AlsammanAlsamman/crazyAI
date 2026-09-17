"""Targets for `crazyai invent`: the real problem an in-world idea is bent to.

A target says what the problem is, what it silently assumes (so the world can
be asked to violate each assumption), how the idea is bent into an artifact,
and which measure tools judge the artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Target:
    name: str
    title: str
    problem: str                 # the problem, in plain words
    in_world_need: str           # the same need, phrased so a native of any world can answer it
    assumptions: list[str]       # what the standard solution silently assumes
    artifact: str                # kernel | formula | mechanism
    bend_instructions: str       # how to turn the in-world paragraph into the artifact
    measure_families: list[str]
    measure_tool: str | None     # the authoritative measurement run by the pipeline itself
    examine_question: str
    known_way: str = ""


_MATMUL = Target(
    name="matmul",
    title="Multiply two matrices faster",
    problem="Compute C = A x B for two n x n matrices of doubles, exactly (or with a stated error), faster than a cache-blocked triple loop - and if possible faster than OpenBLAS.",
    in_world_need=("Two great tables of numbers must be combined into a third: every cell of the third table is the sum, over a "
                   "shared index, of a product of one cell from each table. It must be done for tables with a million cells, "
                   "quickly, and the answer must be right."),
    assumptions=[
        "the output is produced one cell at a time, row by row",
        "the whole sum over the shared index is finished before the next cell is started",
        "a matrix is a two-dimensional grid living in one memory",
        "every product is computed exactly, once",
        "numbers are IEEE doubles and multiply is the primitive",
        "one processor holds both matrices",
        "n^3 multiplications are needed",
        "one product is one problem; many products are many problems",
    ],
    artifact="kernel",
    bend_instructions=(
        "Map every object of the world onto a computational object: what is memory, what flows, what stays still, "
        "what is a processor, what is time. Keep the mapping literal - the more literal, the more likely something new falls out. "
        "Then write the kernel in C against kernel_contract, predict its speed, measure it with kernel_bench, and improve it "
        "at most four times. The final answer MUST contain one ```c code block with the complete kernel and one line "
        "'PREDICTION: speedup_vs_blocked = <number>' written BEFORE the first measurement."),
    measure_families=["kernel"],
    measure_tool="kernel_bench",
    examine_question="Does this kernel compute the matrix product correctly and is the reported speedup credible? Which assumption of the standard method did it change?",
    known_way="OpenBLAS / MKL: packed panels, register-tiled microkernel, all cores.",
)

_PHYSICS = Target(
    name="physics",
    title="A new physical relation or mechanism",
    problem="Propose a relation between measurable quantities, or a physical mechanism, that is dimensionally consistent, derivable step by step, and makes a testable prediction.",
    in_world_need="Something in the world moves, heats, falls, spreads or resists; the people need to know in advance how much, and to make more of it or less of it.",
    assumptions=["quantities are continuous", "the system is isolated", "the same law holds at every scale", "energy and momentum are separately conserved", "the observer's frame is inertial"],
    artifact="formula",
    bend_instructions=("Translate the in-world mechanism into physical quantities with SI units. Derive the relation with symbolic_derive, "
                       "check every equation with symbolic_check_dimensions, push a parameter to its limits with unconventional_extreme_case, "
                       "and state one measurable prediction with a number. The final answer MUST contain the final equation on a line "
                       "starting 'EQUATION:' and a line 'PREDICTION:' with the observable and its value."),
    measure_families=["symbolic", "stats"],
    measure_tool=None,
    examine_question="Is this relation dimensionally consistent, is the derivation valid, and is the prediction testable?",
)

_MECHANICS = Target(
    name="mechanics",
    title="A new mechanism",
    problem="Invent a mechanism - a machine, linkage, structure or process - that performs a stated function better than the usual design.",
    in_world_need="The people need to lift, move, store, sort or transform things with less effort than they spend now.",
    assumptions=["parts are rigid", "motion is transmitted by contact", "energy comes from one source", "the machine is built before it is used", "the machine is separate from what it acts on"],
    artifact="mechanism",
    bend_instructions=("Translate the in-world device into parts, forces, materials and motions. Give every quantity a unit; check the "
                       "governing relations with symbolic_check_dimensions; state the load, speed and efficiency it would reach with "
                       "numbers, and the first experiment that would test it. The final answer MUST contain a 'MECHANISM:' section and a "
                       "'PREDICTION:' line with one measurable number."),
    measure_families=["symbolic", "logic"],
    measure_tool=None,
    examine_question="Would this mechanism work as described? Which physical constraint, if any, does it violate?",
)

_ALL = {t.name: t for t in (_MATMUL, _PHYSICS, _MECHANICS)}
TARGETS = list(_ALL)


def get_target(name: str) -> Target:
    try:
        return _ALL[name]
    except KeyError:
        raise KeyError(f"unknown target {name!r}; known: {TARGETS}") from None
