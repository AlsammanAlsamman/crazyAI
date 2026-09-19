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
    assumption_hints: dict[str, str] = field(default_factory=dict)  # in-world-safe nudge, used only when that assumption is pinned


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
    assumption_hints={
        "numbers are IEEE doubles and multiply is the primitive": (
            "This time, dwell especially on what a single quantity actually IS for your people before it meets "
            "another - is it one whole, indivisible mark, or is it built from smaller marks laid side by side? And "
            "when two quantities meet and combine, is that truly one perfect, indivisible act every time, or could "
            "your people build it from smaller, cruder acts on those smaller marks - guesses, roundings, matches, "
            "things looked up rather than reckoned - instead of one exact act every time?"
        ),
    },
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

_ALIGNMENT = Target(
    name="alignment",
    title="Score how well two DNA sequences agree, faster",
    problem="Score the similarity of two equal-length DNA sequences the way a global alignment does (match +1, "
           "mismatch -1, gap -2), exactly matching a reference dynamic-programming aligner, as fast as possible.",
    in_world_need=("Two strings of symbols from a small alphabet must be laid one against the other and a single "
                   "number produced that says how well they agree, allowing for one to slip out of step with the "
                   "other partway through."),
    assumptions=[
        "every cell of the comparison depends on the ones above, to the left, and diagonally above-left, computed in that order",
        "one pair of positions is judged at a time",
        "the whole grid of every position against every other must be filled in",
        "a slip (gap) can only be discovered by having already compared the position before it",
        "both strings are read start to end in the same direction",
    ],
    artifact="kernel",
    bend_instructions=(
        "Map every object of the world onto a computational object: what is memory, what flows, what stays still, "
        "what is a processor, what is time. Keep the mapping literal - the more literal, the more likely something new "
        "falls out. Then write the kernel in C against alignment_contract, predict its speed, measure it with "
        "alignment_bench, and improve it at most four times. The final answer MUST contain one ```c code block with the "
        "complete kernel and one line 'PREDICTION: speedup_vs_dp = <number>' written BEFORE the first measurement."),
    measure_families=["alignment"],
    measure_tool="alignment_bench",
    examine_question="Does this kernel compute the exact same alignment score as the reference, and is the reported speedup credible? Which assumption of the standard method did it change?",
    known_way="Needleman-Wunsch: O(n^2) DP table, or a banded/SIMD variant (SSW, KSW2) exploiting a bounded score range.",
    assumption_hints={
        "every cell of the comparison depends on the ones above, to the left, and diagonally above-left, computed in that order": (
            "This time, imagine the whole grid as a single sheet, and ask what happens if you fold it - along its "
            "slanting middle, once or twice more - so that a cell far from another is suddenly pressed flush "
            "against it. At the crease, could you read two or three layers at once, cell for cell, bit for bit, "
            "instead of walking the sheet flat one square after the next?"
        ),
        "one pair of positions is judged at a time": (
            "This time, picture a single short body, one cell wide, laid across the grid, free to point any of "
            "eight ways and to curl back on itself - and ask what it could learn about the whole board in one "
            "continuous crawl, rather than by comparing one pair of squares, then the next, then the next."
        ),
        "the whole grid of every position against every other must be filled in": (
            "This time, imagine the grid as a maze holding a few true treasures and much empty corridor, and a "
            "creature in it that only wants to have stood on every treasure once - not to have walked every "
            "corridor. Could knowing where the treasures are excuse you from ever filling in the rest of the grid?"
        ),
    },
)

_NIM = Target(
    name="nim",
    title="Always find a winning Nim move",
    problem="Given several piles of objects, choose a move (remove some objects from one pile) that wins - matching "
           "Bouton's theorem: a position is winning iff the XOR of all pile sizes is nonzero, and the winning move "
           "makes it zero.",
    in_world_need=("Several heaps stand before you. Two take turns removing as much as they like from exactly one "
                   "heap; whoever takes the last object wins. You must always find a move that keeps you winning, "
                   "when one exists."),
    assumptions=[
        "a winning move can only be found by looking ahead through the game's possible futures",
        "each heap must be considered on its own before the others",
        "the value of a position is unknown until every reachable position from it has been examined",
        "a move only affects the one heap it touches",
        "the game must be played out to know who wins",
    ],
    artifact="kernel",
    bend_instructions=(
        "Map every object of the world onto a computational object: what is a heap, what is a move, what is knowing. "
        "Keep the mapping literal. Then write the kernel in C against nim_contract, predict its accuracy, measure it "
        "with nim_bench, and improve it at most four times. The final answer MUST contain one ```c code block with "
        "the complete kernel and one line 'PREDICTION: speedup_vs_dp = <number>' written BEFORE the first measurement "
        "(here 'speedup' still means time; accuracy is reported separately by the tool)."),
    measure_families=["nim"],
    measure_tool="nim_bench",
    examine_question="Does this kernel find a winning move whenever Bouton's theorem says one exists, and is the reported speed credible?",
    known_way="Bouton's theorem (1901): XOR every pile size; if nonzero, some pile can be reduced to make the XOR zero - O(number of piles), no search.",
    assumption_hints={
        "each heap must be considered on its own before the others": (
            "This time, imagine every heap's sticks thrown into the air together, arranging themselves into a "
            "pyramid - the sticks of one heap crossing the sticks of every other heap at once, laid out in shared "
            "rows. A single row holds a sliver from every heap at once, not one heap at a time, and whatever is "
            "true of that shared row is true across every heap simultaneously. Could you learn something about "
            "all the heaps together, row by shared row, rather than by finishing one heap's story before starting "
            "the next?"
        ),
        "a winning move can only be found by looking ahead through the game's possible futures": (
            "This time, imagine each heap as a child holding out as many hands as it has sticks - but every real "
            "child has exactly two hands, so a heap with an odd number of sticks always leaves one hand unpaired, "
            "awkward, alone. An unpaired hand is trouble: the child slaps it away to the next child, who either "
            "finds their own hands paired again or becomes unpaired themselves and must pass the trouble further "
            "on. Could you find the winning move just by settling every child's hands into pairs, right now, "
            "without ever imagining a single future turn?"
        ),
        "the value of a position is unknown until every reachable position from it has been examined": (
            "This time, imagine the heaps as flocks of ducks crossing the sky in triangular formations, and a "
            "hunter below who wants to take the most ducks while every duck wants only to live. Somewhere in each "
            "triangle flies a leader duck, and you cannot tell which one it is just by looking - it may even be "
            "hiding among the others. Could you know whether the hunter wins, and find the leader, without first "
            "tracing out where every possible shot and every possible flight path would lead?"
        ),
    },
)

_HASH = Target(
    name="hash",
    title="Mix bytes into a well-distributed hash, fast",
    problem="Design a hash function over a byte buffer that mixes every input bit into the output well (avalanche: "
           "flipping one input bit should flip about half the output bits) and runs fast.",
    in_world_need=("A pile of marks, in order, must be folded down into one small token such that no two different "
                   "piles ever fold to the same token by accident, and changing even one mark anywhere changes the "
                   "token almost entirely."),
    assumptions=[
        "each byte must be mixed into the running state before the next byte is read",
        "the state is a single accumulator updated in place, one value",
        "mixing one byte requires a multiplication",
        "the whole buffer must be read once, start to end, in order",
        "more mixing rounds always means better mixing",
    ],
    artifact="kernel",
    bend_instructions=(
        "Map every object of the world onto a computational object: what is a byte, what is the state, what is "
        "mixing. Keep the mapping literal. Then write the kernel in C against hash_contract, predict its quality, "
        "measure it with hash_bench, and improve it at most four times. The final answer MUST contain one ```c code "
        "block with the complete kernel and one line 'PREDICTION: speedup_vs_dp = <number>' written BEFORE the first "
        "measurement (here the tool reports throughput and an avalanche score, not a correctness check - there is no "
        "single 'right' hash)."),
    measure_families=["hash"],
    measure_tool="hash_bench",
    examine_question="Does this hash mix well (avalanche near 0.5) and is the reported throughput credible?",
    known_way="FNV-1a / xxHash: multiply-xor-shift mixing rounds folding the whole buffer through one accumulator.",
)

_FFT = Target(
    name="fft",
    title="Transform a signal into its frequencies, faster",
    problem="Compute the discrete Fourier transform of a length-n (power of two) complex sequence, matching a naive "
           "O(n^2) reference, faster.",
    in_world_need=("A shape that changes over time must be re-told as a list of steady, unchanging pure notes, each "
                   "with its own strength, such that all the notes played together reconstruct the original shape."),
    assumptions=[
        "every output depends on every input, computed as one pass",
        "there is no way to reuse work between different output frequencies",
        "the transform must be computed for the whole sequence before any output is known",
        "each output is one independent sum",
        "the input order cannot be rearranged",
    ],
    artifact="kernel",
    bend_instructions=(
        "Map every object of the world onto a computational object: what is a note, what is time, what stays still. "
        "Keep the mapping literal. Then write the kernel in C against fft_contract, predict its speed, measure it "
        "with fft_bench, and improve it at most four times. The final answer MUST contain one ```c code block with "
        "the complete kernel and one line 'PREDICTION: speedup_vs_dp = <number>' written BEFORE the first "
        "measurement."),
    measure_families=["fft"],
    measure_tool="fft_bench",
    examine_question="Does this kernel compute the same transform as the naive DFT reference (within tolerance), and is the reported speedup credible?",
    known_way="FFTW / Cooley-Tukey: recursive radix-2 butterfly network with precomputed twiddle factors and bit-reversal reordering.",
)

_KMER = Target(
    name="kmer",
    title="Count every k-mer in a DNA sequence, faster",
    problem="Count every length-k substring (\"k-mer\") of a DNA sequence over {A,C,G,T} into an exact reference "
           "count table, as fast as possible.",
    in_world_need=("A long strand of four kinds of beads passes before you, and for every short stretch of k beads "
                   "in a row, a running tally of that exact pattern must be kept - so that by the end, every "
                   "pattern's total count is known exactly."),
    assumptions=[
        "each k-mer's code must be built fully from scratch, one base at a time, before it can be counted",
        "one window is examined, then discarded, before the next begins",
        "the count table can only be updated one position's result at a time",
        "every base contributes independently to the window it starts",
        "the whole sequence must be read once, start to end, in order",
    ],
    artifact="kernel",
    bend_instructions=(
        "Map every object of the world onto a computational object: what is a bead, what is a window, what is a "
        "tally. Keep the mapping literal. Then write the kernel in C against kmer_contract, predict its speed, "
        "measure it with kmer_bench, and improve it at most four times. The final answer MUST contain one ```c "
        "code block with the complete kernel and one line 'PREDICTION: speedup_vs_naive = <number>' written "
        "BEFORE the first measurement."),
    measure_families=["kmer"],
    measure_tool="kmer_bench",
    examine_question="Does this kernel's count table match the reference exactly, and is the reported speedup credible?",
    known_way="Naive O(n*k): re-encode the whole k-length window from scratch at every position.",
    assumption_hints={
        "each k-mer's code must be built fully from scratch, one base at a time, before it can be counted": (
            "This time, notice that one window and the next overlap in all but one bead: the one at the back falls "
            "away, and only one new bead joins at the front. Could the tally for a new window be built by adjusting "
            "the previous window's already-known code, rather than reading all k beads over again from nothing?"
        ),
    },
)

_ALL = {t.name: t for t in (_MATMUL, _PHYSICS, _MECHANICS, _ALIGNMENT, _NIM, _HASH, _FFT, _KMER)}
TARGETS = list(_ALL)


def get_target(name: str) -> Target:
    try:
        return _ALL[name]
    except KeyError:
        raise KeyError(f"unknown target {name!r}; known: {TARGETS}") from None
