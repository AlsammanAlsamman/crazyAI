# MAPPING

**SEED 1 — "The ribs inside the gold are counted in a fixed order to give every leaf its position."**

| World object | Problem object |
|---|---|
| Gold statue, ribs from trunk to lotus petal | The m = log₂(n) fixed transform stages (half-widths h = 1,2,4,…,n/2) |
| Ribs "never move," counted once, fixed order | The stage order is fixed and stage-independent (bit b's butterfly doesn't depend on processing order of other bits) |
| "one rib, one place in the pattern" | Each stage owns exactly one bit position of the index |

Assumption broken: *"the input order cannot be rearranged"* — the statue fixes a structural order (by bit/stage) instead of by raw index value.

**SEED 2 — "I walk the straight path flipping one leaf... until every leaf has stood lit and unlit in combination with every other leaf exactly once."**

| World object | Problem object |
|---|---|
| Straight garden path | The linear data buffer |
| m leaves, one per rib | The m bits that select a butterfly pair at a given stage |
| Flip exactly one leaf per step | Move from one index to its butterfly partner by toggling exactly one bit |
| Every combination visited once, no repeats | Every (block, offset) pair at a stage is touched exactly once — no output recomputed, no pair skipped |
| Black = full cycle finished | Stage loop naturally terminates when h reaches n |

Assumption broken: *"there is no way to reuse work between different outputs"* — neighbors reached by a single flip share almost everything; only the differing bit needs new arithmetic.

**SEED 3 — "The jewels pass borrowed light hand to hand... until one comes up dark, telling me the choosing is done."**

| World object | Problem object |
|---|---|
| Jewels along the path | Independent blocks at a given stage |
| Light passed hand to hand | Work handed off between parallel workers |
| Dark jewel = stop | End-of-range test that closes one block/thread's work |

Assumption broken: *"combining two contributions always means an ordinary addition"* — here "combining" is control-flow bookkeeping (who does what range), not arithmetic.

# CHOSEN SEED

SEED 2. It is the most literal ("flip exactly one leaf, visit every combination exactly once, reuse the previous state") and the most different in shape from the stated known way: instead of a top-down recursive split, it describes a bottom-up sweep organized stage-by-stage (rib-by-rib) over pairs that differ in one bit — the iterative, non-recursive form of the butterfly.

# ASSUMPTION BROKEN

"There is no way to reuse work between different outputs." Two indices that are butterfly partners (differ in exactly one bit at the current stage) let `out[j]` and `out[j+h]` be produced together from the *same* pair `(a,b)` via `a+b` / `a-b` — one read of each leaf, two outputs, no independent O(n) sum per output.

# ARTIFACT

Literal object map: gold statue/ribs → the m fixed stages (h); garden path → the `out` buffer; leaves → slots `j` and `j+h`; "lit/dark, never half-lit" → the ±1 stripe engraved as `a+b`/`a-b`, written once and never revisited ("steady, needing no more moonlight"); jewels handing light along the path → the independent blocks distributed to OpenMP threads at each stage.

```c
#include <omp.h>

void kernel(int n, const double *in, double *out) {
    /* copy the garden: leaves start exactly as the input, in the straight path's order */
    for (int i = 0; i < n; i++) out[i] = in[i];

    /* walk the ribs, fixed order, trunk to lotus petal: h = 1, 2, 4, ..., n/2 */
    for (int h = 1; h < n; h <<= 1) {
        int step = h << 1;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i += step) {
            for (int j = i; j < i + h; j++) {
                double a = out[j];
                double b = out[j + h];
                out[j]     = a + b;   /* leaf raised: this rib's stripe, +1 branch */
                out[j + h] = a - b;   /* leaf flat:   this rib's stripe, -1 branch */
            }
        }
    }
}
```

This is the standard iterative Fast Walsh–Hadamard Transform: log₂(n) stages, each O(n) work, total O(n log n), pure add/subtract, in place.

# PREDICTION

The reference `kernel` is worse than plain O(n²): its inner `popcount` is a bit-loop, not a single instruction, so the reference is really O(n² log n). Ours is O(n log n). The asymptotic ratio is roughly n·log n divided by whatever constant-factor overhead OpenMP thread dispatch and memory bandwidth add at the sizes `wht_bench` uses. I expect a large but not fully "theoretical" speedup once parallel launch overhead and cache effects are accounted for.

PREDICTION: speedup_vs_naive = 300

# MEASUREMENT

Not performed in this session — no tools (including `wht_bench`) were available here; per the task's explicit override, only the reasoning, prediction, and artifact were to be produced, and the pipeline will compile and measure the kernel above externally.

# VERDICT

Pending the pipeline's actual `wht_bench` run against the prediction of ≈300×. The kernel is a straightforward, literal translation of SEED 2 (single-bit-flip pairing → shared add/subtract instead of independent per-output sums) and is algorithmically equivalent to the recursive butterfly ("known way"), just expressed iteratively/bottom-up as the seed describes, so I expect it to pass correctness trivially; the only open question the measurement will settle is how close real hardware gets to the O(n log n) vs O(n² log n) asymptotic gap — if the pipeline's `n` is small, OpenMP overhead could pull the observed number well below 300, and that would be reported honestly rather than adjusted after the fact.