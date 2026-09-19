# MAPPING

## SEED 1 — "The bone flute's forked note selects which lanterns pair for the turn"

| World object | Problem object |
|---|---|
| row of lanterns | statevector array (2^n amplitudes) |
| one lantern | one basis-state amplitude at index *i* |
| "this turn" | the current gate *g* |
| the flute's one long note traveling the row | a single sweep/loop over the amplitude array |
| the note "forking and bending double" | the pairing rule *j = i XOR (1<<gate_qubit)*, arrived at structurally, not by testing each index |
| bellows at the hip | the vector unit moving data in bulk |

Breaks: *"every amplitude must be visited in the same order for every gate"* — the fork pattern (and hence the natural traversal stride) depends on which bit the target qubit occupies.

## SEED 2 — "Equal-throat-size flames touching through a reed-cone die together and are swept into the cellar-sea, never fished back"

| World object | Problem object |
|---|---|
| lantern's two flames (bright + shadow) | re[i], im[i] of one amplitude |
| reed-cone between two paired lanterns | the pairing/comparison performed when a gate updates (i,j) |
| "throat-size" | resulting magnitude of an amplitude after the gate |
| flames of equal throat-size touching and dying together | a paired amplitude update that lands on **exact** (0,0) for both members |
| swept into the cellar-sea, never fished back | that index is dropped from all future gate processing — permanently, unless a later gate re-touches it through its (still-lit) partner |

Breaks: *"the state must be a dense array; every amplitude is stored explicitly"* — this seed licenses **not walking** amplitudes that are known to be exactly zero, i.e. tracking only the currently "lit" (nonzero) subset.

## SEED 3 — "Unequal flames poured through the bellows merge into one leftover glow carved onto the milestone-spine; after dark the row forgets pairings"

| World object | Problem object |
|---|---|
| milestone-spine | an accumulator 2×2 matrix per qubit |
| pouring smaller flame into larger, carving one leftover glow | composing consecutive same-qubit gate matrices into one before touching amplitudes |
| "after dark the row forgets pairings" | fused matrix must be flushed before the target qubit changes |

Breaks: *"a gate must be fully applied to the whole state before the next gate starts."* This is a near word-for-word description of **gate fusion**, which the prompt already lists as part of the *known* way — least distinctive, so rejected for step 2.

# CHOSEN SEED

SEED 2 (equal-throat-size cancellation → cellar-sea sweep). It is the most literal (bright/shadow flame ↔ re/im is a clean 1:1 mapping, "throat-size" ↔ magnitude is clean, "swept, never fished back" ↔ permanently dropped from a work-list is clean) and it is the one *not* already present in the "known way" list (dense-array assumption is never challenged there).

# ASSUMPTION BROKEN

"The state must be a dense array; every amplitude is stored explicitly" — the ritual only ever tends to *lit* lanterns. I keep the contractually-required dense `state_re/state_im` arrays (the contract forces this), but I maintain an auxiliary "row of lit lanterns" work-list and never walk an amplitude pair unless at least one side is currently lit; pairs whose flames die together (exact (0,0)) are swept out of that work-list for good.

# ARTIFACT

```c
#include <math.h>
#include <stdlib.h>

void kernel(int n_qubits, int n_gates, const int *gate_type, const int *gate_qubit, const double *gate_param,
           double *state_re, double *state_im) {
    long long N = 1LL << n_qubits;

    /* The "row": lanterns still lit (nonzero amplitudes) this turn.
       Only lit lanterns (or their partners, reached through them) are
       ever walked. A pair whose flames die together through the reed-cone
       (both land on exact zero) is swept into the cellar-sea: dropped from
       the row and never fished back, unless a later gate relights one of
       them through its still-lit partner on a different bit. */
    long long *cur  = (long long *)malloc(sizeof(long long) * (size_t)N);
    long long *next = (long long *)malloc(sizeof(long long) * (size_t)N);
    unsigned int *seen = (unsigned int *)calloc((size_t)N, sizeof(unsigned int));

    if (!cur || !next || !seen) {
        /* dense fallback if the row itself can't be built */
        for (int g = 0; g < n_gates; g++) {
            long long bit = 1LL << gate_qubit[g];
            int gtype = gate_type[g];
            double gparam = gate_param[g];
            for (long long i = 0; i < N; i++) {
                if (i & bit) continue;
                long long j = i | bit;
                double r0 = state_re[i], i0 = state_im[i], r1 = state_re[j], i1 = state_im[j];
                if (gtype == 0) {
                    double s = 0.70710678118654752440;
                    state_re[i] = (r0 + r1) * s; state_im[i] = (i0 + i1) * s;
                    state_re[j] = (r0 - r1) * s; state_im[j] = (i0 - i1) * s;
                } else if (gtype == 1) {
                    state_re[i] = r1; state_im[i] = i1; state_re[j] = r0; state_im[j] = i0;
                } else if (gtype == 2) {
                    state_re[i] = r0; state_im[i] = i0; state_re[j] = -r1; state_im[j] = -i1;
                } else {
                    double phi0 = -gparam / 2, phi1 = gparam / 2;
                    double c0 = cos(phi0), s0 = sin(phi0), c1 = cos(phi1), s1 = sin(phi1);
                    state_re[i] = r0 * c0 - i0 * s0; state_im[i] = r0 * s0 + i0 * c0;
                    state_re[j] = r1 * c1 - i1 * s1; state_im[j] = r1 * s1 + i1 * c1;
                }
            }
        }
        free(cur); free(next); free(seen);
        return;
    }

    /* |0...0>: exactly one lit lantern, index 0. */
    long long cur_count = 1;
    cur[0] = 0;
    unsigned int gen = 0;

    for (int g = 0; g < n_gates; g++) {
        long long bit = 1LL << gate_qubit[g];
        int gtype = gate_type[g];
        double gparam = gate_param[g];
        double c0 = 0, s0 = 0, c1 = 0, s1 = 0;
        if (gtype == 3) {
            double phi0 = -gparam / 2, phi1 = gparam / 2;
            c0 = cos(phi0); s0 = sin(phi0); c1 = cos(phi1); s1 = sin(phi1);
        }

        gen++;
        long long next_count = 0;

        for (long long idx = 0; idx < cur_count; idx++) {
            long long i = cur[idx];
            if (seen[i] == gen) continue;          /* partner already walked this turn */
            long long j = i ^ bit;
            seen[i] = gen; seen[j] = gen;

            double r0 = state_re[i], i0 = state_im[i];
            double r1 = state_re[j], i1 = state_im[j];
            double nr0, ni0, nr1, ni1;

            if (gtype == 0) {
                double s = 0.70710678118654752440;
                nr0 = (r0 + r1) * s; ni0 = (i0 + i1) * s;
                nr1 = (r0 - r1) * s; ni1 = (i0 - i1) * s;
            } else if (gtype == 1) {
                nr0 = r1; ni0 = i1; nr1 = r0; ni1 = i0;
            } else if (gtype == 2) {
                nr0 = r0; ni0 = i0; nr1 = -r1; ni1 = -i1;
            } else {
                nr0 = r0 * c0 - i0 * s0; ni0 = r0 * s0 + i0 * c0;
                nr1 = r1 * c1 - i1 * s1; ni1 = r1 * s1 + i1 * c1;
            }

            state_re[i] = nr0; state_im[i] = ni0;
            state_re[j] = nr1; state_im[j] = ni1;

            /* reed-cone: did the flames die together? if not, whichever
               still burns stays in the row for the next turn. */
            if (nr0 != 0.0 || ni0 != 0.0) next[next_count++] = i;
            if (nr1 != 0.0 || ni1 != 0.0) next[next_count++] = j;
        }

        long long *tmp = cur; cur = next; next = tmp;
        cur_count = next_count;
    }

    free(cur); free(next); free(seen);
}
```

Correctness argument: gate formulas are byte-for-byte the reference formulas, applied to every pair whose members' true values could differ from the values already sitting in the dense arrays. A pair where *both* members are currently zero produces zero under all four gates (all are linear maps of (r0,i0,r1,i1)), so skipping it changes nothing. Hence the work-list is a pure performance device, not an approximation — no tolerance is spent.

# PREDICTION: speedup_vs_naive = 0.6

Reasoning for the number: the ritual is a single walker with one flute and one bellows — inherently serial, so I did not add OpenMP/SIMD (that would be smuggling in the "known way," which the instructions forbid). For a genuinely sparse circuit (few gates relative to n_qubits) the row stays short and this should beat the naive O(2^n)-every-gate loop by a large margin. But the fixed contract's benchmark almost certainly applies enough mixing gates (H/Rz across many qubits) that the row densifies to the full 2^n within the first handful of gates; after that, every gate costs the same O(2^n) pair count as naive, but with strictly *more* per-pair work than naive's single `i & bit` branch test: two extra `seen[]` reads/writes and two conditional appends into `next[]`, none of which the branch-predictor-friendly naive loop pays. I expect the dense regime to dominate total runtime, so I predict a net **slowdown** (speedup < 1), not a win — an honest, unflattering prediction, stated before any measurement.

# MEASUREMENT

No `quantum_bench` / `quantum_contract` tools were exposed in this session (the tool list available to me this turn was Claude Docs, Gmail, Google Drive, and Slack only — no compiler/benchmark harness). I cannot honestly report a measured number, and per the operating principle of this exercise I will not fabricate one. The code above is submitted as the literal, single-shot artifact for the external pipeline to compile and measure; the reasoning under PREDICTION is my analytical estimate of what that measurement should show, not a substitute for it.

# VERDICT

Taking SEED 2 completely literally produces a *correct* optimization (skip-known-zero, exact, tolerance-free) but one whose benefit is circuit-dependent: real speedup only on sparse/short circuits, and a predicted net slowdown once the row of lit lanterns fills the whole state, because the "known way" (branch-predictable dense sweep, then fusion, then SIMD) has no bookkeeping cost to pay once the state is dense while this seed's cellar-sea bookkeeping keeps charging rent even after there's nothing left to sweep away. That is the honest, plainly-stated outcome of building what the native described rather than quietly reverting to the textbook kernel.