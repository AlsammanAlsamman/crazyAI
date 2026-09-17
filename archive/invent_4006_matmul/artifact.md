## MAPPING

For each seed, the world→problem correspondence, and the standard assumption it targets:

**SEED 1 — "Birds that turn mid-flight over a crossing perform the multiplication."**

| World object | Problem object |
|---|---|
| flock of grey birds along the shared line | one FMA instance per k, for a fixed (i,j) |
| grain-value / seed-value | A[i,k] / B[k,j] |
| the turning | the multiply-add itself, happening at one instant |
| blue fire | a liveness flag: this (i,j) accumulation still open |

Breaks: nothing new — it is a literal restatement of "every product is computed exactly, once," just insisting the product exists only *at* the FMA instant. Useful as vocabulary, not as a design change. Rejected as *not different enough* from the known way (this is what every microkernel already does).

**SEED 2 — "A running sum is held safe from the power-cutting wind by hair wrapped around a white stone."**

| World object | Problem object |
|---|---|
| white stone | the accumulator value for C[i,j] |
| hair wrapped tighter with every drop | a compensation/carry term added alongside the sum (Kahan-style), so low bits aren't "spilled" |
| power-cutting wind | loss of the partial sum before it's "set" (register spill / interruption) |

Breaks: implicitly, "numbers are IEEE doubles and multiply is the primitive" — it asks for a *protected*, error-compensated running sum rather than a bare double. This is a real, literal idea (compensated summation) but it trades speed for accuracy, and the contract already tolerates "stated error," so it buys us nothing on the speed axis. Rejected as orthogonal to the speed goal.

**SEED 3 — "A sum is only trusted once two independently-computed workers wake and insist, as the vowed couple, that they arrived at the same number."**

| World object | Problem object |
|---|---|
| the two great tables | matrices A (i×k) and B (k×j), each held still in memory |
| the couple | two independent hardware thread-groups |
| "compute the sum twice by two separate paths down the shared index" | traverse k in two different orders (forward vs. backward), giving two independently-rounded sums for the same C[i,j] |
| "wake up and insist they are the same couple" | compare the two results within a stated tolerance |
| "if they wake as strangers, burn the crossing again from scratch" | on mismatch, recompute deterministically and trust that instead |
| ash, exhausted birds, empty hair-baskets — thrown away | the scratch buffer is freed once verification is done |

Breaks: **"every product is computed exactly, once"** — directly and literally. Every cell's sum (hence every product feeding it) is computed *twice*, by two structurally different paths, and is only allowed into C after the two paths agree. Nothing in OpenBLAS, MKL, or the cache-blocked reference does this — they trust a single silent computation. This is both the most literal reading (the vow/verification is not a metaphor for anything else — it *is* a redundant computation with an equality check) and the most different from the known way.

## CHOSEN SEED
SEED 3 — the vow-matched couple.

## ASSUMPTION BROKEN
"Every product is computed exactly, once." Here every output cell is produced by two independent thread-groups traversing the shared index k in opposite orders, and is committed to C only if both groups' sums agree to within a stated tolerance; otherwise it is recomputed once, deterministically, as the trusted fallback.

## ARTIFACT

Literal object map used to write the kernel: A and B are the two "great tables," staying still in memory; the "shared line" is the k-loop; "processors" are OpenMP threads, split into two groups (Group A = forward-k, Group B = reverse-k) that run *concurrently*, each with half the available cores; "time" is wall-clock spent by whichever group finishes last; the "hair-wrapped stone" for each C[i,j] is just `Crow[j]` in ordinary memory (SEED 2 not adopted, so no compensation term); the "acknowledgment/bell-note" is the per-row tolerance check; a mismatch "burns the crossing again from scratch" via one clean full pass.

```c
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* ---- Path A: forward order down the shared index k ---- */
static void compute_forward(int n, const double *A, const double *B, double *C,
                             int row_lo, int row_hi) {
    const int BK = 256;
    for (int i = row_lo; i < row_hi; i++) {
        double *Crow = C + (size_t)i * n;
        memset(Crow, 0, (size_t)n * sizeof(double));
        for (int k0 = 0; k0 < n; k0 += BK) {
            int k1 = (k0 + BK < n) ? (k0 + BK) : n;
            for (int k = k0; k < k1; k++) {
                double a = A[(size_t)i * n + k];
                const double *Brow = B + (size_t)k * n;
                for (int j = 0; j < n; j++)
                    Crow[j] += a * Brow[j];
            }
        }
    }
}

/* ---- Path B: an independent worker, reading the shared index backward ---- */
static void compute_reverse(int n, const double *A, const double *B, double *C,
                             int row_lo, int row_hi) {
    const int BK = 256;
    for (int i = row_lo; i < row_hi; i++) {
        double *Crow = C + (size_t)i * n;
        memset(Crow, 0, (size_t)n * sizeof(double));
        for (int k1 = n; k1 > 0; k1 -= BK) {
            int k0 = (k1 - BK > 0) ? (k1 - BK) : 0;
            for (int k = k1 - 1; k >= k0; k--) {
                double a = A[(size_t)i * n + k];
                const double *Brow = B + (size_t)k * n;
                for (int j = 0; j < n; j++)
                    Crow[j] += a * Brow[j];
            }
        }
    }
}

void kernel(int n, const double *A, const double *B, double *C) {
    if (n <= 0) return;

    double *CB = (double *)malloc((size_t)n * (size_t)n * sizeof(double));

    if (!CB) {
        /* no room for a second worker: compute once, trust it, move on */
        int T = omp_get_max_threads();
        #pragma omp parallel for schedule(static) num_threads(T)
        for (int i = 0; i < n; i++)
            compute_forward(n, A, B, C, i, i + 1);
        return;
    }

    int T = omp_get_max_threads();
    if (T < 2) T = 2;
    int half = T / 2;
    int otherhalf = T - half;

    /* the couple, working concurrently down the shared index, each on half the crossings' cores */
    #pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
        if (tid < half) {
            for (int i = tid; i < n; i += half)
                compute_forward(n, A, B, C, i, i + 1);
        } else {
            int t2 = tid - half;
            for (int i = t2; i < n; i += otherhalf)
                compute_reverse(n, A, B, CB, i, i + 1);
        }
    }

    /* the couple wakes: do they insist on the same number, row by row? */
    int mismatches = 0;
    #pragma omp parallel for schedule(static) reduction(+:mismatches)
    for (int i = 0; i < n; i++) {
        const double *rc = C + (size_t)i * n;
        const double *rb = CB + (size_t)i * n;
        double rowdiff = 0.0, rowmax = 0.0;
        for (int j = 0; j < n; j++) {
            double d = fabs(rc[j] - rb[j]);
            double m = fabs(rc[j]);
            double m2 = fabs(rb[j]);
            if (m2 > m) m = m2;
            if (d > rowdiff) rowdiff = d;
            if (m > rowmax) rowmax = m;
        }
        if (rowdiff > 1e-9 * (rowmax + 1.0)) mismatches++;
    }

    if (mismatches) {
        /* strangers, not the vowed couple: burn the crossing down, redo once, trusted */
        #pragma omp parallel for schedule(static) num_threads(T)
        for (int i = 0; i < n; i++)
            compute_forward(n, A, B, C, i, i + 1);
    }

    free(CB);
}
```

## PREDICTION
Derivation: each of the two groups performs the *full* O(n³) work but with only T/2 cores each (T = hardware threads), so each group's wall-time is ≈ 2× what a single all-core pass would take; the groups run concurrently, so total wall-time ≈ 2×(single-threaded blocked time)/T. Against a single-threaded cache-blocked reference this gives an expected speedup of roughly T/2. Assuming a typical multi-core evaluation box (~7–8 threads), I expect a genuine but *halved* win relative to what an honest single-path, all-core, cache-blocked-and-vectorized kernel would get — i.e., this design pays a real, literal price (2×) for the vow-verification, and should land well short of OpenBLAS, and possibly short of a fully-parallel un-verified blocked loop.

PREDICTION: speedup_vs_blocked = 3.5

## MEASUREMENT
Not measured in this session — the `kernel_bench`/`kernel_contract` tools were not available to me here (only unrelated Gmail/Slack/Drive/Docs tools were exposed). The prediction above stands as the pre-registered number; it must be checked against the actual harness before any claim of success is made. I am not reporting a measured result because none was taken — reporting one would be fabrication.

## VERDICT
Unverified. The artifact is a faithful, literal instantiation of SEED 3 (real redundant computation via two independently-ordered traversals of the shared index, with a genuine tolerance-based acceptance/rejection and a deterministic fallback on mismatch) rather than a relabeled cache-blocked loop — the 2× redundancy is real, paid-for work, not decoration. My own derivation says this should be *slower* than an honest single-path parallel kernel using all cores, and likely far from OpenBLAS; if the harness's "cache-blocked triple loop" baseline turns out to already be multi-threaded, the predicted 3.5× speedup is almost certainly too optimistic and should be revised down (possibly below 1×) once actually measured — that outcome would be an honest negative result for this seed, not a bug to hide.