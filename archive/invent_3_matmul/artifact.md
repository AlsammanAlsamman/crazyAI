# MAPPING (all three seeds)

| SEED | World-object | Problem-object | Silent assumption it breaks |
|---|---|---|---|
| 1. Fixed post (boy in the oak) | The boy, forever motionless in the crown | The row index `i`, held fixed for an entire sweep | The root he sits on | The shared/contraction index `k`, walked while `i` is pinned | "one cell at a time, row by row" — instead of finishing `C[i][j]` before moving on, you finish one `(i,k)` pair across *all* `j` before advancing `k` |
| 2. Covered cart | Two streets/cities | Two disjoint memory domains (a CPU register vs. shared/aliasable memory; or thread A's cache lines vs. thread B's) | The dark, covered cart | A CPU register (or a thread-private scratch buffer) carrying a value across that boundary | The "unmarked power" that seizes an uncovered crossing | Cache-coherence traffic (MESI snooping / false sharing) that stalls a core when a partially-written shared cache line is touched by another core | "one processor holds both matrices" / "a matrix lives in one memory" — breaks it by insisting cross-domain transfers happen through private, non-shared carriers |
| 3. Cairn at the quern | The quern | The FMA (multiply-accumulate) unit | The cairn | A per-`(i,j)` accumulator kept in a register, not in the shared `C` array, until fully summed | "read total weight, write to third table, throw stones in river" | Commit to memory **once**, after the whole `k`-loop, then discard the register | "the whole sum over the shared index is finished before the next cell is started" is inverted in the *given* reference kernel, which does an incremental read-modify-write of `C[i][j]` in memory on every single `k` — SEED 3 forbids touching the real `C` at all until every product for that cell has landed |

# CHOSEN SEED

**SEED 2 — the covered cart.** It is the most literal *and* the most different from the known way: OpenBLAS's packing already moves data around, but it never frames the *reason* as "an uncovered value gets seized." Taken literally, this seed says something OpenBLAS's documentation doesn't say out loud: a value in flight between two memory domains must never be visible, even momentarily, to another actor (core) unless it is safely walled off — otherwise you pay a coherence-traffic tax ("seizure"). The most literal computational object that is genuinely private, unglanced, and unswervable by any other core is a **CPU register**. So the seed's instruction is: *never let a partial product or partial sum touch shared/aliasable memory — carry it in registers, end to end, and only expose the finished stone to the shared table once.*

# ASSUMPTION BROKEN

"One processor holds both matrices" and "a matrix is a two-dimensional grid living in one memory" — broken by treating `A`, `B`, and `C` as three separate "cities," with every value that crosses between them required to travel in a private carrier (register) rather than as a raw shared-memory reference, and by giving each OpenMP thread its own disjoint, cache-line-safe territory of `C` so no cross-core "seizure" (false sharing / coherence stalls) can occur.

# ARTIFACT — literal object mapping

- **Oak / fixed post (SEED 1, supporting)**: row index `i`, pinned for a 4-row register tile; `A[i][k]` never reloaded from memory more than once per `(i,k)`.
- **Two streets/cities (SEED 2)**: `B`'s column-panel (its own "street") vs. the 4×4 output tile living in registers (a different "city"); the crossing between them is a `_mm256_set1_pd` broadcast (the boulder) and an FMA (the quern) — both pure register ops, never touching shared memory mid-flight.
- **Covered cart**: the `__m256d` accumulator registers `c0..c3` — genuinely private per core, invisible to any other thread, so no coherence "seizure" is possible while they're in flight.
- **Cairn (SEED 3, supporting)**: the same `c0..c3` registers, accumulated over the *entire* `k` range before a single store to `C` is issued.
- **Rich man's stop**: the register file itself — the one "place" values are staged before being fired.
- **River**: nothing is retained after the store; registers are simply reused for the next tile.
- **Thread territory**: OpenMP splits row-panels (`ii` blocks of 4 rows) across cores — disjoint row ranges of `C`, so no two cores ever write the same cache line.

```c
#include <immintrin.h>

#define MR 4
#define NR 4

/* The cairn: a 4x4 tile accumulated entirely in registers (the covered
   cart) across the full k range, and written to the shared table C
   exactly once, after every boulder meant for it has arrived. */
static inline void microkernel_4x4(int n, int i0, int j0,
                                    const double *A, const double *B,
                                    double *C)
{
    __m256d c0 = _mm256_setzero_pd();
    __m256d c1 = _mm256_setzero_pd();
    __m256d c2 = _mm256_setzero_pd();
    __m256d c3 = _mm256_setzero_pd();

    const double *a0 = A + (size_t)(i0 + 0) * n;
    const double *a1 = A + (size_t)(i0 + 1) * n;
    const double *a2 = A + (size_t)(i0 + 2) * n;
    const double *a3 = A + (size_t)(i0 + 3) * n;

    for (int k = 0; k < n; k++) {
        /* the second table's street: a row of B, read once */
        __m256d b = _mm256_loadu_pd(B + (size_t)k * n + j0);

        /* the boulder: A[i,k], fired across (broadcast) into the cart */
        __m256d va0 = _mm256_set1_pd(a0[k]);
        __m256d va1 = _mm256_set1_pd(a1[k]);
        __m256d va2 = _mm256_set1_pd(a2[k]);
        __m256d va3 = _mm256_set1_pd(a3[k]);

        /* ground on the quern, stacked onto the cairn (register) */
        c0 = _mm256_fmadd_pd(va0, b, c0);
        c1 = _mm256_fmadd_pd(va1, b, c1);
        c2 = _mm256_fmadd_pd(va2, b, c2);
        c3 = _mm256_fmadd_pd(va3, b, c3);
    }

    /* the cairn is read and written into the third table exactly once */
    _mm256_storeu_pd(C + (size_t)(i0 + 0) * n + j0, c0);
    _mm256_storeu_pd(C + (size_t)(i0 + 1) * n + j0, c1);
    _mm256_storeu_pd(C + (size_t)(i0 + 2) * n + j0, c2);
    _mm256_storeu_pd(C + (size_t)(i0 + 3) * n + j0, c3);
}

/* scalar fallback for edge tiles (n not a multiple of MR/NR) */
static inline void microkernel_scalar(int n, int i0, int i1, int j0, int j1,
                                       const double *A, const double *B,
                                       double *C)
{
    for (int i = i0; i < i1; i++) {
        const double *arow = A + (size_t)i * n;
        for (int j = j0; j < j1; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += arow[k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = sum;
        }
    }
}

void kernel(int n, const double *A, const double *B, double *C)
{
    const int NB = 256; /* column-panel width: keeps a B panel resident
                            in cache and reused across a thread's row tiles */

    int mfull = (n / MR) * MR;

    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < mfull; ii += MR) {
        for (int jj = 0; jj < n; jj += NB) {
            int jend  = (jj + NB < n) ? jj + NB : n;
            int jfull = jj + ((jend - jj) / NR) * NR;
            int j;
            for (j = jj; j < jfull; j += NR)
                microkernel_4x4(n, ii, j, A, B, C);
            if (j < jend)
                microkernel_scalar(n, ii, ii + MR, j, jend, A, B, C);
        }
    }

    /* leftover rows if n is not a multiple of MR */
    if (mfull < n)
        microkernel_scalar(n, mfull, n, 0, n, A, B, C);
}
```

Correctness note: this computes the exact same summation order over `k` as the reference for each `(i,j)` cell, so results match to floating-point rounding; FMA fuses multiply+add into a single rounding step (one rounding instead of two per term), so the result is at least as accurate as the reference, not less.

PREDICTION: speedup_vs_blocked = 10

Reasoning behind the number (stated honestly, before any measurement): AVX2 gives a nominal 4x over scalar doubles, FMA saves roughly another 1.5–2x over separate mul+add on the inner loop, and eliminating the repeated read-modify-write of `C[i][j]` on every `k` iteration (which the naive/blocked reference pays, since it can't prove `A/B/C` don't alias) removes real memory traffic. Multiplied by modest multi-core scaling (this design has no `k`/`m` cache packing beyond a single `NB=256` column panel, so it will not scale as cleanly as OpenBLAS and will likely become bandwidth-limited on many cores). Net honest guess: ~10x over a cache-blocked scalar triple loop, well short of a fully packed multi-threaded OpenBLAS.

# MEASUREMENT

Not executed in this session — no `kernel_bench` tool is available here; per the task instructions this artifact and prediction are handed to the downstream pipeline, which will compile it (`gcc -O3 -march=native -fopenmp -lm`) and run the actual benchmark. No fabricated numbers are reported.

# VERDICT

Pending the pipeline's real measurement against the cache-blocked triple loop. The design is a faithful, literal reading of the "covered cart" seed — every partial product and partial sum lives only in CPU registers (the one truly private, un-seizable memory in the machine) until the whole `k`-sum for a tile is complete, at which point it is written to the shared `C` table exactly once. The known risk, stated plainly: because the "cart" (register tile) is only 4×4 and the cache blocking is limited to a single column-panel width, this kernel will very likely fall short of OpenBLAS (which additionally packs panels for both `A` and `B` and uses larger, hand-tuned microkernels), and its advantage over the naive blocked loop could shrink for very large `n` where memory bandwidth, not register reuse, becomes the bottleneck.