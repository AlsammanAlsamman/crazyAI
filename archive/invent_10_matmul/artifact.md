# MAPPING

For each seed: world‑object → problem‑object, and which of the eight silent assumptions it breaks.

**SEED 1 — "Pebble‑and‑vein crossings ring their product the instant two strung strands touch."**

| World object | Problem object |
|---|---|
| iron pebble | one `double` (an element of A or B) |
| strand, one per row | row *i* of A — a contiguous run of *n* doubles |
| vein, one per column | column *j* of B, **re‑cast as a contiguous run** by transposing B into `Bt` (a vein only behaves like a strand — something you can "comb" in one pass — once it is stored contiguously) |
| crossing (strand meets vein) | the output cell (i,j): the dot product A[i,:]·B[:,j] |
| "touching pebbles ring the instant they touch" | one SIMD‑lane multiply/FMA — several products exist in the same instruction |
| "the whole lattice is built before a single note sounds" | `Bt` is fully materialized (and A already resident) *before* any arithmetic starts — no data is discovered on the fly |
| "every crossing rings at once" | SIMD lanes *and* OpenMP threads evaluate many (i,j,k) triples concurrently, not one multiply‑add per scalar‑loop tick |

Breaks assumption **#8** ("one product is one problem; many products are many problems") and, by extension, **#1** ("output produced one cell at a time, row by row") — many products, and even many cells across threads, are treated as one simultaneous act.

**SEED 2 — "A listening‑spine sums every ring… and falls silent when the total is complete."**

| World object | Problem object |
|---|---|
| listening spine | the FMA accumulator (a vector register) attached to one (i,j) cell |
| "ring after ring in order" | successive FMAs along k feeding the same accumulator |
| "remembers everything until read and wiped clean" | the register persists across the k‑loop, then is reset for the next cell |
| "falls silent, no clock, no count kept" | **not honored literally** — real hardware/C needs a trip count; I keep an explicit k‑bound for correctness. This seed is implemented as the *mechanism* inside SEED 1's cells rather than as an independent break. |

Reinforces **#2** (the sum genuinely does finish before the cell is stored — but the *schedule* around it is no longer "cell 0 fully done, then cell 1"; many spines exist concurrently).

**SEED 3 — "A crosswise verifying comb re‑plucks each finished pebble… restring that one row alone."**

| World object | Problem object |
|---|---|
| second thinner comb at half‑tension | a cheap alternate recomputation of a finished cell (different summation order/width) |
| "sing back the same pitch" | numeric agreement within the stated error tolerance |
| "restring that one row alone" | selective recompute of just the offending row, not the whole matrix |
| "strands stripped and cast to loam" | scratch buffers (like `Bt`) freed after use, not retained |

Breaks **#4** ("every product is computed exactly, once") — it explicitly budgets for computing a cell twice when it doesn't trust the run. **Not implemented**: our kernel uses plain deterministic FMA dot products with no reordering, precision‑dropping, or scheduling nondeterminism to distrust, so a verify‑and‑restring pass would add pure overhead with nothing to catch. Noted as the natural next feature if a lower‑precision or reassociated variant is tried later.

# CHOSEN SEED

**SEED 1** — most literal (a "vein" only becomes strand‑like, i.e. combable, once B is transposed into contiguous columns) and most different from the known way: OpenBLAS/blocked‑ikj reuse data across many cells via panel packing and register tiling; SEED 1 instead says *build the whole crossed lattice first, then let every crossing ring at once* — i.e., transpose once, then treat every (i,j) dot product as an independent, fully‑vectorized, thread‑parallel "chord."

# ASSUMPTION BROKEN

\#8 ("one product is one problem; many products are many problems") and \#1 ("output produced one cell at a time, row by row") — replaced by: many products per cell ring together in one FMA instruction (SIMD chord), and many cells across many rows ring together across threads (OpenMP), instead of a strictly sequential scalar walk. \#6 ("one processor holds both matrices") is also broken — many threads hold read‑only access to the same A/Bt lattice concurrently.

# ARTIFACT

```c
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>

void kernel(int n, const double *A, const double *B, double *C) {
    size_t nn = (size_t)n * (size_t)n;

    /* Hang B as vertical veins: transpose it into a contiguous buffer
       so that column j of B becomes contiguous row j of Bt -- a vein
       only becomes something you can comb once it is strung like a
       strand. Built once, entirely, before any crossing rings. */
    size_t bytes = nn * sizeof(double);
    size_t bytes_aligned = ((bytes + 63) / 64) * 64;
    if (bytes_aligned == 0) bytes_aligned = 64;
    double *Bt = (double *)aligned_alloc(64, bytes_aligned);

    if (!Bt) {
        /* fallback: degrade to the reference algorithm, never fail */
        memset(C, 0, nn * sizeof(double));
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++) {
                double a = A[(size_t)i * n + k];
                for (int j = 0; j < n; j++)
                    C[(size_t)i * n + j] += a * B[(size_t)k * n + j];
            }
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            Bt[(size_t)j * n + k] = B[(size_t)k * n + j];

    /* Every row-strand i is struck against every vein j. Many hands
       (threads) pull their own comb across disjoint rows; within one
       crossing, a comb-stroke rings 8 shared beads (2 SIMD lanes of 4)
       at once -- a chord, not a scale. Two independent listening
       spines run down the crossing so one keeps ringing while the
       other is still being read, hiding FMA latency; both fall silent
       (loop ends) at the same point and are summed once. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const double *arow = A + (size_t)i * n;
        double *crow = C + (size_t)i * n;

        for (int j = 0; j < n; j++) {
            const double *vein = Bt + (size_t)j * n;

            __m256d spine0 = _mm256_setzero_pd();
            __m256d spine1 = _mm256_setzero_pd();

            int k = 0;
            int limit8 = n - (n % 8);
            for (; k < limit8; k += 8) {
                __m256d a0 = _mm256_loadu_pd(arow + k);
                __m256d b0 = _mm256_loadu_pd(vein + k);
                spine0 = _mm256_fmadd_pd(a0, b0, spine0);

                __m256d a1 = _mm256_loadu_pd(arow + k + 4);
                __m256d b1 = _mm256_loadu_pd(vein + k + 4);
                spine1 = _mm256_fmadd_pd(a1, b1, spine1);
            }

            int limit4 = n - (n % 4);
            for (; k < limit4; k += 4) {
                __m256d a0 = _mm256_loadu_pd(arow + k);
                __m256d b0 = _mm256_loadu_pd(vein + k);
                spine0 = _mm256_fmadd_pd(a0, b0, spine0);
            }

            __m256d spine = _mm256_add_pd(spine0, spine1);
            double buf[4];
            _mm256_storeu_pd(buf, spine);
            double total = (buf[0] + buf[1]) + (buf[2] + buf[3]);

            /* the last few beads have no partner comb-width left;
               sung one at a time */
            for (; k < n; k++)
                total += arow[k] * vein[k];

            /* pluck the spine's total note into a fresh pebble */
            crow[j] = total;
        }
    }

    free(Bt);
}
```

# PREDICTION

Reasoning: memory traffic through `Bt`/B scales the same way as the reference (each row *i* streams the full B/Bt once, so both are O(n) re‑reads of an n×n matrix, likely bandwidth‑bound for large n on a single core). The two gains over the reference are (a) OpenMP parallelism across rows, which the reference has none of, and (b) explicit dual‑accumulator FMA vectorization, comparable to or slightly better than the compiler's auto‑vectorization of the reference's inner SAXPY loop. Multi‑core scaling for a bandwidth‑bound kernel typically saturates well below the core count, and the O(n²) transpose is amortized and cheap. Net expectation: solidly faster than the single‑threaded cache‑blocked reference, well short of a fully packed/register‑tiled OpenBLAS microkernel.

PREDICTION: speedup_vs_blocked = 4.5

# MEASUREMENT

No `kernel_bench` call was made in this session — per the environment note above, no tools are available here; I was asked to reason, predict, and hand off the complete artifact for the external pipeline to compile and measure. This section is intentionally left as **pending**: the pipeline should report the measured `speedup_vs_blocked` (and, if run, `speedup_vs_openblas`) against the PREDICTION line above.

# VERDICT

Unmeasured as of this answer — plain statement, not a claim of success. The mapping is literal (transposed "veins," SIMD "chord," dual‑accumulator "spine," thread‑parallel "many combs at once") and structurally different from OpenBLAS's packed‑panel microkernel, so it is a genuine test of the seed rather than a relabeled cache‑blocked loop. If the pipeline's measurement falls short of the 4.5× prediction, the next (unused) improvement steps, in order of expected leverage, would be: (1) block the i/j loops so several rows share one pass over `Bt` while it's still cache‑resident, cutting repeated memory traffic; (2) widen the register‑blocking to a small (i×j) micro‑tile (e.g. 4×4 cells at once) so each pass over a strand/vein pair services multiple output cells; (3) switch to AVX‑512 if available; (4) only then consider SEED 3's verify‑and‑restring pass, and only if a numerically riskier variant (e.g. reduced‑width or reordered accumulation) is introduced to chase further speed.