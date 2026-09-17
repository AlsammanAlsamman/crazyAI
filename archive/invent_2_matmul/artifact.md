# MAPPING (all three seeds)

| World object | Problem object | Assumption it breaks |
|---|---|---|
| **SEED 1 — the wall** | | |
| wall, same stones/street | one shared memory region holding both operand matrices | "a matrix is a two-dimensional grid living in one memory" (here the region is dual-purpose) |
| western/eastern face, "unseeing" the other | each side (thread/core) reads only its own contiguous layout, never the other's stride pattern | "one processor holds both matrices" (each side only *sees* its own face) |
| **SEED 2 — the grows** | | |
| grow, rooted at a shared index | one FMA (multiply-add) instruction issued at index k | "one product is one problem; many products are many problems" (products become concurrent, schedulable events, not a serial list) |
| "grow's root belongs to neither city" | the CPU's vector/FMA execution unit — not part of either matrix's memory | "numbers are IEEE doubles and multiply is the primitive" (the real primitive here is a *timed* FMA event, not a bare multiply) |
| "takes as long to fruit as its numbers are large" | FMA pipeline latency | same as above — latency, not just the op, is a resource to manage |
| **SEED 3 — the baskets** (chosen) | | |
| basket at the row/column crossing | the accumulator state for one output cell C[i,j] | "the whole sum over the shared index is finished before the next cell is started" |
| "fills from many grows... weighed whole at once" | products are *not* summed incrementally one at a time; they accumulate into several independent partial sums and are combined in a single reduction step at the end | same, plus "the output is produced one cell at a time, row by row" (many baskets/cells are in flight concurrently across threads) |
| third neutral wall | matrix C, a memory region distinct from A and B, written only once per cell, after weighing | "the output is produced one cell at a time" |
| burn spent grows / reuse baskets | accumulator registers are zeroed and reused for the next cell, no per-cell allocation | (support detail, not a broken assumption) |

# CHOSEN SEED
SEED 3 — the baskets. It is the most literal (a basket = an accumulator; "weighed whole" = one final reduction instead of a running sum) and it is the one that diverges furthest from the known way: OpenBLAS's microkernel *immediately* folds each product into a running register sum in a fixed sequential order inside a tiny tile; the native design instead insists the sum is *deferred* and *batched*, and that many baskets (cells) are filled concurrently rather than row-by-row.

# ASSUMPTION BROKEN
"The whole sum over the shared index is finished before the next cell is started" (and, as a corollary, "the output is produced one cell at a time, row by row"). Concretely: instead of one running accumulator per cell that must complete its serial chain of dependent adds before the next cell begins, I keep several **independent** partial sums per cell (so the CPU can have many FMAs "growing" at once, hiding each other's latency) and only perform the summed weighing once, at the very end of that cell's k-loop — and many cells across the grid are being filled at the same wall-clock instant by different threads.

# ARTIFACT

Literal object mapping used in the code below:
- **wall / two faces** → B is re-laid onto its own "face" `Bt` (transposed into scratch memory) so that a row of A and a "column" of B are both contiguous — each side is read without straddling the other's stride.
- **grow** → one `_mm256_fmadd_pd` call, rooted at a 4-wide slice of the shared index k, reading 4 doubles from A's row and 4 from Bt's row at once.
- **basket** → the 8 independent `acc0..acc7` vector accumulators for one cell (i,j) — spent grows keep landing in them without ever being pre-summed.
- **"weigh the whole basket at once"** → the single tree-reduction (`acc0+acc1+...+acc7` → horizontal sum) done exactly once, after the k-loop, not interleaved with it.
- **third neutral wall** → `C`, written once per cell after weighing.
- **processor / city** → an OpenMP thread, each owns disjoint rows `i`, never touching another thread's rows.
- **burn / reuse** → accumulators are re-`setzero`'d for the next cell; no persistent per-cell state.

```c
#include <immintrin.h>
#include <string.h>
#include <stdlib.h>

void kernel(int n, const double *A, const double *B, double *C) {
    size_t N = (size_t)n;
    if (N == 0) return;

    /* the wall's second face: B re-laid so a "column" becomes contiguous,
       so a grow can draw sap from both faces without crossing strides */
    double *Bt = (double *)malloc(N * N * sizeof(double));
    if (!Bt) {
        /* fallback: plain correct triple loop, never leave C wrong */
        memset(C, 0, N * N * sizeof(double));
        for (size_t i = 0; i < N; i++)
            for (size_t k = 0; k < N; k++) {
                double a = A[i * N + k];
                for (size_t j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
            }
        return;
    }

    #pragma omp parallel for schedule(static)
    for (long k = 0; k < (long)N; k++)
        for (size_t j = 0; j < N; j++)
            Bt[j * N + (size_t)k] = B[(size_t)k * N + j];

    #pragma omp parallel for schedule(static)
    for (long ii = 0; ii < (long)N; ii++) {
        size_t i = (size_t)ii;
        const double *arow = A + i * N;
        double *crow = C + i * N;

        for (size_t j = 0; j < N; j++) {
            const double *brow = Bt + j * N;   /* the paired column, now a face of its own */

            /* eight independent baskets-in-progress: many grows fruit concurrently,
               none is summed into another until the whole basket is weighed */
            __m256d acc0 = _mm256_setzero_pd(), acc1 = _mm256_setzero_pd();
            __m256d acc2 = _mm256_setzero_pd(), acc3 = _mm256_setzero_pd();
            __m256d acc4 = _mm256_setzero_pd(), acc5 = _mm256_setzero_pd();
            __m256d acc6 = _mm256_setzero_pd(), acc7 = _mm256_setzero_pd();

            size_t k = 0;
            for (; k + 32 <= N; k += 32) {
                acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(arow+k),    _mm256_loadu_pd(brow+k),    acc0);
                acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(arow+k+4),  _mm256_loadu_pd(brow+k+4),  acc1);
                acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(arow+k+8),  _mm256_loadu_pd(brow+k+8),  acc2);
                acc3 = _mm256_fmadd_pd(_mm256_loadu_pd(arow+k+12), _mm256_loadu_pd(brow+k+12), acc3);
                acc4 = _mm256_fmadd_pd(_mm256_loadu_pd(arow+k+16), _mm256_loadu_pd(brow+k+16), acc4);
                acc5 = _mm256_fmadd_pd(_mm256_loadu_pd(arow+k+20), _mm256_loadu_pd(brow+k+20), acc5);
                acc6 = _mm256_fmadd_pd(_mm256_loadu_pd(arow+k+24), _mm256_loadu_pd(brow+k+24), acc6);
                acc7 = _mm256_fmadd_pd(_mm256_loadu_pd(arow+k+28), _mm256_loadu_pd(brow+k+28), acc7);
            }
            for (; k + 4 <= N; k += 4)
                acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(arow+k), _mm256_loadu_pd(brow+k), acc0);

            /* weigh the whole basket at once, in one reduction, not incrementally */
            __m256d s01 = _mm256_add_pd(acc0, acc1);
            __m256d s23 = _mm256_add_pd(acc2, acc3);
            __m256d s45 = _mm256_add_pd(acc4, acc5);
            __m256d s67 = _mm256_add_pd(acc6, acc7);
            __m256d total = _mm256_add_pd(_mm256_add_pd(s01, s23), _mm256_add_pd(s45, s67));

            double buf[4];
            _mm256_storeu_pd(buf, total);
            double s = (buf[0] + buf[1]) + (buf[2] + buf[3]);

            for (; k < N; k++) s += arow[k] * brow[k];  /* leftover grows, planted last */

            crow[j] = s;   /* dropped once, whole, onto the third neutral wall */
        }
    }

    free(Bt);
}
```

# PREDICTION
Basis: transpose cost is O(n²) versus O(n³) compute (cheap); the dot-product form removes strided B access entirely; 8 independent accumulator streams should come close to hiding AVX2 FMA latency (Skylake-class: 4-cycle latency, 2/cycle throughput ⇒ ~8 in flight needed); OpenMP parallelizes over disjoint rows with zero synchronization. Risk, stated honestly: unlike the known way, this kernel does **not** block/reuse a B-panel across multiple rows of A, so for n large enough that `Bt` (n²·8 bytes) exceeds L3, it degrades toward memory-bandwidth-bound and the advantage shrinks — I expect this to land clearly ahead of a single-threaded cache-blocked triple loop mainly *because* of threading + latency-hiding, but likely still behind OpenBLAS's packed microkernel, which additionally reuses panels across tiles.

PREDICTION: speedup_vs_blocked = 4.0

# MEASUREMENT
Not run in this session — no `kernel_bench` tool is available here (per the environment note). The prediction above was written before any measurement, as required; the pipeline should compile and run the artifact against `kernel_bench` to fill this in.

# VERDICT
Pending actual measurement. Honest expectation, stated plainly: this basket-based (deferred-reduction, transpose-first, multi-accumulator, thread-per-row) kernel should beat a single-threaded cache-blocked triple loop by a solid margin dominated by core count and latency-hiding, but is unlikely to beat OpenBLAS, since it deliberately omits the panel-reuse blocking that SEED 1 (not SEED 3) would have suggested and that OpenBLAS relies on. If measurement shows it losing even to the blocked baseline at large n, the most likely cause is `Bt` falling out of L3 and the kernel becoming bandwidth-bound — the next (unspent) improvement would be to reintroduce a modest row-block of A (reusing each `Bt` row across a handful of `i`'s) without fully collapsing back into BLAS-style panel blocking.