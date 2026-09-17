# MAPPING

| World object | Problem object | Assumption(s) broken |
|---|---|---|
| **SEED 1 — mirror-lake crossing** | | |
| East avenue of stone houses, door numbers carved by the resident mason | Rows of A, row-major, each row's n values fixed in place | — |
| South avenue of houses along the far bank | Columns of B, replanted as contiguous rows (they must be physically "planted" — B's columns aren't naturally contiguous) | *"a matrix is a two-dimensional grid living in one memory"* — B's column-data is forced to live in a second, purpose-built memory region |
| The whole million reflections falling **at once**, "never one drawn before another" | Every output cell (i,j) is dispatched to hardware concurrently — no scan order over the output grid | *"the output is produced one cell at a time, row by row"* |
| Crossing marks the shared index | The (i,j) pair whose sum runs over k | — |
| **SEED 2 — neighbouring suns** | | |
| Pull = product, "no more, no less" | Scalar/vector FMA `a*b`, exact IEEE double | Breaks nothing new — it's the identity mapping (multiply stays the primitive). Least differentiating seed. |
| **SEED 3 — forgetting scratch-tablet** | | |
| Tablet that forgets the instant a cloud crosses the sun | A private register/local accumulator, never touching main memory until done | *"a matrix is a two-dimensional grid living in one memory"* for the accumulator — partial sums live nowhere in RAM at all |
| Chalk cheap, stone not; cut sum once into new house | C[i][j] is written exactly once, no repeated read-modify-write traffic | Removes the memory-traffic version of *"whole sum finished before next cell started"* (baseline `+=` hits C's memory n times per row; here C is touched once per crossing) |

# CHOSEN SEED

**SEED 1** (the mirror-lake crossing). It is the most literal (avenue → row-major array, avenue → transposed array, lake → output index space, "all at once" → concurrent dispatch) and the most different from the known way: OpenBLAS's whole architecture is a *deliberately ordered* sweep over packed panels for cache reuse — the opposite of "never one drawn before another." Taking the seed literally means: don't impose a blocking order at all; let every (i,j) crossing be independent and let the hardware (cores + SIMD lanes) realize the simultaneity, only paying the "planting" cost once (transposing B) so that each crossing's reflection-pair is contiguous.

# ASSUMPTION BROKEN

*"The output is produced one cell at a time, row by row."* — replaced by: the whole output grid is treated as a set of independent, unordered crossings dispatched to all cores at once (OpenMP), each crossing itself resolved by a private, memory-invisible scratch accumulator (SIMD register) that never round-trips through C until the single, final "carve into stone."

# ARTIFACT

```c
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static void kernel_fallback(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a = A[(size_t)i * n + k];
            const double *Brow = B + (size_t)k * n;
            double *Crow = C + (size_t)i * n;
            for (int j = 0; j < n; j++) Crow[j] += a * Brow[j];
        }
    }
}

void kernel(int n, const double *A, const double *B, double *C) {
    if (n <= 0) return;

    /* Plant the second avenue: replant B's columns as contiguous rows
       (chalk/scratch memory, not stone) so a "south-house" is a
       contiguous run of doubles, just like an "east-house" row of A. */
    double *Bt = (double *)malloc((size_t)n * n * sizeof(double));
    if (!Bt) { kernel_fallback(n, A, B, C); return; }

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++) {
        const double *Brow = B + (size_t)k * n;
        for (int j = 0; j < n; j++) Bt[(size_t)j * n + k] = Brow[j];
    }

    /* Every (i,j) crossing is an independent reflection: let all cores
       take crossings in whatever order they reach them ("never one
       drawn before another"). Within one crossing, chalk each
       neighbouring-suns pull (a*b) into a private, instantly-forgettable
       scratch accumulator (SIMD registers), sum orbit by orbit over the
       shared index k, then cut the finished sum permanently into the
       new house of C exactly once. */
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < n; i++) {
        const double *Arow = A + (size_t)i * n;
        double *Crow = C + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            const double *Bcol = Bt + (size_t)j * n;
            __m256d acc0 = _mm256_setzero_pd();
            __m256d acc1 = _mm256_setzero_pd();
            int k = 0;
            int limit = n - (n % 8);
            for (; k < limit; k += 8) {
                __m256d a0 = _mm256_loadu_pd(Arow + k);
                __m256d b0 = _mm256_loadu_pd(Bcol + k);
                acc0 = _mm256_fmadd_pd(a0, b0, acc0);
                __m256d a1 = _mm256_loadu_pd(Arow + k + 4);
                __m256d b1 = _mm256_loadu_pd(Bcol + k + 4);
                acc1 = _mm256_fmadd_pd(a1, b1, acc1);
            }
            __m256d acc = _mm256_add_pd(acc0, acc1);
            double tmp[4];
            _mm256_storeu_pd(tmp, acc);
            double sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            for (; k < n; k++) sum += Arow[k] * Bcol[k];   /* wipe the tablet: */
            Crow[j] = sum;                                  /* stone, cut once */
        }
    }

    free(Bt);
}
```

PREDICTION: speedup_vs_blocked = 8.0

# MEASUREMENT

Not executed. No `kernel_bench`/`kernel_contract` tool call was available to me in this session (the environment explicitly disabled tool access here and stated the compile/measure step happens downstream in the pipeline). I am not going to fabricate a number — the line above is a stated prediction only, written before any measurement, as required. Reasoning behind the 8.0 estimate: the dominant expected gain is thread-level parallelism (multi-core OpenMP over i, roughly core-count-scaling minus the O(n²) transpose overhead, which is negligible next to O(n³) work for large n), with a smaller, uncertain contribution from the 8-wide AVX2/FMA dot-product since `-O3 -march=native` likely already autovectorizes the baseline's inner `j` loop reasonably well. Main risks to the prediction: (1) unknown core count on the actual test machine — fewer cores means a smaller ratio, more cores a larger one; (2) the transpose pass and the dot-product's read pattern of `Bt` (a fresh column-row per `j`, reused across all `i`) is bandwidth-bound for large `n` if `Bt` doesn't fit cache, which a true cache-blocked reference partially avoids by tiling — this could erode or even reverse the advantage at large `n`.

# VERDICT

Unverified in this session — genuinely, not just formally: I have not run the code. The construction is a faithful, literal reading of SEED 1 (simultaneous, unordered crossings realized as OpenMP-over-i + AVX2-FMA-over-k, with B's columns physically "replanted" into a contiguous second avenue, and C written exactly once per crossing per SEED 3's scratch-tablet). It is a legitimate, plausibly-fast GEMM strategy, but it is *not* guaranteed to beat a real cache-blocked triple loop at large `n`, because it deliberately forgoes the very thing that makes blocking work (deliberate reuse ordering) in favor of the seed's "no timing, all at once" instruction — if the pipeline's measurement shows it losing at large `n`, the honest next step (an explicit "improvement pass," not a silent reversion to the textbook method) would be to block the `j` loop so several rows of `Bt` are reused from cache across a tile of `i` before eviction, while keeping the "unordered across tiles" spirit of the seed intact.