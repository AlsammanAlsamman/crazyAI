# MAPPING (per-seed tables)

**SEED 1 — "field of basins, ice-notches"**

| World object | Problem object |
|---|---|
| carved basin | one array cell of A or B |
| ice-notch, counted by breath | the numeric value stored in that cell |
| table turned crosswise | B addressed/laid out in the transposed orientation so its "grain" runs opposite to A's |
| shared wall where a basin from field 1 meets one from field 2 | the virtual meeting of A[i][k] and B[k][j] at index k |

Assumption broken: mostly *"a matrix is a two-dimensional grid living in one memory"* (two separate, cross-oriented grids) — this is storage/layout commentary, not really a computation rule. Weakest, least novel seed.

**SEED 2 — "the freezing seed races down the whole shared edge instantly"**

| World object | Problem object |
|---|---|
| the seed touched once to a wall | a single scalar value loaded once: `a = A[i][k]` |
| every connected wall forced into the same crystal, instantly, not cell by cell | that one scalar is broadcast into a SIMD register and applied to an entire row of B/C in one vector instruction, not element-by-element |
| "races down the whole shared edge faster than I could walk it" | the update sweeps the whole row j=0..n-1 in vector-width strides, not a scalar loop |
| "a million cells lock in the time of one breath-ring" | many independent (row, k) broadcasts happen concurrently across cores/threads |

Assumption broken: *"the output is produced one cell at a time, row by row"*, *"the whole sum over the shared index is finished before the next cell is started"*, and *"one product is one problem; many products are many problems"* — this seed directly attacks the sequential, cell-complete-before-moving-on execution model.

**SEED 3 — "comb-toothed taper + one-way draining cane"**

| World object | Problem object |
|---|---|
| comb-tooth on the seed's taper | one lane of a fused multiply, i.e. the product A[i][k]·B[k][j] |
| column height proportional to both notch-counts | the product magnitude |
| hollow cane draining one shared line | the accumulator that sums over k into C[i][j] |
| "corridor only runs one way" | the k-reduction is a strictly forward accumulation |
| "throw out basins, keep only the cane's rings" | discard intermediate products, keep only the running sum (no materialized 3-D tensor) |

Assumption broken: this one mostly **restates** the standard MAC+reduction — closest to the textbook method, least different from the known way.

# CHOSEN SEED

SEED 2 (the freezing seed / instantaneous connected propagation). It is the most literal (a single touch → an entire connected wall-network changes state in one uncounted instant) and the most different from OpenBLAS's known approach (panel-packing + register-tiled microkernel is still fundamentally "finish each output tile's dot-products," whereas the seed insists that one broadcast event completes an entire shared row/edge before "walking" to the next one — i.e. rank‑1 / outer‑product streaming rather than inner‑product accumulation per output cell).

# ASSUMPTION BROKEN

- "the output is produced one cell at a time, row by row" — broken: each broadcast of A[i][k] updates an entire row of C in one vector sweep, and several rows are updated by the *same* B-row touch simultaneously.
- "the whole sum over the shared index is finished before the next cell is started" — broken: C[i][j] accumulates progressively across k, row-wide, not completed one (i,j) pair at a time.
- "one product is one problem; many products are many problems" — broken: a single scalar touch (one load of A[i][k]) is amplified by hardware broadcast into many simultaneous products across a full SIMD-width row, and further across a block of rows.

# ARTIFACT: every object mapped to a computational object

- basin (A) → row-major array `A[n×n]`, unchanged, read-only.
- basin (B), "crosswise" → row-major array `B[n×n]`, read row-wise (its rows are exactly the vectors the seed races along).
- the shared wall (i,k) meets (k,j) → nothing is stored for it; it exists only for the instant the broadcast register holds `A[i][k]`.
- the freezing seed touched once → `_mm256_set1_pd(A[i][k])`: one load, one broadcast into a vector register.
- "races down the whole edge faster than I could walk it" → the FMA sweep over `j = 0..n` in 4-wide (AVX2) chunks: `C[i][j..j+3] += a * B[k][j..j+3]` in one instruction, not n scalar steps.
- comb-tooth taper → each of the 4 SIMD lanes is one tooth; deeper tapers (AVX-512, 8 lanes) would just widen the comb.
- "everywhere at once, a million cells lock in one breath" → OpenMP parallel-for splits row-blocks across cores: many (row, k) touches happen concurrently, not sequentially.
- the draining cane, "corridor runs one way" → the in-place accumulator `C[i][:]`, only ever added to as k advances — this is the k-reduction, done implicitly by repeated FMA into the same memory, never revisited backward.
- "throw out the basins and walls, keep the cane's rings" → no 3-D product tensor is ever materialized; only the running row of C survives past each k-step.
- processor → one OpenMP thread per row-block (RB=4 rows), so several "canes" drain in parallel.
- time → sequential only along k for a fixed row-block (the corridor); everything else (rows, blocks, SIMD lanes) is simultaneous.

```c
#include <string.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void kernel(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));

    const int RB = 4; /* rows "frozen" together by one shared B-row touch */

    #pragma omp parallel for schedule(dynamic)
    for (int ib = 0; ib < n; ib += RB) {
        int imax = (ib + RB <= n) ? ib + RB : n;
        int rb = imax - ib;

        for (int k = 0; k < n; k++) {
            /* touch the seed: load each row's scalar once, broadcast it */
            __m256d va[4];
            for (int r = 0; r < rb; r++)
                va[r] = _mm256_set1_pd(A[(size_t)(ib + r) * n + k]);

            const double *Brow = B + (size_t)k * n;
            int j = 0;
            /* the freezing races down the whole shared edge, 4 doubles at a time */
            for (; j + 4 <= n; j += 4) {
                __m256d bvec = _mm256_loadu_pd(Brow + j);
                for (int r = 0; r < rb; r++) {
                    double *Crow = C + (size_t)(ib + r) * n;
                    __m256d cvec = _mm256_loadu_pd(Crow + j);
                    cvec = _mm256_fmadd_pd(va[r], bvec, cvec);
                    _mm256_storeu_pd(Crow + j, cvec);
                }
            }
            /* remainder: the edge that doesn't fill a whole comb */
            for (; j < n; j++) {
                double b = Brow[j];
                for (int r = 0; r < rb; r++) {
                    double *Crow = C + (size_t)(ib + r) * n;
                    Crow[j] += A[(size_t)(ib + r) * n + k] * b;
                }
            }
        }
    }
}
```

PREDICTION: speedup_vs_blocked = 6

Reasoning behind the number: single scalar A[i][k] load feeding an AVX2 FMA sweep (~4x arithmetic throughput over a scalar inner loop) times row-block reuse (RB=4 rows share each B-row load, cutting B traffic ~4x) times multi-core OpenMP parallelism (assume a modest 4–8 logical cores on the bench host, partly bandwidth-limited so not linear). I expect this to clearly beat a single-threaded cache-blocked scalar/auto-vectorized triple loop, but not to approach OpenBLAS (no packing, no register-tiled microkernel, no NUMA/bandwidth tuning), so I'm predicting a moderate, not extreme, multiple.

# MEASUREMENT

Not run by me — no `kernel_bench` (or any execution) tool was available in this session; per the task instructions, the pipeline compiles and measures the artifact above independently. I am not fabricating a number here.

# VERDICT

Unverified pending pipeline measurement. Expected outcome, stated plainly for honest comparison against the prediction above: this kernel should beat a naive/cache-blocked single-threaded triple loop by a solid margin (SIMD + row-block B-reuse + threads), but is very unlikely to beat OpenBLAS, which additionally packs A/B into cache-resident panels and uses a much wider register-blocked microkernel with tuned tile sizes — this artifact has none of that, only the broadcast/outer-product idea the seed dictated. If measurement comes back below the predicted 6x, the likely culprit (worth a follow-up iteration) is memory bandwidth on the B-row reload across row-blocks for large n, which true panel-packing would fix and this literal translation deliberately does not attempt, to stay faithful to the seed's "no packing, just an instantaneous broadcast" model.