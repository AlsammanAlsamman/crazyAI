# MAPPING (per seed)

**Seed 1 — "A row's number is cut into its trough as notches setting how wide the water may run."**

| World object | Problem object |
|---|---|
| stone spine | contiguous linear memory address space |
| trough cut for a row | a matrix row, laid out contiguously |
| notches (width) | the actual data values of that row (doubles) |
| second table's troughs, cut *at right angles*, fed from the far millrace | matrix B accessed *by column* — i.e. B needs its own contiguous physical channel, not a stride into row-major storage |

Assumption broken: **"a matrix is a two-dimensional grid living in one memory."** The myth insists A and B are carved into *two physically different, orthogonally-oriented channel systems* — it is not one grid read two ways, it is two distinct contiguous layouts (row troughs vs. column troughs). Literally: B must be transposed/packed into its own contiguous layout before use, not just indexed with a stride.

**Seed 2 — "A stack of nested owls, one per shared step, presses each pair of flows through a narrow bone throat that passes only their product."**

| World object | Problem object |
|---|---|
| one crossing (row-trough × column-trough) | one output cell C[i][j] |
| the owl at that crossing | the multiply-accumulate unit dedicated to that cell |
| nested owls, one per shared step | one FMA lane per step of the shared index k |
| "the whole yard floods together, every crossing filled in the same breath" | **every C[i][j] is computed simultaneously/independently, not row by row** |
| owl throat = product only, never sum | the multiply is a separate, pure operation from the accumulation |

Assumption broken: **"the output is produced one cell at a time, row by row."** The myth explicitly rejects sequential row-by-row completion — the entire crossing field is flooded in one simultaneous event. This is the most literal and most different-from-panel-packing idea: total independence and simultaneity of *all* output cells, not a systolic sweep of tiles.

**Seed 3 — "Each owl's throat-drip falls into the one below, pooling by gravity down the stack until the full sum collects in the cistern at the bottom."**

| World object | Problem object |
|---|---|
| falling drip from owl to owl | partial sum passed from one accumulator stage to the next |
| pooling by gravity | addition (accumulation) |
| cistern at the bottom, read only once "still" | the final scalar C[i][j], read only after the full reduction over k completes (no partial reads) |

Assumption broken: **"the whole sum over the shared index is finished before the next cell is started"** — this seed actually *keeps* that assumption for a single cell (sum finishes before the cistern is read) but reframes *how* it finishes: not one long serial dependency chain, but a stack of independent lanes pooling their partial sums only at the bottom. This breaks "every product is computed exactly, once, in a single accumulator" — instead many parallel partial pools are combined at the end.

# CHOSEN SEED

**Seed 2** — "the whole yard floods together, every crossing filled in the same breath." It is the most literal (every output cell = one independent owl-stack = one independent dot-product unit) and the most different from the known way (OpenBLAS/MKL packs panels and sweeps a register-tiled *microkernel* over blocks; the myth instead insists on total, simultaneous independence of every single crossing, with no notion of a tile boundary at all — parallelism is native to the *output grid*, not to cache-shaped panels).

# ASSUMPTION BROKEN

"The output is produced one cell at a time, row by row" (and, as a consequence, "one processor holds both matrices" — here *every* crossing gets its own independent compute event, which is naturally a many-core, fully data-parallel decomposition over the *entire* i×j grid, not a serial sweep).

# ARTIFACT — literal translation

Objects, made literal:
- **Stone spine / troughs** → A stays row-major (row troughs); B is transposed once into `Bt`, giving it its own contiguous "column-trough" layout (Seed 1, needed so a crossing's two flows are each contiguous).
- **A crossing / an owl-stack** → one C[i][j], computed as an independent AVX2 dot product.
- **Nested owls (one per shared step), throat passes only the product** → FMA lanes; four independent accumulator lanes so no single owl waits on the one below it — they all drip in parallel and only pool at the very end (this literally follows "never their sum" happening inside one throat, and the *pooling* happening only when drips are combined).
- **"Do not read a cistern until its water has gone still"** → the scalar sum is only materialized after the full k-reduction (horizontal add) completes; no partial writes to C.
- **"The whole yard floods together"** → the entire i×j grid is independent work handed to OpenMP; blocking is only used to keep a batch of A-rows resident while one pass over all "column troughs" (Bt rows) is made — a batch of crossings shares one flooding event, rather than each row waiting its turn.

```c
#include <immintrin.h>
#include <omp.h>

static inline double hsum256(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    lo = _mm_add_pd(lo, hi);
    __m128d hh = _mm_unpackhi_pd(lo, lo);
    __m128d s = _mm_add_sd(lo, hh);
    return _mm_cvtsd_f64(s);
}

void kernel(int n, const double *A, const double *B, double *C) {
    if (n <= 0) return;
    size_t N = (size_t)n;

    /* Seed 1: cut the second table into column-troughs, at right angles to
       the row-troughs, fed from its own millrace -- transpose B so a
       "crossing" always joins two contiguous flows. */
    double *Bt = (double*)_mm_malloc(N * N * sizeof(double), 64);
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++) {
        const double *Brow = B + (size_t)k * N;
        for (int j = 0; j < n; j++)
            Bt[(size_t)j * N + k] = Brow[j];
    }

    /* A batch of row-troughs that all get flooded by the same sweep across
       every column-trough before being let go ("the whole yard floods
       together, every crossing filled in the same breath"). Batch size
       chosen so the batch of A rows stays resident while Bt is swept once. */
    int IB = (int)(262144 / (8 * N));
    if (IB < 1) IB = 1;
    if (IB > 64) IB = 64;
    int nblocks = (n + IB - 1) / IB;

    #pragma omp parallel for schedule(dynamic)
    for (int bi = 0; bi < nblocks; bi++) {
        int i0 = bi * IB;
        int i1 = i0 + IB; if (i1 > n) i1 = n;

        for (int j = 0; j < n; j++) {
            const double *Brow = Bt + (size_t)j * N;

            for (int i = i0; i < i1; i++) {
                const double *Arow = A + (size_t)i * N;

                /* Seed 2/3: a stack of nested owls -- independent lanes whose
                   throats pass only the product; their drips pool by simple
                   addition, combined only at the bottom. */
                __m256d acc0 = _mm256_setzero_pd();
                __m256d acc1 = _mm256_setzero_pd();
                __m256d acc2 = _mm256_setzero_pd();
                __m256d acc3 = _mm256_setzero_pd();

                int k = 0;
                int limit = n - (n % 16);
                for (; k < limit; k += 16) {
                    acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(Arow + k),      _mm256_loadu_pd(Brow + k),      acc0);
                    acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(Arow + k + 4),  _mm256_loadu_pd(Brow + k + 4),  acc1);
                    acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(Arow + k + 8),  _mm256_loadu_pd(Brow + k + 8),  acc2);
                    acc3 = _mm256_fmadd_pd(_mm256_loadu_pd(Arow + k + 12), _mm256_loadu_pd(Brow + k + 12), acc3);
                }
                for (; k + 4 <= n; k += 4)
                    acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(Arow + k), _mm256_loadu_pd(Brow + k), acc0);

                /* the cistern is read only once the water has gone still */
                double sum = hsum256(_mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3)));
                for (; k < n; k++) sum += Arow[k] * Brow[k];

                C[(size_t)i * N + j] = sum;
            }
        }
    }

    _mm_free(Bt);
}
```

PREDICTION: speedup_vs_blocked = 1.5

Honest caveats behind that number: this design gets its win from (a) both operands being read fully contiguously (great hardware prefetch, no strided column access), (b) 4-way independent FMA accumulation hiding latency, and (c) row-batching (`IB`) cutting repeated re-streaming of `Bt` roughly `IB`-fold. But it only blocks reuse in one dimension (the A-row batch), unlike a true cache-blocked triple loop which tiles all three dimensions — so I do not expect this to approach OpenBLAS (which additionally uses a register-tiled microkernel and packs panels for L1/L2/L3 simultaneously); I'd be surprised if it beats OpenBLAS at all, and 1.5x over the blocked baseline is a modest, not heroic, claim.

# MEASUREMENT

Not available in this session — no `kernel_bench`/`kernel_contract` tool was exposed to me here (the harness stated tools would be invoked by the pipeline after this artifact is produced, not by me). I am not fabricating a number: the prediction above is the pre-registered guess: 1.5× vs. the cache-blocked baseline, likely below OpenBLAS. Whoever runs the pipeline should report the actual measured ratio in this slot before trusting the verdict below.

# VERDICT

Unverified pending actual `kernel_bench` run. Structurally: this is a legitimate, literal descendant of the myth (transpose-for-contiguity + per-cell independent flooding + multi-lane pooled reduction), distinct in shape from panel-packed microkernel GEMM, and it should land somewhere in the neighborhood of a decent cache-blocked loop — plausibly a modest win from the contiguous-access + multi-accumulator combination, but not a plausible beat of a tuned OpenBLAS install. If measurement shows it underperforming 1.0×, the honest diagnosis would be that `Bt` re-streaming across blocks dominates and `IB` needs enlarging or a second blocking dimension (over `j`) needs adding — i.e., the single-axis "flooding" reading of Seed 2 needs a second orthogonal batch (flood a block of columns too), which would be the natural next (5th) iteration if allowed.