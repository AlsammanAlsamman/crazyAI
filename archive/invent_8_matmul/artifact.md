# MAPPING

**SEED 1 — notched puppet-rods (rows, held motionless)**

| World object | Problem object |
|---|---|
| puppet-rod | one row `A[i, :]` |
| notch burned to a pitch-depth at position along the rod | value `A[i,k]` at offset `k`, exact discrete encoding |
| "held steady before the flame... value holds still no matter how the fire gutters" | row `i` is loaded once and kept resident (registers/L1), reused unchanged across the whole sweep — an exact, noise-immune representation, not re-derived each time |
| fire | the arithmetic pipeline / clock driving the computation |

Breaks: **"a matrix is a two-dimensional grid living in one memory."** The row is physically pulled out of the grid and carried as its own portable object (a rod), independent of the 2-D array it came from.

**SEED 2 — pigment-washed foam-sheets (columns, translucent)**

| World object | Problem object |
|---|---|
| foam-sheet, one per column | column `B[:, j]`, re-expressed as a *separate*, contiguous physical object |
| pigment layer thickness in a cell | magnitude of `B[k,j]`, encoded as *how much light passes*, i.e. an analog attenuation, not a symbolic operand |
| shadow "fattened or starved" after crossing rod-gap then pigment | the product `A[i,k]·B[k,j]` realized as a physical cascade of two attenuations, not a call to an IEEE multiply instruction |

Breaks: **"numbers are IEEE doubles and multiply is the primitive."** Multiplication here is not an opcode applied to two stored numbers — it is a side effect of light being forced through two independently-encoded, independently-stored media.

**SEED 3 — wet clay + chained tally-keepers**

| World object | Problem object |
|---|---|
| clay mark on the wall | the accumulator register bound to one output cell `C[i,j]` |
| "shadows... land on the same mark, sharing the same shared index, clay darkens again, layer atop layer" | repeated FMA accumulation over `k` into that one cell |
| tally-keeper "one bound to each output mark," turning its head only when a new shadow crosses its spot | one independent accumulator/thread per output cell (or per block of cells) — no cell waits on any other, all live and darkening concurrently |
| reading knots into the third table, then scraping clay smooth | horizontal-reduce the accumulator into `C[i,j]`; scratch state is discarded, nothing persists between calls |

Breaks: **"the output is produced one cell at a time, row by row."** Every output mark has its own tally-keeper active at once — cells are not visited in row-major sequence, they accumulate independently and in parallel as shadow-events arrive.

# CHOSEN SEED

**SEED 2** (the foam-sheets). It is the most literal and the most different from "the known way": OpenBLAS packs B into small register-tiled *panels* mixing rows and columns; the native instead makes each **whole column** of B into one physically separate, translucent, contiguous object before any rod ever meets it. That is a full transpose-into-contiguous-columns done once, up front, purely so that every later "shadow" (dot-product term) crosses stride-1 memory on both sides. Seeds 1 and 3 are folded in as the supporting mechanism (row stays resident; per-cell accumulator is independent/parallel), but the sheet-transpose is the load-bearing, literal idea.

# ASSUMPTION BROKEN

"Numbers are IEEE doubles and multiply is the primitive" — refined at the memory-layout level: **a matrix is not consulted in its native 2-D layout at all.** B is re-expressed once, physically, into `n` independent contiguous "sheets" (its transpose), so that the multiply-accumulate that follows never touches a strided column again.

# ARTIFACT

```c
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

void kernel(int n, const double *A, const double *B, double *C) {
    size_t nn = (size_t)n * n;

    /* SEED 2: wash table B onto foam-sheets, one contiguous "sheet" per column. */
    size_t bytes = ((nn * sizeof(double) + 63) / 64) * 64;
    double *Bt = (double*)aligned_alloc(64, bytes ? bytes : 64);

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++) {
        const double *Brow = B + (size_t)k * n;
        for (int j = 0; j < n; j++)
            Bt[(size_t)j * n + k] = Brow[j];   /* sheet j, cell k */
    }

    const int JBLK = 64;   /* how many sheets stay "hung" together      */
    const int IBLK = 32;   /* how many rods pass those sheets before swap */

    /* SEED 3: one tally-keeper (block of independent accumulators) per
       output mark, all working concurrently, no cell waits on another. */
    #pragma omp parallel for schedule(dynamic, 1)
    for (int jj = 0; jj < n; jj += JBLK) {
        int jmax = jj + JBLK < n ? jj + JBLK : n;
        for (int ii = 0; ii < n; ii += IBLK) {
            int imax = ii + IBLK < n ? ii + IBLK : n;
            for (int i = ii; i < imax; i++) {
                /* SEED 1: rod = row i, loaded once, held steady across all j */
                const double *rod = A + (size_t)i * n;
                double *Crow = C + (size_t)i * n;

                for (int j = jj; j < jmax; j++) {
                    const double *sheet = Bt + (size_t)j * n; /* paired sheet */

                    __m256d acc0 = _mm256_setzero_pd();
                    __m256d acc1 = _mm256_setzero_pd();
                    int k = 0;
                    for (; k + 8 <= n; k += 8) {
                        __m256d a0 = _mm256_loadu_pd(rod + k);
                        __m256d a1 = _mm256_loadu_pd(rod + k + 4);
                        __m256d b0 = _mm256_loadu_pd(sheet + k);
                        __m256d b1 = _mm256_loadu_pd(sheet + k + 4);
                        acc0 = _mm256_fmadd_pd(a0, b0, acc0);  /* shadow, layer atop layer */
                        acc1 = _mm256_fmadd_pd(a1, b1, acc1);
                    }
                    __m256d acc = _mm256_add_pd(acc0, acc1);
                    double buf[4];
                    _mm256_storeu_pd(buf, acc);
                    double sum = buf[0] + buf[1] + buf[2] + buf[3];
                    for (; k < n; k++) sum += rod[k] * sheet[k];  /* tail */

                    Crow[j] = sum;  /* read the knots into the third table */
                }
            }
        }
    }

    free(Bt);  /* scrape the clay smooth, split the rods for kindling */
}
```

# PREDICTION

Threading (OpenMP over independent column-blocks, each a disjoint set of "clay marks") should give most of the win on a multi-core box; the transpose makes both operands stride-1 on the hot loop, and the 8-wide FMA unroll should get close to the compiler's own auto-vectorized cache-blocked loop per core. Net effect: roughly core-count-scaling minus transpose/reduction overhead.

PREDICTION: speedup_vs_blocked = 5.0

# MEASUREMENT

Not executed. `kernel_bench` / `kernel_contract` are not present among the tools available to me in this session (only Docs/Gmail/Drive tools are exposed here), so I have no way to actually compile or time this artifact myself. Per the session's explicit instruction, the external pipeline is expected to compile and measure the code block above; I am not fabricating a number for this section. This is a real gap, not a rounding-off — the prediction above is a reasoned estimate, not a result.

# VERDICT

Honest status: the artifact is a literal, defensible translation of the native's description (rod = resident row, sheet = transposed contiguous column, clay-mark/tally-keeper = independent per-cell accumulator run in parallel) into a transpose‑then‑vectorized‑dot‑product kernel with OpenMP and AVX2 FMA — a structurally different route from both the naive triple loop and from OpenBLAS's panel-packed microkernel. Whether it actually beats a good cache-blocked loop (let alone OpenBLAS) is unverified here and should not be claimed until `kernel_bench` is actually run by the pipeline. If it underperforms, the next iterations (not yet taken) would be: (1) tune `JBLK`/`IBLK` to L2 size, (2) move to AVX-512 or 2×2 register-blocked micro-tiles to cut the horizontal-reduce overhead, (3) parallelize the transpose more coarsely to hide its cost, (4) fall back to the saxpy/ikj broadcast form (which needs no transpose at all) if the transpose cost dominates for the tested `n`.