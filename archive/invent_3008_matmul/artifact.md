# MAPPING

| World object | Problem object | Silent assumption it targets |
|---|---|---|
| **Seed 1 — thieves at a crossing** | | |
| rib (row, planted upright, holding notes) | row *i* of A; note on the rib = A[i][k] | |
| flat (column, planted crosswise, holding weights) | column *j* of B; weight on the flat = B[k][j] | |
| crossing | one (i,k,j) triple | |
| trained thief, "owes his debt to his one meeting only" | one independent multiply‑add task, no shared mutable state with any other triple | breaks **(h)** "one product is one problem; many products are many problems" — the myth insists every product genuinely *is* its own schedulable problem, not a step fused invisibly into one thread's dot‑product |
| pressing two sparks into one coal | the scalar product A[i][k]·B[k][j] | |
| avenue shared by thieves of one destination corner | the reduction path for a fixed (i,j) | |
| basin, pooling, slow burning-together | the accumulator for C[i][j] | |
| **Seed 2 — basins pool concurrently** | | |
| many avenues feeding many basins at once, none privileged | all C[i][j] cells receive coal simultaneously as k advances | breaks **(b)** "whole sum over shared index finished before next cell started" — the myth has every basin gaining smoke *in lock-step*, not cell-by-cell to completion |
| **Seed 3 — read only a cold basin** | | |
| "I will not read a number off a fire that still moves" | a cell may only be trusted once its full reduction is provably done, independent of loop-nest text order | breaks **(a)** "output produced one cell at a time, row by row" — readiness is gated by completion of a basin, not by textual row order, and that gate must be built explicitly, not inherited for free from serial execution |

# CHOSEN SEED

**Seed 1** (trained thieves, one per crossing, pressing sparks into a coal, carried down a shared avenue into a basin). It is the most literal of the three: every world-noun (rib, flat, crossing, thief, spark, coal, avenue, basin, ash) has a one-to-one computational counterpart, and it is the most different in *organizing principle* from "the known way": OpenBLAS/MKL think in terms of pre-packed panels and a fused register microkernel that never treats a single scalar product as a first-class unit; the myth insists on flattening the whole n³ index cube into independent, individually-owned units that only meet again at the destination.

# ASSUMPTION BROKEN

**(h) "one product is one problem; many products are many problems."** The standard cache-blocked triple loop fuses multiply-then-accumulate inside one thread's serial dot product — a product is never an independent, ownable unit. The myth makes every (i,k,j) triple its own thief with his own crossing, and only regroups them afterward by shared destination corner (basin). Practically this licenses handing out *all n³ crossings at once* across two independent axes of parallelism simultaneously — SIMD lanes (thieves working adjacent j-corners of the same i,k crossing) and OpenMP threads (each owning a whole district of avenues, i.e. a band of rows i) — rather than treating parallelism as an afterthought bolted onto a single serial reduction.

# ARTIFACT: world objects → computational objects

- **Memory that stays still:** A and B (row-major, read-only "ribs" and "flats"); a block of C rows owned by exactly one thread for its whole lifetime = the "basin" that only that thread's thieves may pour into (no cross-thread write to the same row ⇒ no lock/atomic needed to honor "read only a cold basin" — ownership itself enforces it).
- **What flows:** sparks = operand loads (A[i][k] broadcast, B[k][j] SIMD loads); coal = product, pressed and immediately carried into the basin in one hardware step via FMA (press + carry fused, exactly as the myth's thief does both in one motion).
- **Processor:** an OpenMP thread = district boss owning a band of avenues (rows *i*); an AVX2 lane = one thief within that district, working one destination corner (one *j*) per instruction, 4 (or 8, unrolled) thieves working in lockstep per k-step.
- **Time:** k advances as discrete shifts; a k-block (KB) is the "night's work" a rib/flat pair puts in before its notes/weights, once every assigned thief for that block has crossed, become ash (never revisited — the loop never returns to an already-consumed k range for that row, so old A/B cache lines can be evicted).
- **Basin cold / trustworthy read:** guaranteed structurally — a given C row is written only by the one thread that owns it, across the entire kk sweep, so by the time the loop nest exits that row is provably cold; no separate barrier primitive is needed.

```c
#include <immintrin.h>
#include <omp.h>
#include <string.h>

void kernel(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));

    const int KB = 256;  /* k-crossings a rib/flat feed before turning to ash   */
    const int JB = 512;  /* width of the district of destination corners a
                             row keeps resident (its basin block)               */

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double *Crow = C + (size_t)i * n;
        for (int jj = 0; jj < n; jj += JB) {
            int jmax = jj + JB < n ? jj + JB : n;
            for (int kk = 0; kk < n; kk += KB) {
                int kmax = kk + KB < n ? kk + KB : n;
                for (int k = kk; k < kmax; k++) {
                    double a = A[(size_t)i * n + k];
                    __m256d va = _mm256_set1_pd(a);
                    const double *Brow = B + (size_t)k * n;
                    int j = jj;
                    /* 8-wide: two lanes of thieves pressing sparks in parallel */
                    for (; j + 8 <= jmax; j += 8) {
                        __m256d vb0 = _mm256_loadu_pd(Brow + j);
                        __m256d vb1 = _mm256_loadu_pd(Brow + j + 4);
                        __m256d vc0 = _mm256_loadu_pd(Crow + j);
                        __m256d vc1 = _mm256_loadu_pd(Crow + j + 4);
                        vc0 = _mm256_fmadd_pd(va, vb0, vc0); /* press + carry into basin */
                        vc1 = _mm256_fmadd_pd(va, vb1, vc1);
                        _mm256_storeu_pd(Crow + j,     vc0);
                        _mm256_storeu_pd(Crow + j + 4, vc1);
                    }
                    for (; j + 4 <= jmax; j += 4) {
                        __m256d vb = _mm256_loadu_pd(Brow + j);
                        __m256d vc = _mm256_loadu_pd(Crow + j);
                        vc = _mm256_fmadd_pd(va, vb, vc);
                        _mm256_storeu_pd(Crow + j, vc);
                    }
                    for (; j < jmax; j++) {
                        Crow[j] += a * Brow[j];   /* lone thief, scalar crossing */
                    }
                }
            }
        }
    }
}
```

PREDICTION: speedup_vs_blocked = 5.0

# MEASUREMENT

No `kernel_bench` / `kernel_contract` tool was reachable in this session (the function list actually available to me contains only Docs/Gmail/Drive/Slack tools, not the symbolic/kernel benchmarking tools named in the task). Per the task's own fallback instruction, I am not fabricating a number here — the prediction above was written down before any execution, and actual compilation/benchmarking against the cache-blocked triple loop (and against OpenBLAS) must be run by the pipeline that has those tools. I have nothing honest to report in this slot beyond the stated prediction; I am not going to quietly invent a measured figure.

# VERDICT

Reasoned expectation, not yet confirmed: this kernel gives every thief a genuinely independent crossing and lets an entire OpenMP thread privately own its basin (its row-band) for the whole k-sweep, so it should scale close to core-count on the threading axis and pick up a further ~2–4× from explicit AVX2/FMA over a scalar cache-blocked reference that may or may not auto-vectorize as aggressively — hence the conservative combined estimate of ~5×. It should plainly **not** be expected to beat OpenBLAS: OpenBLAS's packed panels + register-tiled microkernel avoid the repeated cache/TLB round-trips this kernel still takes on the C block during the kk-sweep, and its microkernel keeps far more of the "basin" resident in registers than a JB=512 block can. If measurement shows this literal thief-per-crossing design losing to OpenBLAS, that is the honest, expected outcome, not a bug in the mapping.