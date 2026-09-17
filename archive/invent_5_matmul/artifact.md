# MAPPING

**SEED 1** — *Row-frames and column-frames cross at an easel where light through two painted seams yields a product.*

| World object | Problem object |
|---|---|
| row-frame, panes left→right in shared index order | row *i* of A, laid out along *k* |
| column-frame, panes top→bottom in same shared order | column *j* of B, laid out along *k* (implies B stored/read column-contiguous, i.e. transposed) |
| pane / painted seam (thickness=magnitude, tilt=sign) | one double value, encoded as an analog quantity rather than a bit pattern |
| easel at the crossing | the arithmetic unit assigned to output cell (i,j) |
| light through two seams → darkening | the multiply, done as one physical pass, not a counted operation |
| eye "trained... to read the amount without counting" | an approximate, non-exact readout of the product |

Breaks: **"every product is computed exactly, once"** (assumption 4) — the multiply is a trained analog read, not an exact digital multiply — and weakly **"numbers are IEEE doubles, multiply is the primitive"** (assumption 5).

**SEED 2** — *A blank glass strip on the easel's hat accumulates darkenings as the shared index walks the corridor.*

| World object | Problem object |
|---|---|
| blank glass strip on the easel's hat | a private accumulator register for one output cell |
| "one atop the last" accumulation | the running partial sum over *k* |
| the corridor | the memory address space along which the shared index (k) is swept |
| "so the shared index can walk past a **fixed** row and **fixed** column **without either frame moving**" | row *i* of A and column *j* of B are held stationary (resident in registers/cache) for the whole *k* sweep — nothing is re-fetched, re-blocked, or re-tiled; only the index cursor advances |
| strip's final shade → set onto a new frame at the same crossing | the completed dot product is written once into C[i][j] |
| wiping the strip clean | the accumulator is scratch, never spills into the next cell's sum |

Breaks: **"the whole sum over the shared index is finished before the next cell is started"** is not just unbroken here, it's *literalized and enforced* as the organizing principle — which is itself a break from the **known way** (OpenBLAS/blocked-ikj), which deliberately does *not* finish a cell's sum in one pass: it streams partial rank-1 updates across many blocks/panels, touching each C cell many times. So this seed inverts the *known* method's core trick. It also breaks **"a matrix is a two-dimensional grid living in one memory"** — the column-frame is explicitly re-hung "crosswise" so that column data lives contiguously in its own frame, i.e. B is not read as a 2D grid in its native layout but re-laid-out into a second, index-contiguous memory.

**SEED 3** — *Finished frames rest on god's stands, held true by a dried man and a white pillar; any disagreement forces a full re-walk, never a patch.*

| World object | Problem object |
|---|---|
| god's stands | the finished C matrix storage |
| dried man + white pillar holding a frame level | some redundancy/checksum keeping a finished cell's value trustworthy |
| disagreement between two frames at a done crossing | a fault (bit-flip, race, mis-accumulation) detected post hoc |
| "re-walked, never patched" | algorithm-based fault tolerance: redo the whole dot product, don't try to correct in place |

Breaks: **"every product is computed exactly, once"** (assumption 4) again, but from the *reliability* side (compute redundantly, verify, discard on mismatch) rather than the *precision* side. This is a correctness/robustness technique, not a speed technique — redundant computation or checksums cost cycles, working against the "faster" requirement.

# CHOSEN SEED

**SEED 2.** It is the most literal (corridor, fixed frames, walking index map cleanly onto memory addresses and register residency) and the most different from the known way: OpenBLAS's whole design is panel-packing + partial rank-1 updates across many small blocks, i.e. it *never* finishes one C cell's sum in a single uninterrupted pass — it revisits each cell dozens of times as blocks stream through cache. SEED 2 says the opposite: hold row *i* and column *j* stationary, walk the index once, finish the sum, write once. SEED 1 (analog/approximate multiply) and SEED 3 (fault-tolerant re-walk) are both about *correctness under approximation/faults*, not about a novel data-flow — less useful for "faster," and less literal a translation of a concrete mechanism into a concrete kernel structure.

# ASSUMPTION BROKEN

Assumption 2, **"the whole sum over the shared index is finished before the next cell is started,"** combined with assumption 3, **"a matrix is a 2D grid living in one memory."** Concretely: B is re-hung crosswise (transposed once, O(n²), into a second contiguous layout) so that a "column-frame" is contiguous top-to-bottom just like a "row-frame" is contiguous left-to-right; then, for every output cell, the *whole* dot product is accumulated in a private register strip while both operand frames stay fixed (no re-fetch, no partial-sum revisits to C), and only the finished scalar ever touches C. This is the opposite data-flow discipline from cache-blocked/BLAS GEMM, which is built entirely around *not* doing this (deliberately interleaving partial updates across blocks to get reuse).

# ARTIFACT

```c
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void kernel(int n, const double *A, const double *B, double *C) {
    /* "Hang the column-frame crosswise": transpose B once so each
       column becomes a contiguous row-frame running top to bottom.
       This is the one-time cost of re-laying B into index-contiguous
       memory; everything after this never touches B's original layout. */
    double *Bt = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!Bt) {
        /* fallback: plain correct triple loop, never leave C wrong */
        memset(C, 0, (size_t)n * n * sizeof(double));
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

    const int JR = 4; /* number of easels (columns) sharing one fixed row-frame at a time */

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const double *arow = A + (size_t)i * n;   /* fixed row-frame: never moves for this i */
        int j = 0;
        for (; j + JR <= n; j += JR) {
            const double *b0 = Bt + (size_t)(j + 0) * n;
            const double *b1 = Bt + (size_t)(j + 1) * n;
            const double *b2 = Bt + (size_t)(j + 2) * n;
            const double *b3 = Bt + (size_t)(j + 3) * n;

            __m256d strip0 = _mm256_setzero_pd(); /* blank glass strips */
            __m256d strip1 = _mm256_setzero_pd();
            __m256d strip2 = _mm256_setzero_pd();
            __m256d strip3 = _mm256_setzero_pd();

            int k = 0;
            for (; k + 4 <= n; k += 4) {
                __m256d a = _mm256_loadu_pd(arow + k); /* index walks the corridor */
                strip0 = _mm256_fmadd_pd(a, _mm256_loadu_pd(b0 + k), strip0);
                strip1 = _mm256_fmadd_pd(a, _mm256_loadu_pd(b1 + k), strip1);
                strip2 = _mm256_fmadd_pd(a, _mm256_loadu_pd(b2 + k), strip2);
                strip3 = _mm256_fmadd_pd(a, _mm256_loadu_pd(b3 + k), strip3);
            }

            double sum0, sum1, sum2, sum3;
            {
                __m128d lo, hi, s;
                lo = _mm256_castpd256_pd128(strip0); hi = _mm256_extractf128_pd(strip0, 1);
                s = _mm_add_pd(lo, hi); s = _mm_hadd_pd(s, s); sum0 = _mm_cvtsd_f64(s);
                lo = _mm256_castpd256_pd128(strip1); hi = _mm256_extractf128_pd(strip1, 1);
                s = _mm_add_pd(lo, hi); s = _mm_hadd_pd(s, s); sum1 = _mm_cvtsd_f64(s);
                lo = _mm256_castpd256_pd128(strip2); hi = _mm256_extractf128_pd(strip2, 1);
                s = _mm_add_pd(lo, hi); s = _mm_hadd_pd(s, s); sum2 = _mm_cvtsd_f64(s);
                lo = _mm256_castpd256_pd128(strip3); hi = _mm256_extractf128_pd(strip3, 1);
                s = _mm_add_pd(lo, hi); s = _mm_hadd_pd(s, s); sum3 = _mm_cvtsd_f64(s);
            }
            for (; k < n; k++) { /* remainder of the corridor walk */
                double a = arow[k];
                sum0 += a * b0[k]; sum1 += a * b1[k];
                sum2 += a * b2[k]; sum3 += a * b3[k];
            }
            /* finished frame set onto its stand */
            C[(size_t)i * n + j + 0] = sum0;
            C[(size_t)i * n + j + 1] = sum1;
            C[(size_t)i * n + j + 2] = sum2;
            C[(size_t)i * n + j + 3] = sum3;
        }
        for (; j < n; j++) { /* leftover columns, one easel at a time */
            const double *b = Bt + (size_t)j * n;
            __m256d strip = _mm256_setzero_pd();
            int k = 0;
            for (; k + 4 <= n; k += 4)
                strip = _mm256_fmadd_pd(_mm256_loadu_pd(arow + k), _mm256_loadu_pd(b + k), strip);
            __m128d lo = _mm256_castpd256_pd128(strip), hi = _mm256_extractf128_pd(strip, 1);
            __m128d s = _mm_add_pd(lo, hi); s = _mm_hadd_pd(s, s);
            double sum = _mm_cvtsd_f64(s);
            for (; k < n; k++) sum += arow[k] * b[k];
            C[(size_t)i * n + j] = sum;
        }
    }
    free(Bt);
}
```

PREDICTION: speedup_vs_blocked = 4.0

Reasoning behind the number: the reference cache-blocked triple loop repeatedly reads and writes C[i][j] inside the k-loop (rank-1 update style), which is memory-traffic-heavy on C; this kernel writes each C cell exactly once, keeps both operands contiguous and stationary per crossing (literal reading of the seed), uses AVX2 FMA with 4-wide register blocking on j (~4x arithmetic reuse of the loaded A element), and parallelizes over rows with OpenMP. I expect roughly 2–3x from vectorization+reduced C traffic and a further multiplicative factor from however many cores OpenMP gets in the bench environment (conservatively assumed modest, e.g. 2-4 cores effectively available/oversubscribed), giving a combined estimate around 4x. I do **not** expect this to beat OpenBLAS, which additionally blocks for L2/L3, packs both operands into micro-panels, and uses wider (MRxNR, often AVX-512) microkernels that reuse *both* A and B blocks in registers, not just A.

# MEASUREMENT

Not executed: no `kernel_bench` / `kernel_contract` tool is present in this session's actual tool set (only Docs/Gmail/Drive/Slack tools are available here), and the environment note explicitly states no tools are available for this task — reasoning and artifact only, to be compiled and measured by the external pipeline. Reporting this plainly rather than fabricating a number: **I did not measure this kernel.**

# VERDICT

Unverified. The prediction above (speedup_vs_blocked ≈ 4.0, and *not* expected to beat OpenBLAS) is a stated, falsifiable estimate based on the literal SEED-2 mapping (stationary fixed frames + one-pass corridor accumulation + register-blocked SIMD dot product), not a measured result. It should be run through `kernel_bench` before being trusted, and if it under- or over-shoots badly, the honest next step is to check whether the transpose cost (O(n²), should be negligible next to O(n³)) or thread scheduling overhead is dominating, rather than silently reverting to the textbook blocked-ikj/BLAS method.