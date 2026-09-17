# MAPPING

| SEED | World-object → Problem-object | Silent assumption it breaks |
|---|---|---|
| **1.** "A strip-bounded shade limits how much of two aligned canvases stays awake at once during the crossing-sum." | shade → cache/register capacity limit; strip/band → a row/column tile of A, B, C; "awake squares" → the working set resident in fast memory during accumulation | "one processor holds both matrices [whole]" — but this is exactly ordinary cache blocking, already the *known* baseline. Least distinctive. |
| **2.** "A pacing tortoise walks a fixed rhythm so light-crossings land on exactly one square each, never drifting." | tortoise → the loop index over the **shared** (reduction) dimension k; fixed rhythm → k advances by exactly one, in order, never reordered/skipped; a light-crossing → one elementary product A[i][k]·B[k][j] landing in cell C[i][j]; "never drifting" → every (i,j) cell in the current tile gets its k-th contribution in the *same* tortoise step, before any cell's sum is finished | **"the whole sum over the shared index is finished before the next cell is started"** and **"the output is produced one cell at a time, row by row."** Both are broken: instead of finishing C[i][j] before touching C[i][j+1], the whole active tile advances one k at a time together (rank‑1 / outer‑product order), finishing only after the tortoise has walked the full length of k. |
| **3.** "Spent light is swept out after each strip so no square's sum is corrupted by a second, stale crossing." | spent light → A/B values already consumed for a finished tile; sweeping out → never re-reading/re-adding that data; stale crossing → double counting or accidental reuse across tiles | Mostly reinforces "every product is computed exactly, once" rather than breaking a new assumption — it's a correctness guard on seed 2, not independently generative. |

# CHOSEN SEED
Seed 2 — the pacing tortoise. It is the most literal (the shared index *is* a physical walker with a fixed step) and the most different from OpenBLAS's approach (dot‑product‑per‑output‑cell inside a packed, register‑held microkernel).

# ASSUMPTION BROKEN
"The whole sum over the shared index is finished before the next cell is started" (row‑by‑row, cell‑by‑cell output). Instead: for a fixed k, *every* cell in the currently‑unshaded tile receives its contribution simultaneously; the sum only becomes final for the whole tile once the tortoise has walked all of k.

Literal object map:
- **Two canvases (A, B):** pinned, read directly from the caller's row‑major layout — never copied/repacked into a scratch layout ("pinned flat, never re-hung mid-work" ⇒ explicitly *no* GotoBLAS-style panel packing, which is the thing that makes this different from the known way).
- **Third canvas (C):** a separate buffer, blanked to zero before any ash settles.
- **The shade / strip:** an `IB × JB` tile of rows×columns that stays "awake" — sized to fit L1/L2.
- **The tortoise:** the `k` loop, single fixed step, driving an outer-product update into the whole tile per step.
- **Light:** `A[i][k]`, broadcast (SIMD) across the row segment it falls on.
- **Ash settling:** direct `+=` into C's memory on every crossing (not a register accumulator held until the end — literal to "smoked into a canvas", not "added by hand").
- **Waiting for ash / sweeping spent light:** each tile is fully finished (whole k walked) before the shade rolls to the next tile; tiles are disjoint, so threads never revisit another thread's ash.
- **Processor:** one OpenMP thread per (row‑block, column‑band) tile — embarrassingly parallel, no re-synchronization needed since tiles don't overlap.

# ARTIFACT

```c
#include <immintrin.h>
#include <string.h>

/* Literal mapping of the tortoise/canvas method to GEMM:
   - A, B are pinned "canvases": read straight from their given
     row-major layout, never copied/repacked into a scratch layout
     ("the canvases ... pinned flat, never re-hung mid-work" - this
     is the deliberate departure from OpenBLAS's packed panels).
   - C is the third canvas, hung between them, blanked to zero
     before any ash (partial product) settles into it.
   - The grid is shaded down to one broad tile at a time: IB rows x
     JB columns. Only that tile of C, and the matching JB-wide strip
     of B, stay "awake" (cache-resident) while it is worked on.
   - The tortoise walks the shared index k at a fixed, single-step
     rhythm (k = 0,1,2,...,n-1). At every step ALL cells of the
     current tile get their crossing burned in together - the sum
     over k is NOT finished for one cell before the next is begun;
     it finishes for the whole tile only after the full walk.
   - Once the walk finishes for a tile, its ash has fully settled
     (values are final) and the shade rolls to the next tile; a
     tile's A/B values are never revisited ("spent light... swept
     out"), and tiles never overlap, so no square gets another
     tile's ash.
*/
void kernel(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));

    const int IB = 8;    /* row-block: the tortoise's shell width, in rows   */
    const int JB = 256;  /* column band: the shade's one broad band         */

    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int ib = 0; ib < n; ib += IB) {
        for (int jb = 0; jb < n; jb += JB) {
            int imax = (ib + IB < n) ? ib + IB : n;
            int jmax = (jb + JB < n) ? jb + JB : n;
            int jw   = jmax - jb;
            int jw4  = jw - (jw % 4);

            /* the tortoise walks the shared index, one fixed step at a time */
            for (int k = 0; k < n; k++) {
                const double *Brow = B + (size_t)k * n + jb;
                for (int i = ib; i < imax; i++) {
                    double a = A[(size_t)i * n + k];       /* the light   */
                    double *Crow = C + (size_t)i * n + jb; /* third canvas */
                    __m256d va = _mm256_set1_pd(a);
                    int j = 0;
                    for (; j < jw4; j += 4) {
                        __m256d vb = _mm256_loadu_pd(Brow + j);
                        __m256d vc = _mm256_loadu_pd(Crow + j);
                        vc = _mm256_fmadd_pd(va, vb, vc);   /* ash burns in */
                        _mm256_storeu_pd(Crow + j, vc);
                    }
                    for (; j < jw; j++)
                        Crow[j] += a * Brow[j];
                }
            }
            /* whole tile's ash has now settled; shade rolls to next tile */
        }
    }
}
```

# PREDICTION
speedup_vs_blocked = 6.0

Reasoning: no packing overhead, AVX2 FMA (4‑wide) inner loop, tile sized to stay L1/L2‑resident so the per‑k read‑modify‑write of C is SRAM‑speed, and embarrassingly‑parallel disjoint tiles across cores. Expected to clear a plain cache‑blocked scalar/auto‑vectorized triple loop comfortably, but likely *not* to beat OpenBLAS, since C is re‑touched in memory on every k (no register‑resident accumulator across the whole k‑reduction) rather than packed + register‑blocked as the known way does.

PREDICTION: speedup_vs_blocked = 6.0

# MEASUREMENT
No tools were available in this session (`kernel_bench` was not invoked here). Per the task's own note, the compile/measure step is performed by the downstream pipeline, not by me in this turn. I am not fabricating a number — this section is honestly empty pending that run. If given the chance to iterate, the first two things I'd check against real numbers: (a) whether IB/JB should scale with actual measured L1/L2 sizes rather than the fixed 8×256 guess, and (b) whether OpenMP `collapse(2)` with `dynamic` scheduling gives good load balance for the tail tiles when n isn't a multiple of IB/JB.

# VERDICT
Pending measurement. The design is a genuine, literal structural departure from the known way (rank‑1/outer‑product accumulation directly into memory, tile‑shaded for locality, zero packing) rather than a relabeled cache‑blocked triple loop — so it's a fair test of the seed, not a disguised textbook microkernel. Predicted to beat the cache‑blocked baseline (~6×) and predicted, honestly, to fall short of OpenBLAS because it deliberately forgoes the register‑resident C accumulator that packing exists to enable.