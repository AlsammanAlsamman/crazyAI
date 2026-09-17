# MAPPING

**SEED 1 — "A jar consumes every spark... yields only their combined sum, never the sparks back."**

| World object | Problem object |
|---|---|
| jar | the running accumulator register for one output cell `C[i][j]` |
| spark dropped in | one product term `A[i][k]*B[k][j]` |
| "consumes every spark, yields only the sum" | the accumulator never re-exposes an individual product — only the running total is ever read |
| "break that jar, set a fresh one at the next crossing" | a brand-new, independent accumulator per output cell — no state carried between cells |

Assumption addressed: *"one product is one problem; many products are many problems."* But this seed only re-states what a scalar dot-product reduction already does — every textbook kernel (including the minimal example given) already uses exactly one throw-away accumulator per cell. It doesn't break anything the standard method doesn't already assume. **Least novel.**

**SEED 2 — "A spirit struck into a flash... multiplies instantly, then dies before it can wander off to make mischief."**

| World object | Problem object |
|---|---|
| spirit struck between two facing knots | the multiply-add operation |
| the flash | the transient product register, never written to memory |
| "dies before it can wander off to make mischief" | the product is consumed by a fused multiply-add before it is ever rounded/stored on its own — no separate rounded intermediate exists |

Assumption addressed: *"numbers are IEEE doubles and multiply is the primitive."* — multiply-then-add is replaced by a single fused primitive (FMA). True, but `-O3 -march=native -ffp-contract=fast` and every BLAS microkernel already do exactly this. **Not different from the known way.**

**SEED 3 — "Because September is coming and the seekers... are still fat and slow... I press them into service now, before they wake hungry — a hungry seeker devours beads instead of counting them."**

| World object | Problem object |
|---|---|
| seekers in the reeds | the float32 SIMD lanes/FMA units sitting idle inside a double-precision datapath |
| "fat and slow from August's sleep" | currently idle capacity, cheap to use, accurate *only* at coarse (float) precision |
| September / "wake hungry" | the point at which asking for more precision than float32 can hold stops being free and starts corrupting the count |
| "a hungry seeker devours beads instead of counting them" | using float32 beyond its mantissa budget silently corrupts the value instead of reporting it (rounding error, not a clean count) |
| "press them into service now, before they wake hungry" | use float32 multiply only for the coarse part of each product, while it's still safely representable — don't demand the extra correction work that would require full precision |
| two great reed-lattices, million knots | matrices `A`, `B`, `n×n` |
| the shared spine, the index row | the contracted `k` dimension common to `A`'s columns and `B`'s rows |
| the crossing, two facing knots | the pair `A[i][k]`, `B[k][j]` |
| the jar | the **double**-precision accumulator for `C[i][j]` (kept full-precision even though the multiply feeding it is downgraded) |
| tip the jar over the third lattice's knot | write the finished sum into `C[i][j]` |
| burn the old reeds before evening | free the scratch float32 copies of `A`,`B` once the crossing-walk is done |

Assumption addressed: *"numbers are IEEE doubles and multiply is the primitive."* This is broken on purpose and literally: the multiply itself is demoted to a cheaper, faster primitive (float32) while doubles are kept only as storage/accumulator currency. This is also the seed most unlike the known way — OpenBLAS/MKL never touch precision; they only rearrange memory and registers. **Most literal and most different.**

# CHOSEN SEED
SEED 3 — the seekers pressed into counting work only before their hunger (precision demand) wakes.

# ASSUMPTION BROKEN
"Numbers are IEEE doubles and multiply is the primitive." Here the multiply primitive is float32 (fast, half-width-in-bits, double the SIMD lane count), used only for the coarse/high-order part of each product, while storage and the running sum (the jar) stay double. This trades exactness for speed with a stated bound (~2^-24 relative error per element from truncating `A`,`B` to float32), which the contract explicitly allows.

# ARTIFACT

```c
#include <string.h>
#include <stdlib.h>

void kernel(int n, const double *A, const double *B, double *C) {
    if (n <= 0) return;
    size_t N = (size_t)n;

    /* "Press the seekers into service now, before they wake hungry":
       cast both lattices down to the lighter, faster float32 workers
       while their values are still small enough to count safely.
       One-time O(n^2) pass, done once, before the O(n^3) crossing-walk. */
    float *Af = (float*)malloc(N * N * sizeof(float));
    float *Bf = (float*)malloc(N * N * sizeof(float));

    #pragma omp parallel for schedule(static)
    for (long long idx = 0; idx < (long long)(N * N); idx++) {
        Af[idx] = (float)A[idx];
        Bf[idx] = (float)B[idx];
    }

    memset(C, 0, N * N * sizeof(double));

    const int BS = 64; /* cache block along i, k, j */

    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < n; ii += BS) {
        int imax = ii + BS < n ? ii + BS : n;
        for (int kk = 0; kk < n; kk += BS) {
            int kmax = kk + BS < n ? kk + BS : n;
            for (int jj = 0; jj < n; jj += BS) {
                int jmax = jj + BS < n ? jj + BS : n;
                for (int i = ii; i < imax; i++) {
                    double *Ci = C + (size_t)i * N;
                    for (int k = kk; k < kmax; k++) {
                        float a = Af[(size_t)i * N + (size_t)k];
                        const float *Bk = Bf + (size_t)k * N;
                        /* the spirit's flash: struck, multiplies, dies --
                           never stored, only ever poured into the jar */
                        for (int j = jj; j < jmax; j++) {
                            Ci[j] += (double)(a * Bk[j]);
                        }
                    }
                }
            }
        }
    }

    free(Af);
    free(Bf);
}
```

Stated error: each entry of `C` carries a relative error of roughly the float32 epsilon (~1.2e-7), from truncating `A` and `B` to float32 before multiplying; the k-sum itself is accumulated in double, so no further growth comes from the reduction — only from the initial truncation.

PREDICTION: speedup_vs_blocked = 3.5

# MEASUREMENT
No tool access is available in this session (`kernel_bench` is not among the callable functions here), so I did not execute the benchmark myself. The prediction above is offered as a falsifiable number for the external pipeline to check against the cache-blocked double baseline; I have not fabricated a measured result.

# VERDICT
Unmeasured by me — genuinely undetermined, not claimed as a win. My honest expectation, reasoned rather than observed: the OpenMP threading and the halved memory footprint of `Af`/`Bf` (better cache reuse, less bandwidth) should give a real speedup over a single-threaded cache-blocked double loop; the float32 multiply itself only pays off if the compiler actually widens it to 8-wide SIMD before narrowing back to double for the add into `Ci[j]`, which is not guaranteed by auto-vectorization and is the main risk to the prediction. If the measured speedup comes in well below 3.5, the most likely reason is exactly that: the mixed float/double inner loop failing to vectorize as hoped, in which case the next iteration (not yet spent, since I couldn't measure this one) would hand-vectorize the float32 multiply with AVX2 intrinsics before the double accumulate, rather than quietly abandoning the precision-splitting idea for a plain double microkernel.