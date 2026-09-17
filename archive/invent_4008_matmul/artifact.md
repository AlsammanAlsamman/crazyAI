# MAPPING

**SEED 1 — "The goose walks the shared season between two tables to align the index for each product."**

| World object | Problem object |
|---|---|
| the two tables | matrices A (vineyard rows) and B (kitchen floor) |
| the shared season | the index k, common to A's columns and B's rows |
| the goose | the loop/iterator that drives k |
| walking, foot landing on a stake | advancing k one step, synchronized in both A and B |
| aligning the index for each product | guaranteeing A[i,k] and B[k,j] are read at the *same* k |

Assumption broken: **"one product is one problem; many products are many problems."** If k is promoted to the outermost, shared control variable rather than something nested inside a per-cell loop, one "step of the season" produces a whole layer of products (a rank-1 contribution to every output cell at once) — many products handled as one coordinated sweep, not n² separate little problems.

**SEED 2 — "The red cat pounces matching stakes to pluck one electron from each atom and drop the pair into the pot."**

| World object | Problem object |
|---|---|
| atom | one double-precision matrix element |
| electrons circling the atom ("moons", plural) | the decomposable **parts** of that double (a high-order float32 part + a float32 residual) |
| the cat | the multiply/FMA execution unit |
| pounce | one scalar multiply issued against **one chosen component**, not the whole atom |
| matching stakes | the (i,k),(k,j) index pair being combined |
| dropping the pair into the pot | accumulating the resulting partial product into a running double sum |

Assumption broken: **"numbers are IEEE doubles and multiply is the primitive"** and **"n³ multiplications are needed."** The atom is not consumed whole — a single "electron" (component) is plucked at a time, so a double×double product is *rebuilt* from several cheaper, narrower multiplies rather than being one atomic double-multiply.

**SEED 3 — "The fire recites each dropped pair into a multiplying chant, and the coins it yields are saved in a box until poured as the summed tile."**

| World object | Problem object |
|---|---|
| fire reciting a chant | the scalar multiply-add itself |
| coin | one partial product |
| box | a scratch accumulator, distinct from A's, B's, or C's memory |
| pouring the box into the tile | writing the finished sum into C[i,j] once, at the end |
| compost / husks | discarding scratch state after use |

Assumption broken: **"a matrix is a two-dimensional grid living in one memory"** — a third, temporary memory (the box) is required, outside the two input grids and the output grid.

# CHOSEN SEED

SEED 2 (the electron-plucking). It is the most literal (atoms really do have more bits than one float can hold — "moons", plural) and it is the one that diverges hardest from the known way: OpenBLAS never touches the *bits* of a double, it only re-packs and re-tiles whole doubles. Splitting the atom itself into multiple narrower "electrons" and reconstructing the product from partial pounces is a genuinely different primitive, not a re-arrangement of the same one.

# ASSUMPTION BROKEN

"Numbers are IEEE doubles and multiply is the primitive" + "n³ multiplications are needed." Here the primitive is a float32 multiply on an extracted component, and each double×double product is rebuilt from three such pounces (hi·hi + hi·lo + lo·hi), dropping the doubly-negligible lo·lo term as compost.

# ARTIFACT

```c
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

/* "Atom -> electrons": split each double into a float32 high part and a
   float32 low-order residual so that x ~= hi + lo. A later pounce only
   ever touches one electron (hi or lo) at a time. */
static void split_hilo(const double *src, float *hi, float *lo, size_t n2) {
    #pragma omp parallel for schedule(static)
    for (long t = 0; t < (long)n2; t++) {
        double x = src[t];
        float h = (float)x;
        hi[t] = h;
        lo[t] = (float)(x - (double)h);
    }
}

void kernel(int n, const double *A, const double *B, double *C) {
    const size_t N  = (size_t)n;
    const size_t N2 = N * N;

    float *Ahi = (float*)malloc(N2 * sizeof(float));
    float *Alo = (float*)malloc(N2 * sizeof(float));
    float *Bhi = (float*)malloc(N2 * sizeof(float));
    float *Blo = (float*)malloc(N2 * sizeof(float));

    split_hilo(A, Ahi, Alo, N2);
    split_hilo(B, Bhi, Blo, N2);

    memset(C, 0, N2 * sizeof(double));

    const int BI = 64;   /* stake-pairs handled by one goose-walk chunk   */
    const int BK = 128;  /* season chunk                                  */
    const int BJ = 256;  /* pot width kept resident while season is walked */

    #pragma omp parallel for schedule(dynamic)
    for (int i0 = 0; i0 < n; i0 += BI) {
        int imax = (i0 + BI < n) ? i0 + BI : n;
        for (int k0 = 0; k0 < n; k0 += BK) {
            int kmax = (k0 + BK < n) ? k0 + BK : n;
            for (int j0 = 0; j0 < n; j0 += BJ) {
                int jmax = (j0 + BJ < n) ? j0 + BJ : n;
                for (int i = i0; i < imax; i++) {
                    const float *Ahi_row = Ahi + (size_t)i * N;
                    const float *Alo_row = Alo + (size_t)i * N;
                    double *Crow = C + (size_t)i * N;
                    for (int k = k0; k < kmax; k++) {
                        /* the goose's foot lands on stake k */
                        float ah = Ahi_row[k];
                        float al = Alo_row[k];
                        const float *Bhi_row = Bhi + (size_t)k * N;
                        const float *Blo_row = Blo + (size_t)k * N;
                        int j = j0;
#if defined(__AVX2__) && defined(__FMA__)
                        __m256 vah = _mm256_set1_ps(ah);
                        __m256 val = _mm256_set1_ps(al);
                        for (; j + 8 <= jmax; j += 8) {
                            /* the cat pounces: three electron-pairs, not one atom-pair */
                            __m256 bh = _mm256_loadu_ps(Bhi_row + j);
                            __m256 bl = _mm256_loadu_ps(Blo_row + j);
                            __m256 prod = _mm256_mul_ps(vah, bh);        /* hi*hi  */
                            prod = _mm256_fmadd_ps(vah, bl, prod);       /* +hi*lo */
                            prod = _mm256_fmadd_ps(val, bh, prod);       /* +lo*hi */
                            __m128 plo = _mm256_castps256_ps128(prod);
                            __m128 phi = _mm256_extractf128_ps(prod, 1);
                            __m256d dlo = _mm256_cvtps_pd(plo);
                            __m256d dhi = _mm256_cvtps_pd(phi);
                            __m256d clo = _mm256_loadu_pd(Crow + j);
                            __m256d chi = _mm256_loadu_pd(Crow + j + 4);
                            clo = _mm256_add_pd(clo, dlo);   /* coins into the pot */
                            chi = _mm256_add_pd(chi, dhi);
                            _mm256_storeu_pd(Crow + j, clo);
                            _mm256_storeu_pd(Crow + j + 4, chi);
                        }
#endif
                        for (; j < jmax; j++) {
                            float bh = Bhi_row[j];
                            float bl = Blo_row[j];
                            double p = (double)(ah * bh) + (double)(ah * bl) + (double)(al * bh);
                            Crow[j] += p;
                        }
                    }
                }
            }
        }
    }

    free(Ahi); free(Alo); free(Bhi); free(Blo);
}
```

PREDICTION: speedup_vs_blocked = 1.15

# MEASUREMENT

No tool access was available in this session (`kernel_bench`/`kernel_contract` were not present among my callable tools), so this prediction was **not measured** — it is reasoning only, offered before any run, as required. The reasoning behind 1.15: the scheme does ~3× the vector-equivalent multiply-add work of a plain double GEMM, but each float32 op is 2× wider per instruction on AVX2 (8 lanes vs 4), so the raw compute cost is roughly 1.5× worse in instruction-issue terms. The offsetting factor is that A and B are read through half-width (float32) panels, halving the memory traffic a naive/blocked kernel pushes through cache — and a plain cache-blocked triple loop (unlike OpenBLAS's register microkernel) is usually not compute-saturated, so trading extra arithmetic for less bandwidth pressure can net a small win. This is a real trade, not a guaranteed one; it could just as easily land below 1.0.

# VERDICT

Unverified. Predicted a modest win (≈1.15×) over the cache-blocked triple loop, driven by halved A/B memory traffic outweighing the ~1.5× extra vector work from the 3-term split; genuinely possible it instead comes in flat or slightly behind, since plain AVX2 float32 FMA throughput scales almost exactly with its width advantage over double, leaving little headroom once the bandwidth win is spent. No claim is made about beating OpenBLAS — its register-tiled microkernel already keeps FMA units saturated in native double precision, and this scheme does strictly more total arithmetic, so it is expected to lose to OpenBLAS, not beat it. Block sizes (BI=64, BK=128, BJ=256) and the 3-term truncation (dropping lo·lo) are untuned defaults; correctness should be checked against the tolerance kernel_bench uses, since dropping the lo·lo term is expected to introduce roughly 1e-13–1e-11 relative error, not exactness. This is the one artifact submitted for the pipeline to actually compile and measure — the "improve up to four times" loop could not be run here for lack of tool access.