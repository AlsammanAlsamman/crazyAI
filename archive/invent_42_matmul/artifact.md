# MAPPING

Assumed contract (no tools in this session, so I could not read `kernel_contract`): row-major `A`, `B`, `C` of size `n×n`, `C = A×B` overwritten, signature `void matmul(const double *A, const double *B, double *C, int n)`. I also export `kernel(...)` with the same body. If the real name differs, only that one-line wrapper changes.

**SEED 1 — polyp looks at coat, eats its capacitor = product**

| world object | problem object |
|---|---|
| first table planted as coral | A, laid out ("planted") in a fixed spatial layout, one polyp = one A[i][k] |
| landscape that "years" its rows | the (row i, shared-index k) plane: a polyp's position is (i,k), its year is k |
| second table hung as coats | B, one coat = one B[k][j], hung on pipes |
| pipes running by the shared index | lines through the k-dimension; coat of year k hangs beside every polyp of year k → A[i][k] meets B[k][j] for *all* i, j at once (rank-1 / outer-product pairing) |
| horizon dissolves, nothing walks | no operand movement during compute: A[i][k] is broadcast, B[k][·] is a vector, the pairing is local (one FMA lane) |
| polyp eats capacitor; "weighted by both" | product a·b, formed **inside** the FMA (exact, never rounded/stored on its own) |
| a million polyps eat at once | SIMD lanes × register tile × cores |

Breaks: "the output is produced one cell at a time, row by row" and "one product is one problem" — the products for a whole (i,·)×(·,j) sheet are one simultaneous act.

**SEED 2 — smoke drifts along the pipe into one fog; fog's embers are the k-sums**

| world object | problem object |
|---|---|
| smoke from an eaten capacitor | the product value, existing only as a transient inside the FMA |
| pipe with row i and column j | the reduction line over k for output cell (i,j) |
| fog | an accumulator register holding a partial Σ_k; there is one fog per (i,j) and *all* are in flight concurrently |
| "fog only gives up its embers when it has finished" | the accumulator is written to C exactly once, after the **entire** k length — no partial sums stored/reloaded (no kc-blocking) |
| ember set where row and column of its pipe cross | C[i][j] = ember of pipe (i,j) |
| coats empty, coral grey → thrown away | packed copies of A and B are freed |

Breaks: "the whole sum over the shared index is finished before the next cell is started" (all n² sums are open simultaneously) and "every product is computed exactly, once" — half: it is computed exactly, but never *stored*; the product is not a value that exists.

**SEED 3 — the woman floods over the fog; crankshaft goes solid**

| world object | problem object |
|---|---|
| the woman in the machine on the rock | a linear checksum verifier (flooding = summing over all embers in a row) |
| floods out over the fog before it dries | the check runs on each row block as soon as its fogs finish, while C block is still in cache |
| crankshaft goes solid / bird-headed wire twitters | mismatch |Σ_j C[i][j] − Σ_k A[i][k]·(Σ_j B[k][j])| > tolerance → alarm; the row is re-forged exactly |
| the towers never do this | the blocked triple loop / BLAS return no correctness signal |

Breaks: "every product is computed exactly, once" — the woman's check licenses trusting a fast path and paying for a slow exact path only when something is false; also breaks "numbers are IEEE doubles… primitive" only in the sense that the result comes with a **stated bound**, not a promise.

# CHOSEN SEED

SEED 2 (the fog). Seed 1 mapped literally is the broadcast-FMA rank-1 update, which is exactly OpenBLAS's microkernel — I say so plainly rather than pretend otherwise. Seed 3 is orthogonal to speed. Seed 2 is where the native's method genuinely departs from the known way: OpenBLAS blocks the shared index (kc ≈ 256), stores partial C tiles, and reloads them — it "gives up embers before the fog has finished" and repacks A/B panels many times. The native forbids both: **pack once, run every pipe over the full k length, emit each ember exactly once.** The cost of obeying this is that a B strip of full length (n·NR·8 bytes) must live in L2 rather than L1; the gain is zero C read-modify-write traffic and a single O(n²) packing pass instead of O(n³/kc) repacking.

The computational objects: **memory that stays still** = packed A tiles (coral, MR rows × n) and packed B strips (coats, n × NR); **what flows** = nothing but the transient product inside the FMA (smoke); **processors** = the SIMD lanes of a core (polyps of one row-block eating together), with cores = separate regions of the landscape; **time** = the k axis of the pipe; **a fog** = one register accumulator, MR×NR fogs per core at a time; **the ember emerging** = the single store to C.

# ASSUMPTION BROKEN

"The whole sum over the shared index is finished before the next cell is started" — inverted: *no* sum is finished before *every* cell in a tile has started, and *no* cell is written until its sum is finished. Secondary: products are never materialized (FMA), and the result is verified rather than assumed.

# ARTIFACT

```c
/* fog_gemm.c — the native's coral / coat / fog / woman method, taken literally.
 *
 *  coral  = A packed once into row-tiles (MR rows x full k)      -> "planted"
 *  coats  = B packed once into column-strips (full k x NR cols)  -> "hung on pipes"
 *  polyp eats coat = one FMA lane: product formed inside the FMA, never stored
 *  fog    = one accumulator register per (i,j); all MR*NR fogs of a tile are open
 *           at once, and each is stored to C exactly once, after the FULL k length
 *           (no kc blocking, no partial C read-modify-write)
 *  woman  = per-row linear checksum, run on each row block while it is still hot;
 *           if the crankshaft seizes, that row is re-forged exactly in long double.
 *
 *  Assumed contract: row-major, C = A*B overwritten,
 *      void matmul(const double *A, const double *B, double *C, int n);
 *  OpenMP is used only if the harness compiles with -fopenmp; otherwise the
 *  pragmas are ignored and the kernel is single-threaded and still correct.
 */
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ SIMD width */
#if defined(__AVX512F__)
#include <immintrin.h>
typedef __m512d VEC;
#define VLEN 8
#define MR   8                                   /* 8 x 24 tile: 24 fogs */
#define VZERO()        _mm512_setzero_pd()
#define VLOAD(p)       _mm512_load_pd(p)
#define VSET1(x)       _mm512_set1_pd(x)
#define VFMA(a,b,c)    _mm512_fmadd_pd((a),(b),(c))
#define VSTOREU(p,v)   _mm512_storeu_pd((p),(v))
#define FOG_SIMD 1
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
typedef __m256d VEC;
#define VLEN 4
#define MR   4                                   /* 4 x 12 tile: 12 fogs */
#define VZERO()        _mm256_setzero_pd()
#define VLOAD(p)       _mm256_load_pd(p)
#define VSET1(x)       _mm256_set1_pd(x)
#define VFMA(a,b,c)    _mm256_fmadd_pd((a),(b),(c))
#define VSTOREU(p,v)   _mm256_storeu_pd((p),(v))
#define FOG_SIMD 1
#else
#define VLEN 4
#define MR   4
#define FOG_SIMD 0
#endif
#define NR (3 * VLEN)                            /* three vectors of coats per pipe row */

int fog_crankshaft_seized = 0;                   /* how many rows the woman rejected */

/* ------------------------------------------------------------- one tile of fogs */
#if FOG_SIMD
#define DECL_ROW(i) VEC c##i##0 = VZERO(), c##i##1 = VZERO(), c##i##2 = VZERO();
#define EAT_ROW(i)  { VEC a = VSET1(ap[i]);                       \
                      c##i##0 = VFMA(a, b0, c##i##0);            \
                      c##i##1 = VFMA(a, b1, c##i##1);            \
                      c##i##2 = VFMA(a, b2, c##i##2); }
#define EMBER_ROW(i, dst) { VSTOREU((dst), c##i##0);              \
                            VSTOREU((dst) + VLEN, c##i##1);      \
                            VSTOREU((dst) + 2 * VLEN, c##i##2); }

static void fog_tile(const double *restrict ap, const double *restrict bp, int n,
                     double *restrict c, size_t ldc, int mrows, int ncols)
{
    DECL_ROW(0) DECL_ROW(1) DECL_ROW(2) DECL_ROW(3)
#if MR == 8
    DECL_ROW(4) DECL_ROW(5) DECL_ROW(6) DECL_ROW(7)
#endif
    /* the horizon dissolves: every polyp of this tile eats along the whole pipe */
    for (int k = 0; k < n; ++k) {
        VEC b0 = VLOAD(bp), b1 = VLOAD(bp + VLEN), b2 = VLOAD(bp + 2 * VLEN);
        EAT_ROW(0) EAT_ROW(1) EAT_ROW(2) EAT_ROW(3)
#if MR == 8
        EAT_ROW(4) EAT_ROW(5) EAT_ROW(6) EAT_ROW(7)
#endif
        ap += MR;
        bp += NR;
    }
    /* the fog has finished: embers emerge, each exactly once */
    if (mrows == MR && ncols == NR) {
        EMBER_ROW(0, c) EMBER_ROW(1, c + ldc) EMBER_ROW(2, c + 2 * ldc) EMBER_ROW(3, c + 3 * ldc)
#if MR == 8
        EMBER_ROW(4, c + 4 * ldc) EMBER_ROW(5, c + 5 * ldc)
        EMBER_ROW(6, c + 6 * ldc) EMBER_ROW(7, c + 7 * ldc)
#endif
    } else {
        double tmp[MR * NR] __attribute__((aligned(64)));
        EMBER_ROW(0, tmp) EMBER_ROW(1, tmp + NR) EMBER_ROW(2, tmp + 2 * NR) EMBER_ROW(3, tmp + 3 * NR)
#if MR == 8
        EMBER_ROW(4, tmp + 4 * NR) EMBER_ROW(5, tmp + 5 * NR)
        EMBER_ROW(6, tmp + 6 * NR) EMBER_ROW(7, tmp + 7 * NR)
#endif
        for (int i = 0; i < mrows; ++i)
            for (int j = 0; j < ncols; ++j)
                c[(size_t)i * ldc + j] = tmp[i * NR + j];
    }
}
#else  /* portable fallback: same structure, compiler-vectorized */
static void fog_tile(const double *restrict ap, const double *restrict bp, int n,
                     double *restrict c, size_t ldc, int mrows, int ncols)
{
    double acc[MR][NR];
    for (int i = 0; i < MR; ++i) for (int j = 0; j < NR; ++j) acc[i][j] = 0.0;
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < MR; ++i) {
            double a = ap[i];
            for (int j = 0; j < NR; ++j) acc[i][j] += a * bp[j];
        }
        ap += MR; bp += NR;
    }
    for (int i = 0; i < mrows; ++i)
        for (int j = 0; j < ncols; ++j)
            c[(size_t)i * ldc + j] = acc[i][j];
}
#endif

/* -------------------------------------------------- exact re-forging of one row */
static void reforge_row(const double *A, const double *B, double *Ci, int n, int i)
{
    const double *Ai = A + (size_t)i * n;
    for (int j = 0; j < n; ++j) {
        long double s = 0.0L;
        for (int k = 0; k < n; ++k) s += (long double)Ai[k] * (long double)B[(size_t)k * n + j];
        Ci[j] = (double)s;
    }
}

/* ------------------------------------------ the woman floods over one row's fog */
/* v[k] = sum_j B[k][j],  w[k] = sum_j |B[k][j]|.  Returns 1 if the row is believed. */
static int woman_believes_row(const double *A, const double *v, const double *w,
                              const double *Ci, int n, int i)
{
    const double *Ai = A + (size_t)i * n;
    long double expect = 0.0L, bound = 0.0L, actual = 0.0L;
    for (int k = 0; k < n; ++k) {
        expect += (long double)Ai[k] * (long double)v[k];
        bound  += fabsl((long double)Ai[k]) * (long double)w[k];
    }
    for (int j = 0; j < n; ++j) actual += (long double)Ci[j];
    /* each ember carries <= n*u*sum_k|a||b|; summing embers and forming v add two more n*u */
    long double tol = (4.0L * (long double)n + 8.0L) * (long double)DBL_EPSILON * bound;
    long double diff = fabsl(actual - expect);
    return (diff <= tol);                 /* NaN anywhere -> not believed -> re-forged */
}

/* ------------------------------------------------------------------ the method */
static void *fog_alloc(size_t bytes, void **raw)
{
    *raw = malloc(bytes + 128);
    if (!*raw) return NULL;
    uintptr_t p = ((uintptr_t)*raw + 63) & ~(uintptr_t)63;
    return (void *)p;
}

static void fog_matmul(const double *A, const double *B, double *C, int n)
{
    if (n <= 0) return;
    const int mt = (n + MR - 1) / MR;            /* coral tiles  (row blocks of A) */
    const int nt = (n + NR - 1) / NR;            /* pipe strips  (column strips of B) */

    void *rawA, *rawB, *rawV;
    double *Ap = (double *)fog_alloc((size_t)mt * MR * n * sizeof(double), &rawA);
    double *Bp = (double *)fog_alloc((size_t)nt * NR * n * sizeof(double), &rawB);
    double *v  = (double *)fog_alloc((size_t)2 * n * sizeof(double), &rawV);
    if (!Ap || !Bp || !v) {                       /* no landscape: forge everything exactly */
        free(rawA); free(rawB); free(rawV);
        for (int i = 0; i < n; ++i) reforge_row(A, B, C + (size_t)i * n, n, i);
        return;
    }
    double *w = v + n;

    /* plant the coral: tile t holds rows t*MR.. as [k][r], zero beyond the table */
#pragma omp parallel for schedule(static)
    for (int t = 0; t < mt; ++t) {
        double *dst = Ap + (size_t)t * n * MR;
        for (int r = 0; r < MR; ++r) {
            int i = t * MR + r;
            if (i < n) { const double *Ai = A + (size_t)i * n;
                         for (int k = 0; k < n; ++k) dst[(size_t)k * MR + r] = Ai[k]; }
            else       { for (int k = 0; k < n; ++k) dst[(size_t)k * MR + r] = 0.0; }
        }
    }
    /* hang the coats: strip s holds columns s*NR.. as [k][c], zero beyond the table */
#pragma omp parallel for schedule(static)
    for (int s = 0; s < nt; ++s) {
        double *dst = Bp + (size_t)s * n * NR;
        for (int k = 0; k < n; ++k) {
            const double *Bk = B + (size_t)k * n + (size_t)s * NR;
            int full = n - s * NR; if (full > NR) full = NR;
            for (int c2 = 0; c2 < full; ++c2) dst[(size_t)k * NR + c2] = Bk[c2];
            for (int c2 = full; c2 < NR; ++c2) dst[(size_t)k * NR + c2] = 0.0;
        }
    }
    /* the woman prepares her flood: v = B*1, w = |B|*1 */
#pragma omp parallel for schedule(static)
    for (int k = 0; k < n; ++k) {
        const double *Bk = B + (size_t)k * n;
        double s = 0.0, sa = 0.0;
        for (int j = 0; j < n; ++j) { s += Bk[j]; sa += fabs(Bk[j]); }
        v[k] = s; w[k] = sa;
    }

    /* row block = MBT coral tiles (<= 48 rows) that stay in L2 while every strip goes by */
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    int MBT = (mt + 4 * nth - 1) / (4 * nth);
    if (MBT < 1) MBT = 1;
    if (MBT > 48 / MR) MBT = 48 / MR;

#pragma omp parallel for schedule(dynamic, 1)
    for (int tb = 0; tb < mt; tb += MBT) {
        int te = tb + MBT; if (te > mt) te = mt;
        for (int s = 0; s < nt; ++s) {
            int ncols = n - s * NR; if (ncols > NR) ncols = NR;
            const double *bp = Bp + (size_t)s * n * NR;
            for (int t = tb; t < te; ++t) {
                int mrows = n - t * MR; if (mrows > MR) mrows = MR;
                fog_tile(Ap + (size_t)t * n * MR, bp, n,
                         C + (size_t)t * MR * n + (size_t)s * NR, (size_t)n, mrows, ncols);
            }
        }
        /* she floods out over this block's fog before it dries */
        int i0 = tb * MR, i1 = te * MR; if (i1 > n) i1 = n;
        for (int i = i0; i < i1; ++i) {
            if (!woman_believes_row(A, v, w, C + (size_t)i * n, n, i)) {
#pragma omp atomic
                fog_crankshaft_seized++;
                reforge_row(A, B, C + (size_t)i * n, n, i);
            }
        }
    }

    free(rawA); free(rawB); free(rawV);           /* coats are empty, coral has gone grey */
}

void matmul(const double *A, const double *B, double *C, int n) { fog_matmul(A, B, C, n); }
void kernel(int n, const double *A, const double *B, double *C) { fog_matmul(A, B, C, n); }  /* argument order corrected by the pipeline author: the model had no access to kernel_contract */
```

Error statement: every ember C[i][j] is the rounded sum of n exactly-formed products (FMA), so |error| ≤ n·u·Σ_k |A[i][k]||B[k][j]| with u = 2⁻⁵³ — the same bound as any correctly-rounded-FMA GEMM, and the woman checks each row against (4n+8)·u·Σ_k|A[i][k]|·‖B[k,:]‖₁. The check is one-sided: it cannot miss a single wrong ember larger than the bound, but two wrong embers in the same row that cancel to within the bound would pass. Rows that fail are recomputed exactly in 80-bit.

# PREDICTION

Written before any measurement.

- Blocked triple loop at -O3 (no explicit SIMD register tile): ~3–5 GFLOPS on one core.
- This kernel on one AVX2/FMA core: 12 fogs in registers, 16 doubles loaded per 12 vector FMAs, B strip served from L2, A tile from L2 → I expect 65–80% of the per-core peak (~40 GFLOPS at 3.5 GHz), i.e. roughly 10–14× the blocked loop single-threaded. If the harness compiles with `-fopenmp`, multiply by ~0.85·cores.
- Versus OpenBLAS: the full-length pipes cost L1 residency of B and the 4×12 tile is smaller than OpenBLAS's; I predict **0.6–0.8× OpenBLAS**, i.e. the native's method as literally translated will *not* beat the known way. The likely failure mode is the B strip falling out of L2 for large n (n ≳ 4000), where speed will drop further.

PREDICTION: speedup_vs_blocked = 12

# MEASUREMENT

Not performed in this session — `kernel_bench` and `kernel_contract` were not available. The pipeline will compile and time the artifact above. Things to read from the measurement when it arrives:
1. `fog_crankshaft_seized` must be 0 on well-scaled inputs; a nonzero value means the SIMD path produced a wrong ember (or NaN inputs) and the slow exact path masked it — that would be a correctness bug in the tile, not a feature.
2. If speedup ≪ 12 single-threaded, the first suspect is the compiler spilling accumulators (check for `vmovapd` to stack in the k loop); second is the B strip missing L2 for the tested n.
3. If the signature does not match the contract, only the two one-line wrappers at the bottom need editing.

# VERDICT

Provisional, pending the pipeline's numbers. Taken literally, the native's method is: pack once, run every reduction pipe over its full length with all fogs of a tile open at once, form products only inside the FMA, emit each output exactly once, and verify by flooding a checksum over each finished row block. Seed 1 turns out to be the OpenBLAS microkernel itself — I do not claim novelty there. Seed 2's "the fog only gives up its embers when it has finished" is the real departure from the known way (no kc-blocking, no partial-C traffic, single packing pass), and I predict it costs more in L1 residency than it saves, landing well above the blocked loop but short of OpenBLAS. Seed 3 gives something the towers lack — a stated, checked error bound — at O(n²) cost. If the measurement contradicts the prediction in either direction, that is the result to report.