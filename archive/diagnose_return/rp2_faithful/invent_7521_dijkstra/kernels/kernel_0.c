#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* how many chalked, unlocked stones the flock can still circle faster than
   the knot-of-three can be walked. 2048 doubles = 16 KB, L1-resident. */
#define FLOCK_CAP 2048

/* ---- the nightingale flight: index of the smallest letter among k stones ---- */
static inline int flock_argmin(const double *restrict key, int k)
{
#if defined(__AVX2__)
    if (k >= 8) {
        __m256d b0 = _mm256_set1_pd(INFINITY), b1 = _mm256_set1_pd(INFINITY);
        __m256d x0 = _mm256_castsi256_pd(_mm256_set1_epi64x(-1)), x1 = x0;
        __m256i c0 = _mm256_setr_epi64x(0, 1, 2, 3);
        __m256i c1 = _mm256_setr_epi64x(4, 5, 6, 7);
        const __m256i st = _mm256_set1_epi64x(8);
        int i = 0;
        for (; i + 8 <= k; i += 8) {
            __m256d v0 = _mm256_loadu_pd(key + i);
            __m256d v1 = _mm256_loadu_pd(key + i + 4);
            __m256d m0 = _mm256_cmp_pd(v0, b0, _CMP_LT_OQ);
            __m256d m1 = _mm256_cmp_pd(v1, b1, _CMP_LT_OQ);
            b0 = _mm256_min_pd(b0, v0);
            b1 = _mm256_min_pd(b1, v1);
            x0 = _mm256_blendv_pd(x0, _mm256_castsi256_pd(c0), m0);
            x1 = _mm256_blendv_pd(x1, _mm256_castsi256_pd(c1), m1);
            c0 = _mm256_add_epi64(c0, st);
            c1 = _mm256_add_epi64(c1, st);
        }
        double bv[8]; long long bi[8];
        _mm256_storeu_pd(bv, b0);
        _mm256_storeu_pd(bv + 4, b1);
        _mm256_storeu_si256((__m256i *)bi, _mm256_castpd_si256(x0));
        _mm256_storeu_si256((__m256i *)(bi + 4), _mm256_castpd_si256(x1));
        double best = INFINITY; int arg = -1;
        for (int t = 0; t < 8; t++) if (bv[t] < best) { best = bv[t]; arg = (int)bi[t]; }
        for (; i < k; i++) if (key[i] < best) { best = key[i]; arg = i; }
        return arg;
    }
#endif
    {
        double best = INFINITY; int arg = -1;
        for (int i = 0; i < k; i++) if (key[i] < best) { best = key[i]; arg = i; }
        return arg;
    }
}

/* ---- the knot of three threads: 3-ary indexed heap, true decrease-key ---- */
static inline void knot_up(int *restrict hn, int *restrict hp,
                           const double *restrict d, int i)
{
    int u = hn[i];
    double du = d[u];
    while (i > 0) {
        int p = (i - 1) / 3;
        int w = hn[p];
        if (d[w] <= du) break;
        hn[i] = w; hp[w] = i; i = p;
    }
    hn[i] = u; hp[u] = i;
}

static inline void knot_down(int *restrict hn, int *restrict hp,
                             const double *restrict d, int k, int i)
{
    int u = hn[i];
    double du = d[u];
    for (;;) {
        int c = 3 * i + 1;
        if (c >= k) break;
        int bestc = c;
        double bd = d[hn[c]];
        int lim = c + 3; if (lim > k) lim = k;
        for (int t = c + 1; t < lim; t++) {
            double x = d[hn[t]];
            if (x < bd) { bd = x; bestc = t; }
        }
        if (!(bd < du)) break;
        int w = hn[bestc];
        hn[i] = w; hp[w] = i; i = bestc;
    }
    hn[i] = u; hp[u] = i;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;

    /* ---- every stone starts with an unreadable, far debt ---- */
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;

    /* ---- threads, priced one direction at a time: CSR, single offset array ---- */
    int *off = (int *)malloc((size_t)(n + 2) * sizeof(int));
    size_t mm = (size_t)(m > 0 ? m : 1);
    int *edst = (int *)malloc(mm * sizeof(int));
    double *ew = (double *)malloc(mm * sizeof(double));
    int *hnode = (int *)malloc((size_t)n * sizeof(int));   /* flock ids / heap  */
    int *hpos  = (int *)malloc((size_t)n * sizeof(int));   /* stone -> slot     */
    double *fkey = (double *)malloc((size_t)n * sizeof(double)); /* flock letters */
    if (!off || !edst || !ew || !hnode || !hpos || !fkey) {
        free(off); free(edst); free(ew); free(hnode); free(hpos); free(fkey);
        return;
    }

    memset(off, 0, (size_t)(n + 2) * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 2]++;
    for (int i = 3; i <= n + 1; i++) off[i] += off[i - 1];
    for (int i = 0; i < m; i++) {
        int p = off[src[i] + 1]++;
        edst[p] = dst[i];
        ew[p] = weight[i];
    }
    /* after the destructive fill, off[0..n] are exactly the CSR row starts */

    const int *restrict EO = off;
    const int *restrict ED = edst;
    const double *restrict EW = ew;
    double *restrict D = dist_out;

    for (int i = 0; i < n; i++) hpos[i] = -1;          /* nothing chalked yet */

    /* ---- "nothing owed": I am already here, the road from me to me is free ---- */
    D[source] = 0.0;
    hnode[0] = source; fkey[0] = 0.0; hpos[source] = 0;
    int k = 1;
    int knotted = 0;

    /* ================= regime 1: send the nightingales up ================= */
    while (k > 0 && !knotted) {
        int j = flock_argmin(fkey, k);
        if (j < 0) break;
        int u = hnode[j];
        double du = fkey[j];

        /* the bird landed: lock this letter, take the stone out of the flock */
        k--;
        if (j < k) {
            fkey[j] = fkey[k];
            int w = hnode[k];
            hnode[j] = w; hpos[w] = j;
        }
        hpos[u] = -2;                 /* locked: never rebuilt again */

        int e = EO[u], ee = EO[u + 1];
        for (; e < ee; e++) {
            if (e + 8 < ee) __builtin_prefetch(&D[ED[e + 8]], 1, 1);
            int v = ED[e];
            double nd = du + EW[e];
            if (nd < D[v]) {          /* else: the thread reached nowhere, drop it */
                D[v] = nd;
                int p = hpos[v];
                if (p >= 0) fkey[p] = nd;        /* stone stays where it stood */
                else { fkey[k] = nd; hnode[k] = v; hpos[v] = k; k++; }
            }
        }

        /* the garden got crowded: knot the threads into threes */
        if (k > FLOCK_CAP) {
            for (int i = k / 3; i >= 0; i--) knot_down(hnode, hpos, D, k, i);
            knotted = 1;
        }
    }

    /* ================= regime 2: walk the knot of three ================= */
    while (k > 0) {
        int u = hnode[0];
        double du = D[u];
        k--;
        if (k > 0) {
            int w = hnode[k];
            hnode[0] = w; hpos[w] = 0;
            knot_down(hnode, hpos, D, k, 0);
        }
        hpos[u] = -2;

        int e = EO[u], ee = EO[u + 1];
        for (; e < ee; e++) {
            if (e + 8 < ee) __builtin_prefetch(&D[ED[e + 8]], 1, 1);
            int v = ED[e];
            double nd = du + EW[e];
            if (nd < D[v]) {
                D[v] = nd;
                int p = hpos[v];
                if (p >= 0) knot_up(hnode, hpos, D, p);   /* called back on */
                else { hnode[k] = v; hpos[v] = k; knot_up(hnode, hpos, D, k); k++; }
            }
        }
    }

    free(off); free(edst); free(ew); free(hnode); free(hpos); free(fkey);
}
