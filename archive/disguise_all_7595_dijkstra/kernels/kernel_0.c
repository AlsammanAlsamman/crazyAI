#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

/* Index of the minimum of a[0..n), n >= 1. No NaNs are ever stored in a[]. */
static int scan_argmin(const double *a, int n)
{
    int i = 0, bi = 0;
    double best = INFINITY;

#if defined(__AVX512F__)
    if (n >= 16) {
        const __m512d inc = _mm512_set1_pd(16.0);
        __m512d m0 = _mm512_set1_pd(INFINITY), m1 = m0;
        __m512d j0 = _mm512_setzero_pd(),      j1 = _mm512_setzero_pd();
        __m512d c0 = _mm512_set_pd(7.0,6.0,5.0,4.0,3.0,2.0,1.0,0.0);
        __m512d c1 = _mm512_add_pd(c0, _mm512_set1_pd(8.0));
        for (; i + 16 <= n; i += 16) {
            __m512d v0 = _mm512_loadu_pd(a + i);
            __m512d v1 = _mm512_loadu_pd(a + i + 8);
            __mmask8 k0 = _mm512_cmp_pd_mask(v0, m0, _CMP_LT_OQ);
            __mmask8 k1 = _mm512_cmp_pd_mask(v1, m1, _CMP_LT_OQ);
            m0 = _mm512_min_pd(m0, v0);
            m1 = _mm512_min_pd(m1, v1);
            j0 = _mm512_mask_blend_pd(k0, j0, c0);
            j1 = _mm512_mask_blend_pd(k1, j1, c1);
            c0 = _mm512_add_pd(c0, inc);
            c1 = _mm512_add_pd(c1, inc);
        }
        {
            double mb[16], ib[16];
            _mm512_storeu_pd(mb,     m0); _mm512_storeu_pd(mb + 8, m1);
            _mm512_storeu_pd(ib,     j0); _mm512_storeu_pd(ib + 8, j1);
            for (int t = 0; t < 16; t++)
                if (mb[t] < best) { best = mb[t]; bi = (int)ib[t]; }
        }
    }
#elif defined(__AVX2__)
    if (n >= 16) {
        const __m256d inc = _mm256_set1_pd(16.0);
        __m256d m0 = _mm256_set1_pd(INFINITY), m1 = m0, m2 = m0, m3 = m0;
        __m256d j0 = _mm256_setzero_pd(), j1 = j0, j2 = j0, j3 = j0;
        __m256d c0 = _mm256_set_pd(3.0,2.0,1.0,0.0);
        __m256d c1 = _mm256_add_pd(c0, _mm256_set1_pd(4.0));
        __m256d c2 = _mm256_add_pd(c0, _mm256_set1_pd(8.0));
        __m256d c3 = _mm256_add_pd(c0, _mm256_set1_pd(12.0));
        for (; i + 16 <= n; i += 16) {
            __m256d v0 = _mm256_loadu_pd(a + i);
            __m256d v1 = _mm256_loadu_pd(a + i + 4);
            __m256d v2 = _mm256_loadu_pd(a + i + 8);
            __m256d v3 = _mm256_loadu_pd(a + i + 12);
            __m256d k0 = _mm256_cmp_pd(v0, m0, _CMP_LT_OQ);
            __m256d k1 = _mm256_cmp_pd(v1, m1, _CMP_LT_OQ);
            __m256d k2 = _mm256_cmp_pd(v2, m2, _CMP_LT_OQ);
            __m256d k3 = _mm256_cmp_pd(v3, m3, _CMP_LT_OQ);
            m0 = _mm256_min_pd(m0, v0); m1 = _mm256_min_pd(m1, v1);
            m2 = _mm256_min_pd(m2, v2); m3 = _mm256_min_pd(m3, v3);
            j0 = _mm256_blendv_pd(j0, c0, k0); j1 = _mm256_blendv_pd(j1, c1, k1);
            j2 = _mm256_blendv_pd(j2, c2, k2); j3 = _mm256_blendv_pd(j3, c3, k3);
            c0 = _mm256_add_pd(c0, inc); c1 = _mm256_add_pd(c1, inc);
            c2 = _mm256_add_pd(c2, inc); c3 = _mm256_add_pd(c3, inc);
        }
        {
            double mb[16], ib[16];
            _mm256_storeu_pd(mb,      m0); _mm256_storeu_pd(mb + 4,  m1);
            _mm256_storeu_pd(mb + 8,  m2); _mm256_storeu_pd(mb + 12, m3);
            _mm256_storeu_pd(ib,      j0); _mm256_storeu_pd(ib + 4,  j1);
            _mm256_storeu_pd(ib + 8,  j2); _mm256_storeu_pd(ib + 12, j3);
            for (int t = 0; t < 16; t++)
                if (mb[t] < best) { best = mb[t]; bi = (int)ib[t]; }
        }
    }
#endif
    for (; i < n; i++)
        if (a[i] < best) { best = a[i]; bi = i; }
    return bi;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    if (m < 0) m = 0;

    size_t msz = (size_t)(m > 0 ? m : 1);

    /* ---- CSR by counting sort ---- */
    int    *off  = (int *)calloc((size_t)n + 1, sizeof(int));
    int    *fill = (int *)malloc((size_t)n * sizeof(int));
    int    *edst = (int *)malloc(msz * sizeof(int));
    double *ew   = (double *)malloc(msz * sizeof(double));

    for (int e = 0; e < m; e++) off[src[e] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    memcpy(fill, off, (size_t)n * sizeof(int));
    for (int e = 0; e < m; e++) {
        int u = src[e];
        int p = fill[u]++;
        edst[p] = dst[e];
        ew[p]   = weight[e];
    }

    /* ---- the scratch-list ----
       aidx[0..nact) : still-unfinalized nodes
       ad  [0..nact) : their current guesses
       slots [0,   nfin) hold finite guesses  (the only ones ever scanned)
       slots [nfin,nact) hold "no idea yet"   (INFINITY)
       pos[v] = slot of v, or -1 once carved in stone                        */
    int    *aidx = (int *)malloc((size_t)n * sizeof(int));
    int    *pos  = (int *)malloc((size_t)n * sizeof(int));
    double *ad   = (double *)malloc((size_t)n * sizeof(double));

    for (int i = 0; i < n; i++) { aidx[i] = i; pos[i] = i; ad[i] = INFINITY; }

    int nact = n, nfin = 0;
    {   /* put the source in slot 0 with guess 0 */
        int ps = pos[source];
        int a0 = aidx[0];
        aidx[ps] = a0;  pos[a0] = ps;
        aidx[0]  = source; pos[source] = 0;
        ad[ps] = INFINITY;      /* order matters when source == 0 */
        ad[0]  = 0.0;
        nfin = 1;
    }

    while (nfin > 0) {
        int    p  = scan_argmin(ad, nfin);
        double dv = ad[p];
        int    u  = aidx[p];

        dist_out[u] = dv;                      /* carved in stone */

        /* remove u: close the finite run, then close the active run */
        {
            int last = nfin - 1;
            if (p != last) {
                int b = aidx[last];
                aidx[p] = b; ad[p] = ad[last]; pos[b] = p;
            }
            nfin = last;
            int la = nact - 1;
            if (nfin != la) {
                int b = aidx[la];
                aidx[nfin] = b; ad[nfin] = ad[la]; pos[b] = nfin;
            }
            nact = la;
            pos[u] = -1;
        }

        /* walk its outgoing bridges */
        {
            int e0 = off[u], e1 = off[u + 1];
            for (int e = e0; e < e1; e++) {
                int v = edst[e];
                int q = pos[v];
                if (q < 0) continue;           /* already final */
                double nd = dv + ew[e];
                if (nd < ad[q]) {
                    if (q >= nfin) {           /* first real guess: promote */
                        int b = aidx[nfin];
                        aidx[q]    = b;  ad[q]    = ad[nfin]; pos[b] = q;
                        aidx[nfin] = v;  ad[nfin] = nd;       pos[v] = nfin;
                        nfin++;
                    } else {
                        ad[q] = nd;
                    }
                }
            }
        }
    }
    /* nfin == 0 -> every remaining node is unreachable; dist_out already INF */

    free(off); free(fill); free(edst); free(ew);
    free(aidx); free(pos); free(ad);
}
