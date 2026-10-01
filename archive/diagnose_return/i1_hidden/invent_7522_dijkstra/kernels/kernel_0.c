#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

/* ---- the hand hovering over the flock: exhaustive argmin over carried threads ---- */
static int fl_argmin(const double *d, int k)
{
    int    bi = 0;
    double bv = d[0];
    int    i  = 1;
#if defined(__AVX2__)
    if (k >= 16) {
        __m256d v0 = _mm256_loadu_pd(d);
        __m256d v1 = _mm256_loadu_pd(d + 4);
        __m256d i0 = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
        __m256d i1 = _mm256_set_pd(7.0, 6.0, 5.0, 4.0);
        __m256d c0 = i0, c1 = i1;
        const __m256d step = _mm256_set1_pd(8.0);
        for (i = 8; i + 8 <= k; i += 8) {
            c0 = _mm256_add_pd(c0, step);
            c1 = _mm256_add_pd(c1, step);
            __m256d a0 = _mm256_loadu_pd(d + i);
            __m256d a1 = _mm256_loadu_pd(d + i + 4);
            __m256d m0 = _mm256_cmp_pd(a0, v0, _CMP_LT_OQ);
            __m256d m1 = _mm256_cmp_pd(a1, v1, _CMP_LT_OQ);
            v0 = _mm256_blendv_pd(v0, a0, m0);
            i0 = _mm256_blendv_pd(i0, c0, m0);
            v1 = _mm256_blendv_pd(v1, a1, m1);
            i1 = _mm256_blendv_pd(i1, c1, m1);
        }
        {
            double tv[8], ti[8];
            int t;
            _mm256_storeu_pd(tv, v0);     _mm256_storeu_pd(tv + 4, v1);
            _mm256_storeu_pd(ti, i0);     _mm256_storeu_pd(ti + 4, i1);
            bv = tv[0]; bi = (int)ti[0];
            for (t = 1; t < 8; t++) if (tv[t] < bv) { bv = tv[t]; bi = (int)ti[t]; }
        }
    }
#endif
    for (; i < k; i++) if (d[i] < bv) { bv = d[i]; bi = i; }
    return bi;
}

#ifdef _OPENMP
static int fl_argmin_par(const double *d, int k, int nt)
{
    double bv[64]; int bi[64];
    int t; double bestv = INFINITY; int best = 0;
    for (t = 0; t < nt; t++) { bv[t] = INFINITY; bi[t] = 0; }
    #pragma omp parallel num_threads(nt)
    {
        int th = omp_get_thread_num();
        int T  = omp_get_num_threads();
        long long chunk = ((long long)k + T - 1) / T;
        long long s = (long long)th * chunk;
        long long e = s + chunk;
        if (e > (long long)k) e = k;
        if (s < e && th < 64) {
            int j = fl_argmin(d + s, (int)(e - s));
            bv[th] = d[s + j];
            bi[th] = (int)(s + j);
        }
    }
    for (t = 0; t < nt && t < 64; t++) if (bv[t] < bestv) { bestv = bv[t]; best = bi[t]; }
    return best;
}
#endif

/* ---- guard path only: 4-ary lazy cascade (still no decrease-key, no pre-relax query) ---- */
static void sink4(double *hk, int *hv, int sz, int i)
{
    double lk = hk[i];
    int    lv = hv[i];
    for (;;) {
        int c = 4 * i + 1, end, b, t;
        double bk;
        if (c >= sz) break;
        end = c + 4; if (end > sz) end = sz;
        b = c; bk = hk[c];
        for (t = c + 1; t < end; t++) if (hk[t] < bk) { bk = hk[t]; b = t; }
        if (!(bk < lk)) break;
        hk[i] = bk; hv[i] = hv[b]; i = b;
    }
    hk[i] = lk; hv[i] = lv;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    int *head, *eto, *pos, *fid;
    double *ew, *fd;
    int i, k, nt = 1;
    double work, budget;

    if (n <= 0) return;
    for (i = 0; i < n; i++) dist_out[i] = INFINITY;      /* bare corners: light simply lost */
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;                              /* written first */
    if (m <= 0) return;

    head = (int *)malloc((size_t)(n + 1) * sizeof(int));
    eto  = (int *)malloc((size_t)m * sizeof(int));
    ew   = (double *)malloc((size_t)m * sizeof(double));
    pos  = (int *)malloc((size_t)n * sizeof(int));
    fid  = (int *)malloc((size_t)n * sizeof(int));
    fd   = (double *)malloc((size_t)n * sizeof(double));
    if (!head || !eto || !ew || !pos || !fid || !fd) {
        free(head); free(eto); free(ew); free(pos); free(fid); free(fd); return;
    }

    /* the roads out of each corner, gathered so a stone's ducks fly contiguously */
    memset(head, 0, (size_t)(n + 1) * sizeof(int));
    for (i = 0; i < m; i++) head[src[i] + 1]++;
    for (i = 0; i < n; i++) head[i + 1] += head[i];
    {
        int *cur = (int *)malloc((size_t)n * sizeof(int));
        if (!cur) { free(head); free(eto); free(ew); free(pos); free(fid); free(fd); return; }
        memcpy(cur, head, (size_t)n * sizeof(int));
        for (i = 0; i < m; i++) {
            int s = src[i], p = cur[s]++;
            eto[p] = dst[i]; ew[p] = weight[i];
        }
        free(cur);
    }

    /* pos: >=0 index in flock, -1 unreached, -2 stoned OR a house with no far side */
    for (i = 0; i < n; i++) pos[i] = (head[i + 1] > head[i]) ? -1 : -2;

    k = 0;
    if (pos[source] == -1) { pos[source] = 0; fid[0] = source; fd[0] = 0.0; k = 1; }

    work   = 0.0;
    budget = 40.0 * ((double)m + (double)n) + 1.0e6;     /* see VERDICT: the guard */
#ifdef _OPENMP
    nt = omp_get_max_threads(); if (nt > 64) nt = 64; if (nt < 1) nt = 1;
#endif

    while (k > 0) {
        int j, u, e, e1;
        double du;
        if (k > 128 && work > budget) break;             /* flock too wide for one hand */
        work += (double)k;
#ifdef _OPENMP
        if (k >= 65536 && nt > 1) j = fl_argmin_par(fd, k, nt); else
#endif
        j = fl_argmin(fd, k);                            /* the leader duck */
        u  = fid[j];
        du = fd[j];
        k--;                                             /* black stone; flock closes */
        if (j != k) { fid[j] = fid[k]; fd[j] = fd[k]; pos[fid[j]] = j; }
        pos[u] = -2;
        e1 = head[u + 1];
        for (e = head[u]; e < e1; e++) {                 /* loose fresh ducks */
            int v = eto[e];
            double nd = du + ew[e];
            if (nd < dist_out[v]) {
                int p = pos[v];
                dist_out[v] = nd;
                if (p >= 0)       fd[p] = nd;            /* re-spool in place, O(1) */
                else if (p == -1) { pos[v] = k; fid[k] = v; fd[k] = nd; k++; }
            }
        }
    }

    if (k > 0) {
        int hcap = k + 1024, hsz = k;
        double *hk = (double *)malloc((size_t)hcap * sizeof(double));
        int    *hv = (int *)malloc((size_t)hcap * sizeof(int));
        if (hk && hv) {
            int t;
            memcpy(hk, fd,  (size_t)k * sizeof(double));
            memcpy(hv, fid, (size_t)k * sizeof(int));
            for (t = (hsz - 2) / 4; t >= 0; t--) sink4(hk, hv, hsz, t);
            while (hsz > 0) {
                int u = hv[0], e, e1;
                double du = hk[0];
                hsz--; hk[0] = hk[hsz]; hv[0] = hv[hsz];
                if (hsz > 0) sink4(hk, hv, hsz, 0);
                if (du > dist_out[u]) continue;          /* stale duck */
                e1 = head[u + 1];
                for (e = head[u]; e < e1; e++) {
                    int v = eto[e];
                    double nd = du + ew[e];
                    if (nd < dist_out[v]) {
                        dist_out[v] = nd;
                        if (pos[v] != -2) {              /* never spool a slack road */
                            int i2;
                            if (hsz == hcap) {
                                int nc = hcap * 2;
                                double *a = (double *)realloc(hk, (size_t)nc * sizeof(double));
                                int *b;
                                if (!a) goto heapdone;
                                hk = a;
                                b = (int *)realloc(hv, (size_t)nc * sizeof(int));
                                if (!b) goto heapdone;
                                hv = b; hcap = nc;
                            }
                            i2 = hsz++;
                            while (i2 > 0) {
                                int p2 = (i2 - 1) >> 2;
                                if (hk[p2] <= nd) break;
                                hk[i2] = hk[p2]; hv[i2] = hv[p2]; i2 = p2;
                            }
                            hk[i2] = nd; hv[i2] = v;
                        }
                    }
                }
            }
        }
    heapdone:
        free(hk); free(hv);
    }

    free(head); free(eto); free(ew); free(pos); free(fid); free(fd);
}
