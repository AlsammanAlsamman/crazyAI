#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

typedef struct { double w; int v; int pad; } Edge;   /* one road: one 16B stream */
typedef struct { double d; int u; int pad; } HItem;  /* one rung of the keeper's ladder */

/* the hand hovering: argmin over the contiguous flock, then the shuttle back
   to the leader's landing (the min hides among its long cousins in a lane). */
static int flock_leader(const double *restrict fd, int fs)
{
#if defined(__AVX2__)
    if (fs >= 16) {
        __m256d a0 = _mm256_loadu_pd(fd + 0);
        __m256d a1 = _mm256_loadu_pd(fd + 4);
        __m256d a2 = _mm256_loadu_pd(fd + 8);
        __m256d a3 = _mm256_loadu_pd(fd + 12);
        int i = 16;
        for (; i + 16 <= fs; i += 16) {
            a0 = _mm256_min_pd(a0, _mm256_loadu_pd(fd + i + 0));
            a1 = _mm256_min_pd(a1, _mm256_loadu_pd(fd + i + 4));
            a2 = _mm256_min_pd(a2, _mm256_loadu_pd(fd + i + 8));
            a3 = _mm256_min_pd(a3, _mm256_loadu_pd(fd + i + 12));
        }
        {
            __m256d a = _mm256_min_pd(_mm256_min_pd(a0, a1), _mm256_min_pd(a2, a3));
            __m128d h = _mm_min_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd(a, 1));
            h = _mm_min_sd(h, _mm_unpackhi_pd(h, h));
            double best = _mm_cvtsd_f64(h);
            int tail = -1, t, k;
            for (t = i; t < fs; t++) if (fd[t] < best) { best = fd[t]; tail = t; }
            if (tail >= 0) return tail;
            {
                __m256d bv = _mm256_set1_pd(best);
                for (k = 0; k + 16 <= i; k += 16) {
                    __m256d c0 = _mm256_cmp_pd(_mm256_loadu_pd(fd+k+0),  bv, _CMP_EQ_OQ);
                    __m256d c1 = _mm256_cmp_pd(_mm256_loadu_pd(fd+k+4),  bv, _CMP_EQ_OQ);
                    __m256d c2 = _mm256_cmp_pd(_mm256_loadu_pd(fd+k+8),  bv, _CMP_EQ_OQ);
                    __m256d c3 = _mm256_cmp_pd(_mm256_loadu_pd(fd+k+12), bv, _CMP_EQ_OQ);
                    __m256d any = _mm256_or_pd(_mm256_or_pd(c0, c1), _mm256_or_pd(c2, c3));
                    if (_mm256_movemask_pd(any)) {
                        for (t = k; t < k + 16; t++) if (fd[t] == best) return t;
                    }
                }
                for (; k < fs; k++) if (fd[k] == best) return k;
                return 0; /* unreachable: min_pd selects an existing element exactly */
            }
        }
    }
#endif
    {
        int b = 0, t; double bv = fd[0];
        for (t = 1; t < fs; t++) if (fd[t] < bv) { bv = fd[t]; b = t; }
        return b;
    }
}

/* the keeper's ladder: 4-ary heap, lazy deletion (fallback regime only) */
static void h_siftdown(HItem *restrict h, int hs, int i)
{
    double d = h[i].d; int u = h[i].u;
    for (;;) {
        int c = 4 * i + 1, e, b, j; double bd;
        if (c >= hs) break;
        e = c + 4; if (e > hs) e = hs;
        b = c; bd = h[c].d;
        for (j = c + 1; j < e; j++) if (h[j].d < bd) { bd = h[j].d; b = j; }
        if (bd >= d) break;
        h[i] = h[b]; i = b;
    }
    h[i].d = d; h[i].u = u;
}

static void h_push(HItem *restrict h, int *hs, double d, int u)
{
    int i = (*hs)++;
    while (i > 0) {
        int p = (i - 1) >> 2;
        if (h[p].d <= d) break;
        h[i] = h[p]; i = p;
    }
    h[i].d = d; h[i].u = u;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    double *restrict D = dist_out;
    int *off = NULL, *pos = NULL, *fn = NULL;
    double *fd = NULL;
    Edge *eg = NULL;
    HItem *heap = NULL;
    unsigned char *done = NULL;
    int fs = 0, hs = 0, T, i;

    if (n <= 0) return;
    for (i = 0; i < n; i++) D[i] = INFINITY;
    if (source < 0 || source >= n) return;
    D[source] = 0.0;                      /* the white stone, written first */
    if (m <= 0) return;

    /* ---- the grid of roads: one 16B stream, in-place counting scatter ---- */
    off = (int *)malloc((size_t)(n + 2) * sizeof(int));
    eg  = (Edge *)malloc((size_t)m * sizeof(Edge));
    if (!off || !eg) { free(off); free(eg); return; }
    memset(off, 0, (size_t)(n + 2) * sizeof(int));
    for (i = 0; i < m; i++) off[src[i] + 2]++;
    for (i = 2; i <= n + 1; i++) off[i] += off[i - 1];
    for (i = 0; i < m; i++) {
        int u = src[i], p = off[u + 1]++;
        eg[p].w = weight[i]; eg[p].v = dst[i];
    }
    /* off[u] .. off[u+1] is now node u's out-edge range */

    done = (unsigned char *)calloc((size_t)n, 1);
    pos  = (int *)malloc((size_t)n * sizeof(int));
    fn   = (int *)malloc((size_t)n * sizeof(int));
    fd   = (double *)malloc((size_t)n * sizeof(double));
    if (!done || !pos || !fn || !fd) goto cleanup;
    memset(pos, 0xFF, (size_t)n * sizeof(int));   /* -1: no duck on this corner */

    /* the first duck, unless the traveler's corner has no far side */
    if (off[source + 1] > off[source]) { pos[source] = 0; fn[0] = source; fd[0] = 0.0; fs = 1; }

    /* how many ducks one hand can sweep while the keeper climbs once.
       clamped to n, so dense grids and small grids never call the keeper. */
    {
        long long Tll = 512LL + 32LL * (long long)(m / n);
        if (Tll > (long long)n) Tll = (long long)n;
        T = (int)Tll;
    }

    /* ================= the hand: flock sweep, no priority structure ================= */
    while (fs > 0) {
        int k, u, e, e1; double du;

        if (fs > T) {                      /* wrong regime: hand the flock to the ladder */
            heap = (HItem *)malloc(((size_t)n + (size_t)m + 2) * sizeof(HItem));
            if (heap) {
                for (i = 0; i < fs; i++) { heap[i].d = fd[i]; heap[i].u = fn[i]; }
                hs = fs; fs = 0;
                for (i = (hs - 2) / 4; i >= 0; i--) h_siftdown(heap, hs, i);
                break;
            }
            T = n;                         /* no ladder to be had: keep hovering */
        }

        k = flock_leader(fd, fs);
        u = fn[k]; du = fd[k];
        fs--;
        if (k != fs) { fn[k] = fn[fs]; fd[k] = fd[fs]; pos[fn[k]] = k; }
        pos[u] = -1;
        done[u] = 1;                       /* the black stone */

        e1 = off[u + 1];
        for (e = off[u]; e < e1; e++) {
            const Edge *restrict E = eg;
            int v = E[e].v;
            double nd = du + E[e].w;
            if (nd < D[v]) {               /* the only consultation there is */
                D[v] = nd;
                if (off[v + 1] > off[v]) { /* a road to a house with no far side hangs slack */
                    int p = pos[v];
                    if (p >= 0) fd[p] = nd;
                    else { pos[v] = fs; fn[fs] = v; fd[fs] = nd; fs++; }
                }
            }
        }
    }

    /* ================= the ladder: 4-ary heap fallback for the vast thin grid ========= */
    if (heap) {
        while (hs > 0) {
            int u, e, e1; double du;
            u = heap[0].u;
            hs--;
            if (hs > 0) { heap[0] = heap[hs]; h_siftdown(heap, hs, 0); }
            if (done[u]) continue;
            done[u] = 1;
            du = D[u];
            e1 = off[u + 1];
            for (e = off[u]; e < e1; e++) {
                const Edge *restrict E = eg;
                int v = E[e].v;
                double nd = du + E[e].w;
                if (nd < D[v]) {
                    D[v] = nd;
                    if (off[v + 1] > off[v] && !done[v]) h_push(heap, &hs, nd, v);
                }
            }
        }
    }

cleanup:
    free(off); free(eg); free(done); free(pos); free(fn); free(fd); free(heap);
}
