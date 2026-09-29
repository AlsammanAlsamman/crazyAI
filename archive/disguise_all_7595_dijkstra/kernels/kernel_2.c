#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#if defined(__SSE2__) || defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#endif
#if !defined(__GNUC__)
#define __builtin_prefetch(a,b,c) ((void)0)
#define __builtin_ctz(x) 0
#endif

typedef struct { double d; int u; int pad; } HItem;

/* 64-byte aligned block; *raw receives the pointer to free */
static void *aal(size_t bytes, void **raw) {
    void *r = malloc(bytes + 64);
    *raw = r;
    if (!r) return NULL;
    return (void *)((((uintptr_t)r) + 63u) & ~(uintptr_t)63u);
}

/* index of minimum, or -1 if every entry is +INF : two-pass (min, then locate) */
static int argmin_pd(const double *a, int n) {
    int i = 0;
    double best = INFINITY;
#if defined(__AVX__)
    if (n >= 8) {
        __m256d m0 = _mm256_set1_pd(INFINITY), m1 = m0, m2 = m0, m3 = m0;
        for (; i + 16 <= n; i += 16) {
            m0 = _mm256_min_pd(m0, _mm256_loadu_pd(a + i));
            m1 = _mm256_min_pd(m1, _mm256_loadu_pd(a + i + 4));
            m2 = _mm256_min_pd(m2, _mm256_loadu_pd(a + i + 8));
            m3 = _mm256_min_pd(m3, _mm256_loadu_pd(a + i + 12));
        }
        for (; i + 4 <= n; i += 4) m0 = _mm256_min_pd(m0, _mm256_loadu_pd(a + i));
        m0 = _mm256_min_pd(_mm256_min_pd(m0, m1), _mm256_min_pd(m2, m3));
        {
            double b4[4];
            _mm256_storeu_pd(b4, m0);
            best = b4[0];
            if (b4[1] < best) best = b4[1];
            if (b4[2] < best) best = b4[2];
            if (b4[3] < best) best = b4[3];
        }
        for (; i < n; i++) if (a[i] < best) best = a[i];
        if (!(best < INFINITY)) return -1;
        {
            __m256d vb = _mm256_set1_pd(best);
            int j = 0;
            for (; j + 4 <= n; j += 4) {
                int msk = _mm256_movemask_pd(
                    _mm256_cmp_pd(_mm256_loadu_pd(a + j), vb, _CMP_EQ_OQ));
                if (msk) return j + (int)__builtin_ctz((unsigned)msk);
            }
            for (; j < n; j++) if (a[j] == best) return j;
            return -1;
        }
    }
#endif
    {
        int bi = -1;
        for (i = 0; i < n; i++) if (a[i] < best) { best = a[i]; bi = i; }
        return bi;
    }
}

/* ---- 4-ary lazy heap, hole-based sift, child groups on 64B lines ---- */
typedef struct { HItem *h; void *raw; int size, cap; } Heap;

static int hp_init(Heap *H, int cap) {
    void *raw;
    HItem *base;
    if (cap < 16) cap = 16;
    base = (HItem *)aal((size_t)(cap + 4) * sizeof(HItem), &raw);
    if (!base) return 0;
    H->raw = raw; H->h = base + 3; H->size = 0; H->cap = cap;
    return 1;
}
static int hp_grow(Heap *H, int maxcap) {
    void *raw;
    HItem *base;
    int nc = (H->cap > (1 << 29)) ? maxcap : H->cap * 2;
    if (nc > maxcap) nc = maxcap;
    if (nc <= H->cap) return 0;
    base = (HItem *)aal((size_t)(nc + 4) * sizeof(HItem), &raw);
    if (!base) return 0;
    memcpy(base + 3, H->h, (size_t)H->size * sizeof(HItem));
    free(H->raw);
    H->raw = raw; H->h = base + 3; H->cap = nc;
    return 1;
}
static inline void hp_push(HItem *h, int *hs, double d, int u) {
    int i = (*hs)++;
    while (i > 0) {
        int p = (i - 1) >> 2;
        if (h[p].d <= d) break;
        h[i] = h[p];
        i = p;
    }
    h[i].d = d; h[i].u = u;
}
static inline void hp_pop(HItem *h, int *hs) {
    int last = --(*hs);
    double d = h[last].d;
    int u = h[last].u, i = 0;
    for (;;) {
        int c = (i << 2) + 1;
        if (c + 3 < last) {
            double bd = h[c].d, d1 = h[c + 1].d, d2 = h[c + 2].d, d3 = h[c + 3].d;
            int b = c;
            if (d1 < bd) { bd = d1; b = c + 1; }
            if (d2 < bd) { bd = d2; b = c + 2; }
            if (d3 < bd) { bd = d3; b = c + 3; }
            if (bd >= d) break;
            h[i] = h[b]; i = b;
        } else if (c < last) {
            double bd = h[c].d;
            int b = c, k;
            for (k = c + 1; k < last; k++) if (h[k].d < bd) { bd = h[k].d; b = k; }
            if (bd >= d) break;
            h[i] = h[b]; i = b;
            break;                 /* partial group: b provably has no children */
        } else break;
    }
    h[i].d = d; h[i].u = u;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    void *roff = NULL, *red = NULL, *rew = NULL;
    int *off; int *edst; double *ew;
    double w0 = 0.0, nn;
    int uni = 1, use_bfs, use_scan, i;

    if (n <= 0) return;
    for (i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    off  = (int *)   aal((size_t)(n + 2) * sizeof(int),    &roff);
    edst = (int *)   aal((size_t)m * sizeof(int),          &red);
    ew   = (double *)aal((size_t)m * sizeof(double),       &rew);
    if (!off || !edst || !ew) { free(roff); free(red); free(rew); return; }

    /* CSR in one n+2 array: count at [u+2], prefix, scatter through off+1,
       which leaves off[u] == start(u), off[n] == m. */
    memset(off, 0, (size_t)(n + 2) * sizeof(int));
    for (i = 0; i < m; i++) off[src[i] + 2]++;
    for (i = 1; i <= n + 1; i++) off[i] += off[i - 1];
    {
        int *cur = off + 1;
        w0 = weight[0];
        for (i = 0; i < m; i++) {
            int u = src[i], p = cur[u]++;
            double w = weight[i];
            edst[p] = dst[i];
            ew[p] = w;
            uni &= (w == w0);
        }
    }

    use_bfs = uni && (w0 >= 0.0) && isfinite(w0) && ((double)n * w0 < 1.0e300);
    nn = (double)n * (double)n;
    use_scan = !use_bfs && (nn <= 24.0 * (double)m + 4096.0);

    /* ---- every sign reads the same number: nearest-first == hop order ---- */
    if (use_bfs) {
        void *rq;
        int *q = (int *)aal((size_t)n * sizeof(int), &rq);
        if (q) {
            int qh = 0, qt = 0;
            q[qt++] = source;
            while (qh < qt) {
                int u = q[qh++], e = off[u], en = off[u + 1], k, pn;
                double nd = dist_out[u] + w0;
                pn = (en - e > 32) ? e + 32 : en;
                for (k = e; k < pn; k++) __builtin_prefetch(&dist_out[edst[k]], 0, 3);
                for (; e < en; e++) {
                    int v = edst[e];
                    if (dist_out[v] == INFINITY) { dist_out[v] = nd; q[qt++] = v; }
                }
            }
            free(rq); free(roff); free(red); free(rew);
            return;
        }
    }

    /* ---- dense: scan the whole scratch-list, but with AVX2 ---- */
    if (use_scan) {
        void *ra;
        double *act = (double *)aal((size_t)n * sizeof(double), &ra);
        if (act) {
            int it;
            for (i = 0; i < n; i++) act[i] = INFINITY;
            act[source] = 0.0;
            for (it = 0; it < n; it++) {
                int u = argmin_pd(act, n), e, en;
                double du;
                if (u < 0) break;
                du = act[u];
                act[u] = INFINITY;              /* carved in stone */
                e = off[u]; en = off[u + 1];
                for (; e < en; e++) {
                    int v = edst[e];
                    double nd = du + ew[e];
                    if (nd < dist_out[v]) { dist_out[v] = nd; act[v] = nd; }
                }
            }
            free(ra); free(roff); free(red); free(rew);
            return;
        }
    }

    /* ---- sparse: 4-ary lazy heap, no done[] array ---- */
    {
        Heap H;
        int cap0 = n + 64, maxcap = m + 2, settled = 0;
        if (cap0 > maxcap) cap0 = maxcap;
        if (!hp_init(&H, cap0)) { free(roff); free(red); free(rew); return; }
        hp_push(H.h, &H.size, 0.0, source);
        while (H.size > 0) {
            double td = H.h[0].d;
            int u = H.h[0].u, e, en, k, pn;
            hp_pop(H.h, &H.size);
            if (td > dist_out[u]) continue;     /* exact stale test (w >= 0) */
            if (++settled == n) break;          /* bubble covers the question */
            e = off[u]; en = off[u + 1];
            pn = (en - e > 32) ? e + 32 : en;
            for (k = e; k < pn; k++) __builtin_prefetch(&dist_out[edst[k]], 0, 3);
            for (; e < en; e++) {
                int v = edst[e];
                double nd = td + ew[e];
                if (nd < dist_out[v]) {
                    dist_out[v] = nd;
                    if (H.size == H.cap && !hp_grow(&H, maxcap)) { e = en; break; }
                    hp_push(H.h, &H.size, nd, v);
                }
            }
        }
        free(H.raw);
    }
    free(roff); free(red); free(rew);
}
