#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* one road: 16 bytes, single stream -> one load gives far side + centimetres */
typedef struct { double w; int v; int pad; } DuckRoad;
typedef struct { double d; int u; } HItem;

static void hpush(HItem *h, int *hs, double d, int u) {
    int i = (*hs)++; h[i].d = d; h[i].u = u;
    while (i > 0) { int p = (i - 1) >> 1; if (h[p].d <= h[i].d) break;
        HItem t = h[p]; h[p] = h[i]; h[i] = t; i = p; }
}
static HItem hpop(HItem *h, int *hs) {
    HItem top = h[0]; (*hs)--; h[0] = h[*hs];
    int i = 0;
    for (;;) { int l = 2*i+1, r = l+1, s = i;
        if (l < *hs && h[l].d < h[s].d) s = l;
        if (r < *hs && h[r].d < h[s].d) s = r;
        if (s == i) break;
        HItem t = h[s]; h[s] = h[i]; h[i] = t; i = s; }
    return top;
}

/* THE WALK: hand hovering along the flock, thread against thread.
   No early exit -- every shot is traced, then we shuttle back to the landing. */
static inline int duck_leader(const double *restrict cd, int nc)
{
    int best = 0, i = 0;
    double bv = cd[0];
#if defined(__AVX2__)
    if (nc >= 16) {
        const __m256d INFV = _mm256_set1_pd(INFINITY);
        __m256d m0 = INFV, m1 = INFV, m2 = INFV, m3 = INFV;
        __m256d p0 = _mm256_setr_pd(0.0, 1.0, 2.0, 3.0);
        __m256d p1 = _mm256_setr_pd(4.0, 5.0, 6.0, 7.0);
        __m256d p2 = _mm256_setr_pd(8.0, 9.0, 10.0, 11.0);
        __m256d p3 = _mm256_setr_pd(12.0, 13.0, 14.0, 15.0);
        __m256d k0 = p0, k1 = p1, k2 = p2, k3 = p3;
        const __m256d step = _mm256_set1_pd(16.0);
        for (; i + 16 <= nc; i += 16) {
            __m256d a0 = _mm256_loadu_pd(cd + i);
            __m256d a1 = _mm256_loadu_pd(cd + i + 4);
            __m256d a2 = _mm256_loadu_pd(cd + i + 8);
            __m256d a3 = _mm256_loadu_pd(cd + i + 12);
            __m256d c0 = _mm256_cmp_pd(a0, m0, _CMP_LT_OQ);
            __m256d c1 = _mm256_cmp_pd(a1, m1, _CMP_LT_OQ);
            __m256d c2 = _mm256_cmp_pd(a2, m2, _CMP_LT_OQ);
            __m256d c3 = _mm256_cmp_pd(a3, m3, _CMP_LT_OQ);
            m0 = _mm256_blendv_pd(m0, a0, c0); k0 = _mm256_blendv_pd(k0, p0, c0);
            m1 = _mm256_blendv_pd(m1, a1, c1); k1 = _mm256_blendv_pd(k1, p1, c1);
            m2 = _mm256_blendv_pd(m2, a2, c2); k2 = _mm256_blendv_pd(k2, p2, c2);
            m3 = _mm256_blendv_pd(m3, a3, c3); k3 = _mm256_blendv_pd(k3, p3, c3);
            p0 = _mm256_add_pd(p0, step); p1 = _mm256_add_pd(p1, step);
            p2 = _mm256_add_pd(p2, step); p3 = _mm256_add_pd(p3, step);
        }
        double vb[16], kb[16];
        _mm256_storeu_pd(vb,      m0); _mm256_storeu_pd(vb + 4,  m1);
        _mm256_storeu_pd(vb + 8,  m2); _mm256_storeu_pd(vb + 12, m3);
        _mm256_storeu_pd(kb,      k0); _mm256_storeu_pd(kb + 4,  k1);
        _mm256_storeu_pd(kb + 8,  k2); _mm256_storeu_pd(kb + 12, k3);
        bv = vb[0]; best = (int)kb[0];
        for (int t = 1; t < 16; t++) {
            int idx = (int)kb[t];
            if (vb[t] < bv || (vb[t] == bv && idx < best)) { bv = vb[t]; best = idx; }
        }
    }
#endif
    for (; i < nc; i++) if (cd[i] < bv) { bv = cd[i]; best = i; }
    return best;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* ---- the grid of roads, in one stream ---- */
    int *off = (int *)malloc(((size_t)n + 1) * sizeof(int));
    int *cur = (int *)malloc((size_t)n * sizeof(int));
    DuckRoad *E = (DuckRoad *)malloc((size_t)m * sizeof(DuckRoad));
    int *cand = (int *)malloc((size_t)n * sizeof(int));
    int *slot = (int *)malloc((size_t)n * sizeof(int));
    double *cd = (double *)malloc(((size_t)n + 16) * sizeof(double));
    if (!off || !cur || !E || !cand || !slot || !cd) {   /* never expected */
        free(off); free(cur); free(E); free(cand); free(slot); free(cd);
        for (int it = 0; it < n; it++) {                 /* allocation-free last resort */
            int ch = 0;
            for (int e = 0; e < m; e++) {
                double du = dist_out[src[e]];
                if (du < INFINITY) { double nd = du + weight[e];
                    if (nd < dist_out[dst[e]]) { dist_out[dst[e]] = nd; ch = 1; } }
            }
            if (!ch) break;
        }
        return;
    }
    memset(off, 0, ((size_t)n + 1) * sizeof(int));
    for (int e = 0; e < m; e++) off[src[e] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    memcpy(cur, off, (size_t)n * sizeof(int));
    for (int e = 0; e < m; e++) {
        int u = src[e], p = cur[u]++;
        E[p].v = dst[e]; E[p].w = weight[e]; E[p].pad = 0;
    }
    free(cur);

    for (int i = 0; i < n; i++) slot[i] = -1;

    /* ---- loose the first duck from the white stone ---- */
    int nc = 1;
    cand[0] = source; cd[0] = 0.0; slot[source] = 0;

    const double *restrict W = NULL; (void)W;
    double *restrict dist = dist_out;
    const DuckRoad *restrict Er = E;
    const int *restrict offr = off;

    /* ---- the regime sense: walk length vs. threads to lay ---- */
    double logn = log2((double)n + 2.0);
    double avg_deg = (double)m / (double)n;
    double walk_thresh = 4.0 * (1.0 + avg_deg) * logn;
    const int WIN = 256;
    long long win_scanned = 0;
    int win_rounds = 0, allow_switch = (n > 4096);
    HItem *heap = NULL;

    while (nc > 0) {
        int nc_scan = nc;
        int i = duck_leader(cd, nc);
        int u = cand[i];
        double du = cd[i];

        /* black stone: out of the flock, thread final */
        nc--;
        if (i != nc) { int w2 = cand[nc]; cand[i] = w2; cd[i] = cd[nc]; slot[w2] = i; }
        slot[u] = -1;

        /* fresh ducks along its roads */
        int e = offr[u], ee = offr[u + 1];
        for (; e < ee; e++) {
            int v = Er[e].v;
            double nd = du + Er[e].w;
            if (nd < dist[v]) {                 /* also discards every stoned far side */
                dist[v] = nd;
                int s = slot[v];
                if (s >= 0) cd[s] = nd;
                else { cand[nc] = v; cd[nc] = nd; slot[v] = nc; nc++; }
            }
        }

        if (allow_switch) {
            win_scanned += nc_scan;
            if (++win_rounds >= WIN) {
                if ((double)win_scanned > walk_thresh * (double)win_rounds) {
                    heap = (HItem *)malloc(((size_t)m + (size_t)nc + 2) * sizeof(HItem));
                    if (heap) break;            /* pour the flock into nested pens */
                    allow_switch = 0;
                }
                win_scanned = 0; win_rounds = 0;
            }
        }
    }

    /* ---- other regime: nested pens, resumed from the same stones ---- */
    if (heap) {
        char *done = (char *)calloc((size_t)n, 1);
        if (done) {
            int hs = 0;
            for (int t = 0; t < nc; t++) hpush(heap, &hs, cd[t], cand[t]);
            while (hs > 0) {
                HItem top = hpop(heap, &hs);
                int u = top.u;
                if (done[u]) continue;
                done[u] = 1;
                double du = dist[u];
                for (int e = offr[u], ee = offr[u + 1]; e < ee; e++) {
                    int v = Er[e].v;
                    double nd = du + Er[e].w;
                    if (nd < dist[v]) { dist[v] = nd; hpush(heap, &hs, nd, v); }
                }
            }
            free(done);
        } else {                                 /* pens refused: keep walking */
            while (nc > 0) {
                int i = duck_leader(cd, nc);
                int u = cand[i]; double du = cd[i];
                nc--;
                if (i != nc) { int w2 = cand[nc]; cand[i] = w2; cd[i] = cd[nc]; slot[w2] = i; }
                slot[u] = -1;
                for (int e = offr[u], ee = offr[u + 1]; e < ee; e++) {
                    int v = Er[e].v; double nd = du + Er[e].w;
                    if (nd < dist[v]) { dist[v] = nd;
                        int s = slot[v];
                        if (s >= 0) cd[s] = nd;
                        else { cand[nc] = v; cd[nc] = nd; slot[v] = nc; nc++; } }
                }
            }
        }
        free(heap);
    }

    free(off); free(E); free(cand); free(slot); free(cd);
}
