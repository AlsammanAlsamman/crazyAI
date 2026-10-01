#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* one dying note: 16 bytes exactly, so four of them fill one cache line */
typedef struct { double d; int u; int pad; } kd_item;

/* ---------------- the roost: lazy 4-ary heap, no done[] array ------------- */
static void kd_push(kd_item *restrict H, int *restrict hsp, double d, int u)
{
    int i = (*hsp)++;
    while (i > 0) {
        int p = (i - 1) >> 2;
        if (H[p].d <= d) break;
        H[i] = H[p];
        i = p;
    }
    H[i].d = d; H[i].u = u;
}

static void kd_pop(kd_item *restrict H, int *restrict hsp)
{
    int hs = --(*hsp);
    if (hs <= 0) return;
    double d = H[hs].d; int u = H[hs].u;
    int i = 0;
    for (;;) {
        int c = (i << 2) + 1;
        if (c >= hs) break;
        int last = c + 4; if (last > hs) last = hs;
        int b = c; double bk = H[c].d;
        for (int k = c + 1; k < last; k++) { double kk = H[k].d; if (kk < bk) { bk = kk; b = k; } }
        if (bk >= d) break;
        H[i] = H[b];
        i = b;
    }
    H[i].d = d; H[i].u = u;
}

static void kd_roost(int n, const int *restrict off, const int *restrict edst,
                     const double *restrict ew, int source,
                     double *restrict dist, kd_item *restrict H)
{
    for (int i = 0; i < n; i++) dist[i] = INFINITY;
    dist[source] = 0.0;
    int hs = 0;
    kd_push(H, &hs, 0.0, source);
    while (hs > 0) {
        double du = H[0].d;
        int u = H[0].u;
        kd_pop(H, &hs);
        if (du > dist[u]) continue;              /* a louder note, left to rot */
        int e = off[u], ee = off[u + 1];
        for (; e < ee; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist[v]) { dist[v] = nd; kd_push(H, &hs, nd, v); }
        }
    }
}

/* -------- the flat lattice: one fixed slot per place, read four at once ---- */
static void kd_lattice(int n, const int *restrict off, const int *restrict edst,
                       const double *restrict ew, int source,
                       double *restrict dist,
                       double *restrict actv, int *restrict actid, int *restrict posn)
{
    for (int i = 0; i < n; i++) { dist[i] = INFINITY; actv[i] = INFINITY; actid[i] = i; posn[i] = i; }
    for (int i = n; i < n + 16; i++) actv[i] = INFINITY;
    int cnt = n;
    dist[source] = 0.0;
    actv[posn[source]] = 0.0;

    while (cnt > 0) {
        /* pass 1: the quietest unsettled note */
        double best = INFINITY;
        int i = 0;
#if defined(__AVX2__)
        {
            __m256d m0 = _mm256_set1_pd(INFINITY), m1 = m0, m2 = m0, m3 = m0;
            for (; i + 16 <= cnt; i += 16) {
                m0 = _mm256_min_pd(m0, _mm256_loadu_pd(actv + i));
                m1 = _mm256_min_pd(m1, _mm256_loadu_pd(actv + i + 4));
                m2 = _mm256_min_pd(m2, _mm256_loadu_pd(actv + i + 8));
                m3 = _mm256_min_pd(m3, _mm256_loadu_pd(actv + i + 12));
            }
            m0 = _mm256_min_pd(_mm256_min_pd(m0, m1), _mm256_min_pd(m2, m3));
            double t[4];
            _mm256_storeu_pd(t, m0);
            best = t[0];
            if (t[1] < best) best = t[1];
            if (t[2] < best) best = t[2];
            if (t[3] < best) best = t[3];
        }
#endif
        for (; i < cnt; i++) if (actv[i] < best) best = actv[i];
        if (!(best < INFINITY)) break;           /* only desert left */

        /* pass 2: which slot holds it */
        int j = -1;
        i = 0;
#if defined(__AVX2__)
        {
            __m256d bv = _mm256_set1_pd(best);
            for (; i + 4 <= cnt; i += 4) {
                int msk = _mm256_movemask_pd(
                              _mm256_cmp_pd(_mm256_loadu_pd(actv + i), bv, _CMP_EQ_OQ));
                if (msk) { j = i + (int)__builtin_ctz((unsigned)msk); break; }
            }
        }
#endif
        if (j < 0) { for (; i < cnt; i++) if (actv[i] == best) { j = i; break; } }

        int u = actid[j];
        cnt--;                                   /* settled: never read again */
        actv[j] = actv[cnt]; actid[j] = actid[cnt]; posn[actid[j]] = j;
        posn[u] = -1;
        dist[u] = best;

        int e = off[u], ee = off[u + 1];
        for (; e < ee; e++) {
            int v = edst[e];
            double nd = best + ew[e];
            /* a settled v has dist[v] <= best <= nd, so this guard also
               guarantees a settled board is never touched twice */
            if (nd < dist[v]) { dist[v] = nd; actv[posn[v]] = nd; }
        }
    }
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    if (m < 0) m = 0;
    size_t sn = (size_t)n, sm = (size_t)m;

    /* --- pacing the land: boards against roads, before a single bird flies --- */
    double dn = (double)n, dm = (double)m;
    double lg = log2(dn > 2.0 ? dn : 2.0);
    double avgdeg = (dn > 0.0) ? dm / dn : 0.0;
    double notes = dn * (1.0 + log(1.0 + avgdeg));      /* expected carvings */
    if (notes > dm) notes = dm;
    double roost_cost   = notes * (6.0 + 2.0 * lg) + 2.0 * dn;
    double lattice_cost = 0.18 * dn * dn + dm;
    int use_lattice = (lattice_cost <= roost_cost);

    /* --- one block for the roads (CSR) --- */
    size_t csr_bytes = sm * sizeof(double) + (sm + 2 * (sn + 1) + 2) * sizeof(int);
    char *csr = (char *)malloc(csr_bytes + 64);
    if (!csr) {
        for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
        dist_out[source] = 0.0;
        return;
    }
    double *ew = (double *)csr;
    int *ip = (int *)(ew + sm);
    int *off = ip; ip += sn + 1;
    int *cur = ip; ip += sn + 1;
    int *edst = ip;

    memset(off, 0, (sn + 1) * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    memcpy(cur, off, (sn + 1) * sizeof(int));
    for (int i = 0; i < m; i++) {
        int p = cur[src[i]]++;
        edst[p] = dst[i]; ew[p] = weight[i];
    }

    if (use_lattice) {
        char *blk = (char *)malloc((sn + 16) * sizeof(double) + 2 * sn * sizeof(int) + 64);
        if (blk) {
            double *actv = (double *)blk;
            int *q = (int *)(actv + sn + 16);
            kd_lattice(n, off, edst, ew, source, dist_out, actv, q, q + sn);
            free(blk);
            free(csr);
            return;
        }
        use_lattice = 0;                          /* fall through to the roost */
    }

    {
        char *raw = (char *)malloc((sm + 12) * sizeof(kd_item) + 64);
        if (!raw) {
            for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
            dist_out[source] = 0.0;
            free(csr);
            return;
        }
        kd_item *base = (kd_item *)(((uintptr_t)raw + 63) & ~(uintptr_t)63);
        kd_item *H = base + 3;   /* children of i are 4i+1..4i+4 -> one aligned line */
        kd_roost(n, off, edst, ew, source, dist_out, H);
        free(raw);
    }
    free(csr);
}
