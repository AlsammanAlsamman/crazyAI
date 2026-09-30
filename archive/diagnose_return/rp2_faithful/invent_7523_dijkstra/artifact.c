#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#define HAVE_AVX2 1
#endif

/* ================= the hooded figure's slow reading: heap Dijkstra ================= */
typedef struct { double d; int u; } HeapItem;

static void hf_push(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++;
    h[i].d = d; h[i].u = u;
    while (i > 0) { int p = (i - 1) >> 1; if (h[p].d <= h[i].d) break;
                    HeapItem t = h[p]; h[p] = h[i]; h[i] = t; i = p; }
}
static HeapItem hf_pop(HeapItem *h, int *hs) {
    HeapItem top = h[0]; (*hs)--; h[0] = h[*hs];
    int i = 0;
    for (;;) {
        int l = 2 * i + 1, r = l + 1, s = i;
        if (l < *hs && h[l].d < h[s].d) s = l;
        if (r < *hs && h[r].d < h[s].d) s = r;
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t; i = s;
    }
    return top;
}
static void slow_reading(int n, int m, const int *src, const int *dst,
                         const double *weight, int source, double *dist_out)
{
    int *off  = (int *)calloc((size_t)n + 1, sizeof(int));
    int *adj  = (int *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *aw = (double *)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    int *fill = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    memcpy(fill, off, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) { int p = fill[src[i]]++; adj[p] = dst[i]; aw[p] = weight[i]; }
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;
    char *done = (char *)calloc((size_t)n, 1);
    HeapItem *h = (HeapItem *)malloc((size_t)(m + 2) * sizeof(HeapItem));
    int hs = 0; hf_push(h, &hs, 0.0, source);
    while (hs > 0) {
        HeapItem t = hf_pop(h, &hs);
        int u = t.u;
        if (done[u]) continue;
        done[u] = 1;
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = adj[e]; double nd = dist_out[u] + aw[e];
            if (nd < dist_out[v]) { dist_out[v] = nd; hf_push(h, &hs, nd, v); }
        }
    }
    free(off); free(adj); free(aw); free(fill); free(done); free(h);
}

/* ============================== the native's land ================================= */
void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    if (source < 0 || source >= n) { for (int i = 0; i < n; i++) dist_out[i] = INFINITY; return; }
    if (m <= 0) { for (int i = 0; i < n; i++) dist_out[i] = INFINITY; dist_out[source] = 0.0; return; }
    if (n < 64) { slow_reading(n, m, src, dst, weight, source, dist_out); return; }  /* too small a land */

    const int C = 4;
    int nch  = (n + C - 1) / C;
    int npad = nch * C;

    /* ---- roads out of every place (runners), and the shortest road in the land ---- */
    int *foff  = (int *)calloc((size_t)n + 2, sizeof(int));
    int *indeg = (int *)calloc((size_t)npad + 1, sizeof(int));
    double wmin = INFINITY;
    for (int i = 0; i < m; i++) {
        foff[src[i] + 1]++;
        indeg[dst[i]]++;
        if (weight[i] < wmin) wmin = weight[i];
    }
    if (!(wmin >= 0.0)) wmin = 0.0;
    for (int i = 0; i < n; i++) foff[i + 1] += foff[i];
    int *fadj = (int *)malloc((size_t)m * sizeof(int));
    int *fill = (int *)malloc((size_t)n * sizeof(int));
    memcpy(fill, foff, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) fadj[fill[src[i]]++] = dst[i];

    /* ---- the heap of sticks at every place: are the heaps even or ragged? ---- */
    int *clen = (int *)malloc((size_t)nch * sizeof(int));
    long long total = 0;
    for (int c = 0; c < nch; c++) {
        int b = c * C, mx = 0;
        for (int l = 0; l < C; l++) { int d = indeg[b + l]; if (d > mx) mx = d; }
        clen[c] = mx; total += (long long)mx * C;
    }
    int use_ell = 0;
#ifdef HAVE_AVX2
    if (total <= 2LL * m + 8LL * n) use_ell = 1;      /* even enough heaps -> lane-wise throw */
#endif

    size_t *coff = NULL; int *esrc = NULL; double *ew = NULL;
    int *roff2 = NULL, *rsrc = NULL; double *rw = NULL;

    if (use_ell) {
        coff = (size_t *)malloc(((size_t)nch + 1) * sizeof(size_t));
        coff[0] = 0;
        for (int c = 0; c < nch; c++) coff[c + 1] = coff[c] + (size_t)clen[c] * (size_t)C;
        size_t tot = coff[nch];
        esrc = (int *)malloc((tot ? tot : 1) * sizeof(int));
        ew   = (double *)malloc((tot ? tot : 1) * sizeof(double));
        for (int c = 0; c < nch; c++) {                 /* neutral padding slivers */
            int b = c * C, L = clen[c]; size_t base = coff[c];
            for (int l = 0; l < C; l++)
                for (int k = indeg[b + l]; k < L; k++) {
                    size_t idx = base + (size_t)C * (size_t)k + (size_t)l;
                    esrc[idx] = 0; ew[idx] = INFINITY;
                }
        }
        int *rfill = (int *)calloc((size_t)npad, sizeof(int));
        for (int i = 0; i < m; i++) {
            int v = dst[i], c = v >> 2, l = v & 3, k = rfill[v]++;
            size_t idx = coff[c] + (size_t)4 * (size_t)k + (size_t)l;
            esrc[idx] = src[i]; ew[idx] = weight[i];
        }
        free(rfill);
    } else {
        roff2 = (int *)malloc(((size_t)npad + 1) * sizeof(int));
        roff2[0] = 0;
        for (int i = 0; i < npad; i++) roff2[i + 1] = roff2[i] + indeg[i];
        rsrc = (int *)malloc((size_t)m * sizeof(int));
        rw   = (double *)malloc((size_t)m * sizeof(double));
        int *rfill = (int *)malloc((size_t)npad * sizeof(int));
        memcpy(rfill, roff2, (size_t)npad * sizeof(int));
        for (int i = 0; i < m; i++) { int v = dst[i], p = rfill[v]++; rsrc[p] = src[i]; rw[p] = weight[i]; }
        free(rfill);
    }

    /* ---- embers, garrisons, runners ---- */
    double *D = (double *)malloc((size_t)npad * sizeof(double));
    for (int i = 0; i < npad; i++) D[i] = INFINITY;
    D[source] = 0.0;
    char *sealed   = (char *)calloc((size_t)npad, 1);
    int  *unsealed = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) unsealed[i] = i;
    int nuns = n, pending = 0;
    int *fr = (int *)malloc((size_t)npad * sizeof(int));
    int *nx = (int *)malloc((size_t)npad * sizeof(int));
    int *chunks = (int *)malloc((size_t)nch * sizeof(int));
    char *inq   = (char *)calloc((size_t)nch, 1);
    int nf = 1; fr[0] = source;

    long long work = 0, budget = 16LL * m + 32LL * n;   /* the fuel counter */
    int rounds = 0, bailed = 0;

    while (nf > 0) {
        rounds++;
        long long fedges = 0;
        for (int i = 0; i < nf; i++) { int u = fr[i]; fedges += foff[u + 1] - foff[u]; }

        int nc = 0;
        if (fedges * 4 >= (long long)m) {          /* runners on every road anyway: whole land throws */
            for (int c = 0; c < nch; c++) chunks[nc++] = c;
            work += nf;
        } else {                                    /* send runners, mark the receiving hamlets */
            for (int i = 0; i < nf; i++) {
                int u = fr[i];
                for (int e = foff[u]; e < foff[u + 1]; e++) {
                    int v = fadj[e];
                    if (sealed[v]) continue;        /* stick dropped by the wayside */
                    int c = v >> 2;
                    if (!inq[c]) { inq[c] = 1; chunks[nc++] = c; }
                }
            }
            work += fedges;
        }

        int nn = 0;
        if (use_ell) {
#ifdef HAVE_AVX2
            for (int j = 0; j < nc; j++) {
                int c = chunks[j]; inq[c] = 0;
                int L = clen[c];
                if (L == 0) continue;
                const int    *es = esrc + coff[c];
                const double *ws = ew   + coff[c];
                __m256d acc = _mm256_set1_pd(INFINITY);
                for (int k = 0; k < L; k++, es += 4, ws += 4) {   /* one sliver from every heap */
                    __m128i ix  = _mm_loadu_si128((const __m128i *)es);
                    __m256d du  = _mm256_i32gather_pd(D, ix, 8);
                    __m256d wv  = _mm256_loadu_pd(ws);
                    acc = _mm256_min_pd(acc, _mm256_add_pd(du, wv));
                }
                int b = c * 4;
                __m256d cur = _mm256_loadu_pd(D + b);
                __m256d nd  = _mm256_min_pd(acc, cur);
                int mask = _mm256_movemask_pd(_mm256_cmp_pd(nd, cur, _CMP_LT_OQ));
                if (mask) {
                    _mm256_storeu_pd(D + b, nd);
                    if (mask & 1) nx[nn++] = b;
                    if (mask & 2) nx[nn++] = b + 1;
                    if (mask & 4) nx[nn++] = b + 2;
                    if (mask & 8) nx[nn++] = b + 3;
                }
                work += 4LL * L;
            }
#endif
        } else {
            for (int j = 0; j < nc; j++) {
                int c = chunks[j]; inq[c] = 0;
                int b = c * 4;
                for (int l = 0; l < 4; l++) {
                    int v = b + l;
                    int e = roff2[v], en = roff2[v + 1];
                    if (e == en) continue;
                    work += (en - e);
                    double a0 = INFINITY, a1 = INFINITY, a2 = INFINITY, a3 = INFINITY;
                    for (; e + 3 < en; e += 4) {
                        double t0 = D[rsrc[e]]     + rw[e];
                        double t1 = D[rsrc[e + 1]] + rw[e + 1];
                        double t2 = D[rsrc[e + 2]] + rw[e + 2];
                        double t3 = D[rsrc[e + 3]] + rw[e + 3];
                        if (t0 < a0) a0 = t0;
                        if (t1 < a1) a1 = t1;
                        if (t2 < a2) a2 = t2;
                        if (t3 < a3) a3 = t3;
                    }
                    for (; e < en; e++) { double t = D[rsrc[e]] + rw[e]; if (t < a0) a0 = t; }
                    if (a1 < a0) a0 = a1;
                    if (a3 < a2) a2 = a3;
                    if (a2 < a0) a0 = a2;
                    if (a0 < D[v]) { D[v] = a0; nx[nn++] = v; }
                }
            }
        }

        /* ---- the garrison rides in ---- */
        if (nn > 0) {
            double mu = INFINITY;
            for (int i = 0; i < nn; i++) { double d = D[nx[i]]; if (d < mu) mu = d; }
            double thr = mu + wmin;                 /* nothing shorter can still arrive */
            pending += nn;
            if ((long long)pending * 8 >= (long long)nuns) {
                int k2 = 0;
                for (int i = 0; i < nuns; i++) {
                    int v = unsealed[i];
                    if (D[v] <= thr) sealed[v] = 1; else unsealed[k2++] = v;
                }
                work += nuns; nuns = k2; pending = 0;
            }
        }

        { int *t = fr; fr = nx; nx = t; }
        nf = nn;
        if (nf > 0 && (work > budget || rounds >= 2048)) { bailed = 1; break; }
    }

    if (!bailed) for (int i = 0; i < n; i++) dist_out[i] = D[i];   /* read the land aloud */

    free(foff); free(indeg); free(fadj); free(fill); free(clen);
    if (use_ell) { free(coff); free(esrc); free(ew); }
    else         { free(roff2); free(rsrc); free(rw); }
    free(D); free(sealed); free(unsealed); free(fr); free(nx); free(chunks); free(inq);

    if (bailed) slow_reading(n, m, src, dst, weight, source, dist_out);  /* long thin land */
}
