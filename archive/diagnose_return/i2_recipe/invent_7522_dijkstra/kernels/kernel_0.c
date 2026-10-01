/* Single-source shortest distances, built literally from the duck recipe.
   Contract: void kernel(int n, int m, const int *src, const int *dst,
                         const double *weight, int source, double *dist_out); */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

/* --- step 3 engine: one sweep, hand hovering, holding only the shortest ---
   Returns the slot of the minimum of a[0..nact), nact >= 1. Values and their
   indices ride together in vector lanes (index kept as a double; exact for
   any n a machine can allocate). */
static int drs_sweep(const double *restrict a, int nact)
{
    int i = 0, best = 0;
    double bv = a[0];
#if defined(__AVX2__)
    if (nact >= 16) {
        __m256d m0 = _mm256_loadu_pd(a + 0);
        __m256d m1 = _mm256_loadu_pd(a + 4);
        __m256d m2 = _mm256_loadu_pd(a + 8);
        __m256d m3 = _mm256_loadu_pd(a + 12);
        __m256d k0 = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
        __m256d k1 = _mm256_set_pd(7.0, 6.0, 5.0, 4.0);
        __m256d k2 = _mm256_set_pd(11.0, 10.0, 9.0, 8.0);
        __m256d k3 = _mm256_set_pd(15.0, 14.0, 13.0, 12.0);
        __m256d j0 = k0, j1 = k1, j2 = k2, j3 = k3;
        const __m256d bump = _mm256_set1_pd(16.0);
        double vv[4], kk[4];
        __m256d c;
        int t;

        for (i = 16; i + 16 <= nact; i += 16) {
            __m256d v0, v1, v2, v3, c0, c1, c2, c3;
            j0 = _mm256_add_pd(j0, bump);
            j1 = _mm256_add_pd(j1, bump);
            j2 = _mm256_add_pd(j2, bump);
            j3 = _mm256_add_pd(j3, bump);
            v0 = _mm256_loadu_pd(a + i + 0);
            v1 = _mm256_loadu_pd(a + i + 4);
            v2 = _mm256_loadu_pd(a + i + 8);
            v3 = _mm256_loadu_pd(a + i + 12);
            c0 = _mm256_cmp_pd(v0, m0, _CMP_LT_OQ);
            c1 = _mm256_cmp_pd(v1, m1, _CMP_LT_OQ);
            c2 = _mm256_cmp_pd(v2, m2, _CMP_LT_OQ);
            c3 = _mm256_cmp_pd(v3, m3, _CMP_LT_OQ);
            m0 = _mm256_blendv_pd(m0, v0, c0); k0 = _mm256_blendv_pd(k0, j0, c0);
            m1 = _mm256_blendv_pd(m1, v1, c1); k1 = _mm256_blendv_pd(k1, j1, c1);
            m2 = _mm256_blendv_pd(m2, v2, c2); k2 = _mm256_blendv_pd(k2, j2, c2);
            m3 = _mm256_blendv_pd(m3, v3, c3); k3 = _mm256_blendv_pd(k3, j3, c3);
        }
        c  = _mm256_cmp_pd(m1, m0, _CMP_LT_OQ);
        m0 = _mm256_blendv_pd(m0, m1, c); k0 = _mm256_blendv_pd(k0, k1, c);
        c  = _mm256_cmp_pd(m3, m2, _CMP_LT_OQ);
        m2 = _mm256_blendv_pd(m2, m3, c); k2 = _mm256_blendv_pd(k2, k3, c);
        c  = _mm256_cmp_pd(m2, m0, _CMP_LT_OQ);
        m0 = _mm256_blendv_pd(m0, m2, c); k0 = _mm256_blendv_pd(k0, k2, c);
        _mm256_storeu_pd(vv, m0);
        _mm256_storeu_pd(kk, k0);
        bv = vv[0]; best = (int)kk[0];
        for (t = 1; t < 4; t++)
            if (vv[t] < bv) { bv = vv[t]; best = (int)kk[t]; }
    }
#endif
    for (; i < nact; i++)
        if (a[i] < bv) { bv = a[i]; best = i; }
    return best;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    int    *off, *eto, *cur, *cid, *pos, *pred;
    double *ew, *cd, *predw;
    unsigned char *settled;
    long long nfinite;
    int nact, i, e;

    if (n <= 0) return;

    /* ---- step 1: lay out the grid so every corner is visible; a bare spool
       beside each corner; nothing on the traveler's corner, the impossible
       thread on every other; every corner unstoned. ---- */
    off     = (int *)malloc((size_t)(n + 1) * sizeof(int));
    cur     = (int *)malloc((size_t)(n + 1) * sizeof(int));
    eto     = (int *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    ew      = (double *)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    cd      = (double *)malloc((size_t)(n + 16) * sizeof(double));
    cid     = (int *)malloc((size_t)n * sizeof(int));
    pos     = (int *)malloc((size_t)n * sizeof(int));
    pred    = (int *)malloc((size_t)n * sizeof(int));
    predw   = (double *)malloc((size_t)n * sizeof(double));
    settled = (unsigned char *)malloc((size_t)n);
    if (!off || !cur || !eto || !ew || !cd || !cid || !pos || !pred ||
        !predw || !settled) {
        for (i = 0; i < n; i++) dist_out[i] = INFINITY;
        if (source >= 0 && source < n) dist_out[source] = 0.0;
        free(off); free(cur); free(eto); free(ew); free(cd);
        free(cid); free(pos); free(pred); free(predw); free(settled);
        return;
    }

    memset(off, 0, (size_t)(n + 1) * sizeof(int));
    for (i = 0; i < m; i++) off[src[i] + 1]++;
    for (i = 0; i < n; i++) off[i + 1] += off[i];
    memcpy(cur, off, (size_t)(n + 1) * sizeof(int));
    for (i = 0; i < m; i++) {
        int p = cur[src[i]]++;
        eto[p] = dst[i];
        ew[p]  = weight[i];
    }

    for (i = 0; i < n; i++) {
        dist_out[i] = INFINITY;    /* nothing final yet */
        cd[i]       = INFINITY;    /* the impossible thread */
        cid[i]      = i;
        pos[i]      = i;
        pred[i]     = -1;          /* bare spool */
        predw[i]    = 0.0;
        settled[i]  = 0;           /* unstoned */
    }
    for (i = n; i < n + 16; i++) cd[i] = INFINITY;   /* sweep padding */

    nact    = n;
    nfinite = 0;
    if (source >= 0 && source < n) {
        cd[pos[source]] = 0.0;     /* a thread length of nothing */
        nfinite = 1;
    }

    for (;;) {
        int k, u, x, guard, last, moved;
        double written, traced, tol;

        /* ---- step 2: are there unstoned corners wearing a carryable
           thread at all?  (kept as its own act; answered by the running
           tally of bare corners whose thread is no longer impossible) ---- */
        if (nfinite == 0) break;                    /* -> step 10 */

        /* ---- step 3: walk the grid, compare thread against thread, corner
           against corner, holding in mind only the shortest yet touched;
           the corner still holding at the last unstoned corner is the
           leader duck's landing place. ---- */
#ifdef _OPENMP
        if (nact >= (1 << 15)) {
            double bestv = INFINITY;
            int    besti = 0;
            #pragma omp parallel
            {
                int nt = omp_get_num_threads();
                int t  = omp_get_thread_num();
                int chunk = (nact + nt - 1) / nt;
                int lo, hi;
                chunk = (chunk + 15) & ~15;
                lo = t * chunk;
                hi = lo + chunk;
                if (hi > nact) hi = nact;
                if (lo < hi) {
                    int lb = drs_sweep(cd + lo, hi - lo);
                    double lv = cd[lo + lb];
                    #pragma omp critical
                    { if (lv < bestv) { bestv = lv; besti = lo + lb; } }
                }
            }
            k = besti;
        } else {
            k = drs_sweep(cd, nact);
        }
#else
        k = drs_sweep(cd, nact);
#endif
        u       = cid[k];
        written = cd[k];

        /* ---- step 4: do not trust the leader on sight. Trace its thread
           back shuttle by shuttle to the traveler's corner, adding the
           centimetres of each road. Mismatch -> correct the written length
           and return to step 2. Match -> the leader is true.
           (Comparison is made to floating-point tolerance: the trace re-adds
           the same roads in reverse order, so it may differ by a few ULP
           from the value that was written. This is the one, minimal repair.) */
        traced = 0.0;
        x = u;
        guard = 0;
        while (x != source) {
            if (pred[x] < 0) { traced = INFINITY; break; }
            traced += predw[x];
            x = pred[x];
            if (++guard > n) { traced = INFINITY; break; }
        }
        tol = 1e-9 * (1.0 + fabs(written));
        if (!(fabs(traced - written) <= tol)) {
            cd[k] = traced;                         /* correct it */
            continue;                               /* -> step 2 */
        }

        /* ---- step 5: place a black stone. This length is now final and is
           never touched, compared, or rewritten again: the corner leaves the
           swept ground entirely (its slot is filled by the last bare corner),
           and its length moves to where nothing can rewrite it. ---- */
        settled[u]  = 1;
        dist_out[u] = written;
        nfinite--;
        last  = --nact;
        moved = cid[last];
        cid[k]      = moved;
        cd[k]       = cd[last];
        pos[moved]  = k;
        cd[last]    = INFINITY;

        /* ---- step 6: from the newly stoned corner take each road leading
           out, one road at a time, in whatever order they lie. ---- */
        for (e = off[u]; e < off[u + 1]; e++) {
            int v = eto[e];
            int pv;
            double nd;

            /* ---- step 8 (first clause): if the far corner is already
               stoned, throw the fresh thread away unspooled. ---- */
            if (settled[v]) continue;

            /* ---- step 7: add the road's centimetres to the stoned corner's
               final length; that sum is the fresh duck's thread. ---- */
            nd = written + ew[e];

            /* ---- step 8: compare fresh thread against the thread already
               worn by the far corner. Shorter -> cut away the old thread,
               write the fresh length, note the stoned corner as the shuttle.
               Equal or longer -> throw it away unspooled. ---- */
            pv = pos[v];
            if (nd < cd[pv]) {
                if (cd[pv] == INFINITY) nfinite++;
                cd[pv]   = nd;
                pred[v]  = u;
                predw[v] = ew[e];
            }
        }

        /* ---- step 9: return to step 2 and do the whole round again. ---- */
    }

    /* ---- step 10: finished. Every stoned corner holds its true shortest
       length (already in dist_out) and its shuttle (pred/predw), so the road
       back can be read corner by corner. Every corner still bare keeps the
       impossible thread: its light is simply lost. ---- */
    free(off); free(cur); free(eto); free(ew); free(cd);
    free(cid); free(pos); free(pred); free(predw); free(settled);
}
