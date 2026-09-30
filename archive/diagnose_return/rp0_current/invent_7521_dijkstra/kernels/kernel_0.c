#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif

/* ---------------------------------------------------------------
   THE NIGHTINGALE SWEEP
   The flock circles the contiguous array of owed sums (unlocked,
   readable stones only) and drops onto the smallest, singing back
   its slot.  4 lanes (AVX2) or 8 lanes (AVX-512) circle at once.
   --------------------------------------------------------------- */
static inline int nightingale_argmin(const double *restrict a, int n)
{
#if defined(__AVX512F__)
    if (n >= 16) {
        __m512d vmin = _mm512_loadu_pd(a);
        __m512i vidx = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        __m512i vcur = vidx;
        const __m512i v8 = _mm512_set1_epi64(8);
        int i = 8;
        for (; i + 8 <= n; i += 8) {
            vcur = _mm512_add_epi64(vcur, v8);
            __m512d v = _mm512_loadu_pd(a + i);
            __mmask8 lt = _mm512_cmp_pd_mask(v, vmin, _CMP_LT_OQ);
            vmin = _mm512_mask_blend_pd(lt, vmin, v);
            vidx = _mm512_mask_blend_epi64(lt, vidx, vcur);
        }
        double mv[8]; long long mi[8];
        _mm512_storeu_pd(mv, vmin);
        _mm512_storeu_si512((void *)mi, vidx);
        double best = mv[0]; int bi = (int)mi[0];
        for (int k = 1; k < 8; ++k) if (mv[k] < best) { best = mv[k]; bi = (int)mi[k]; }
        for (; i < n; ++i)          if (a[i] < best)  { best = a[i];  bi = i; }
        return bi;
    }
#endif
#if defined(__AVX__)
    if (n >= 8) {
        __m256d vmin = _mm256_loadu_pd(a);
        __m256i vidx = _mm256_setr_epi64x(0, 1, 2, 3);
        __m256i vcur = vidx;
        const __m256i v4 = _mm256_set1_epi64x(4);
        int i = 4;
        for (; i + 4 <= n; i += 4) {
            vcur = _mm256_add_epi64(vcur, v4);
            __m256d v  = _mm256_loadu_pd(a + i);
            __m256d lt = _mm256_cmp_pd(v, vmin, _CMP_LT_OQ);
            vmin = _mm256_blendv_pd(vmin, v, lt);
            vidx = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(vidx),
                                                       _mm256_castsi256_pd(vcur), lt));
        }
        double mv[4]; long long mi[4];
        _mm256_storeu_pd(mv, vmin);
        _mm256_storeu_si256((__m256i *)mi, vidx);
        double best = mv[0]; int bi = (int)mi[0];
        for (int k = 1; k < 4; ++k) if (mv[k] < best) { best = mv[k]; bi = (int)mi[k]; }
        for (; i < n; ++i)          if (a[i] < best)  { best = a[i];  bi = i; }
        return bi;
    }
#endif
    {
        double best = a[0]; int bi = 0;
        for (int i = 1; i < n; ++i) if (a[i] < best) { best = a[i]; bi = i; }
        return bi;
    }
}

/* ---------------------------------------------------------------
   THE CAIRN (fallback for the wide-garden / thin-roads regime):
   a plain lazy binary heap -- the known way, kept in reserve.
   --------------------------------------------------------------- */
typedef struct { double d; int u; } Cairn;

static inline void cairn_push(Cairn *restrict h, int *restrict hs, double d, int u)
{
    int i = (*hs)++;
    h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p].d <= h[i].d) break;
        Cairn t = h[p]; h[p] = h[i]; h[i] = t; i = p;
    }
}

static inline Cairn cairn_pop(Cairn *restrict h, int *restrict hs)
{
    Cairn top = h[0];
    (*hs)--; h[0] = h[*hs];
    int i = 0, s;
    for (;;) {
        int l = 2 * i + 1, r = l + 1;
        s = i;
        if (l < *hs && h[l].d < h[s].d) s = l;
        if (r < *hs && h[r].d < h[s].d) s = r;
        if (s == i) break;
        Cairn t = h[s]; h[s] = h[i]; h[i] = t; i = s;
    }
    return top;
}

#define LOCKED (-2)
#define UNSEEN (-1)

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) dist_out[i] = INFINITY;      /* unreadable debt */
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;                                  /* "nothing owed" */
    if (m <= 0) return;

    /* ---- the roads: directed CSR, each thief's toll, one direction only ---- */
    int    *off  = (int *)   calloc((size_t)n + 1, sizeof(int));
    int    *cur  = (int *)   malloc((size_t)n * sizeof(int));
    int    *edst = (int *)   malloc((size_t)m * sizeof(int));
    double *ew   = (double *)malloc((size_t)m * sizeof(double));
    /* ---- the knot of three threads, one per candidate stone ---- */
    double *fd   = (double *)malloc((size_t)n * sizeof(double)); /* owed sum   */
    int    *fu   = (int *)   malloc((size_t)n * sizeof(int));    /* place-name */
    int    *pos  = (int *)   malloc((size_t)n * sizeof(int));    /* back-thread*/

    if (!off || !cur || !edst || !ew || !fd || !fu || !pos) {
        free(off); free(cur); free(edst); free(ew); free(fd); free(fu); free(pos);
        return;
    }

    for (int i = 0; i < m; ++i) ++off[src[i] + 1];
    for (int i = 0; i < n; ++i) off[i + 1] += off[i];
    memcpy(cur, off, (size_t)n * sizeof(int));
    for (int i = 0; i < m; ++i) {
        int p = cur[src[i]]++;
        edst[p] = dst[i];
        ew[p]   = weight[i];
    }
    free(cur);

    const int    *restrict EO = off;
    const int    *restrict ED = edst;
    const double *restrict EW = ew;
    double *restrict D  = dist_out;
    double *restrict FD = fd;
    int    *restrict FU = fu;
    int    *restrict PS = pos;

    memset(PS, 0xFF, (size_t)n * sizeof(int));               /* all UNSEEN */

    int fs = 0;
    FD[0] = 0.0; FU[0] = source; PS[source] = 0; fs = 1;     /* climb the tower */

    /* "I wait": how much circling the flock may do before the native decides
       the garden is too wide for birds and starts stacking a cairn instead. */
    const double budget = 2.0 * ((double)m + (double)n) * log2((double)n + 2.0) + 4096.0;
    double circled = 0.0;
    int tired = 0;

    while (fs > 0) {
        if (circled > budget) { tired = 1; break; }
        circled += (double)fs;

        int k = nightingale_argmin(FD, fs);                  /* the bird lands  */
        int u = FU[k];
        double du = FD[k];

        PS[u] = LOCKED;                                      /* letter is set   */
        --fs;
        if (k != fs) { FD[k] = FD[fs]; FU[k] = FU[fs]; PS[FU[fs]] = k; }

        /* cast threads from the newly locked house */
        int e = EO[u], ee = EO[u + 1];
        for (; e < ee; ++e) {
            int v = ED[e];
            double nd = du + EW[e];
            /* A locked stone can never pass this test: IEEE addition is
               monotone and every toll is >= 0, so nd >= du >= D[v]. */
            if (nd < D[v]) {
                D[v] = nd;
                int p = PS[v];
                if (p >= 0) FD[p] = nd;                      /* knot re-tied    */
                else { PS[v] = fs; FD[fs] = nd; FU[fs] = v; ++fs; }  /* enrolled */
            }
        }
    }

    if (tired) {
        /* wide garden, thin roads: stack the unlocked stones into a cairn and
           finish the walk the known way, on the very same letters. */
        Cairn *heap = (Cairn *)malloc(((size_t)n + (size_t)m + 2) * sizeof(Cairn));
        if (heap) {
            int hs = 0;
            for (int i = 0; i < fs; ++i) cairn_push(heap, &hs, FD[i], FU[i]);
            while (hs > 0) {
                Cairn t = cairn_pop(heap, &hs);
                int u = t.u;
                if (PS[u] == LOCKED) continue;
                PS[u] = LOCKED;
                double du = D[u];
                int e = EO[u], ee = EO[u + 1];
                for (; e < ee; ++e) {
                    int v = ED[e];
                    double nd = du + EW[e];
                    if (nd < D[v]) { D[v] = nd; cairn_push(heap, &hs, nd, v); }
                }
            }
            free(heap);
        }
    }

    free(off); free(edst); free(ew); free(fd); free(fu); free(pos);
}
