/* Native's recipe, literal: fixed upper row of beast-pawns, shiftable lower row,
   a horse's-head that slips one house, red knots weighed against each other.
   Cost currency = "grains" (half-knots): differing pair 4, empty house 5.
   For any partial arrangement of a[0..i) with b[0..j):  score = ((i+j) - grains)/2,
   so fewest grains == best score, and at (n,n): score = n - grains/2.          */

#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define GR_MIS 4            /* grains: a knot for two different beasts        */
#define GR_GAP 5            /* grains: a knot for a pawn facing an empty house */
#define GR_BIG (1 << 27)    /* a fist too heavy to lift: off the road          */

int kernel(int n, const char *a, const char *b)
{
    const int PAD = 32;
    int i, k, z, x0 = 0, pre, bestgr, W, cost, ilo, ihi, lo, hi, t, base;
    size_t stride;
    int *pool, *A0, *A1, *A2, *cur, *p1, *p2, *tmp, *sufL, *sufU;
    char *cbuf, *up, *lowrev;

    if (n <= 0) return 0;

    /* step 1: scratch one road of houses and cut two furrows beside it.
       The road is the anti-diagonal index; the houses are indexed by i.
       Three rotating diagonal buffers + two suffix-knot tables, one pool.    */
    stride = (size_t)n + 1 + 2 * (size_t)PAD;
    pool = (int *)malloc(3 * stride * sizeof(int) + 2 * ((size_t)n + 2) * sizeof(int));
    cbuf = (char *)malloc(2 * ((size_t)n + 64));
    if (!pool || !cbuf) {                 /* no mud to scratch in: trial 0 only */
        free(pool); free(cbuf);
        for (i = 0; i < n; i++) x0 += (a[i] != b[i]);
        return n - 2 * x0;
    }
    A0 = pool + PAD;
    A1 = pool + stride + PAD;
    A2 = pool + 2 * stride + PAD;
    sufL = pool + 3 * stride;
    sufU = sufL + n + 2;
    up = cbuf;
    lowrev = cbuf + n + 64;

    /* step 2: set the upper row from the first string, one pawn per house,
       and press it into the mud. This row never moves again.                 */
    memcpy(up, a, (size_t)n);
    memset(up + n, 0, 64);

    /* step 3: the lower row from the second string, nose to nose, left loose.
       Kept reversed, because a shoved lower row is read at a sliding offset:
       b[j-1] with j = k-i  is  lowrev[(n-k)+i], contiguous in i.             */
    for (i = 0; i < n; i++) lowrev[i] = b[n - 1 - i];
    memset(lowrev + n, 0, 64);

    /* step 4: set the horse aside, standing nowhere -- the rows unslipped.    */

    /* step 5: walk that trial, one note per house; petal if the beasts match,
       a red knot if they differ.                                             */
    for (i = 0; i < n; i++) x0 += (up[i] != b[i]);

    /* step 6: gather the knots, weigh the fist, lay the thread aside.         */
    bestgr = GR_MIS * x0;

    /* steps 7-9: move the horse -- set down in every house in turn, the lower
       pawns from there on shuffled one house forward, the far pawn shoved off
       the road's end (that is the second empty house). Also, per repair R1,
       the mirror family with the horse in the upper furrow. Each such trial's
       thread is weighed without re-walking the whole road: the knots before
       the horse are a prefix count, those after it a suffix count.            */
    sufL[n] = 0; sufU[n] = 0;
    for (i = n - 1; i >= 1; i--) {
        sufL[i] = sufL[i + 1] + (up[i] != b[i - 1]);   /* lower row slipped */
        sufU[i] = sufU[i + 1] + (b[i] != up[i - 1]);   /* upper row slipped */
    }
    pre = 0;
    for (k = 0; k < n; k++) {
        int cL = GR_MIS * (pre + sufL[k + 1]) + 2 * GR_GAP;
        int cU = GR_MIS * (pre + sufU[k + 1]) + 2 * GR_GAP;
        if (cL < bestgr) bestgr = cL;
        if (cU < bestgr) bestgr = cU;
        pre += (up[k] != b[k]);
    }

    /* step 10: we now hold one thread per house of each furrow, plus the one
       from when the horse stood nowhere at all.                              */

    /* step 11: weigh all kept threads against one another and keep the
       lightest. Its weight also says how far the horse can EVER pay to
       wander: an arrangement with g slips weighs >= 10g grains, so every
       lightest arrangement has g <= bestgr/10. Call that W.                  */
    W = bestgr / (2 * GR_GAP);
    if (W > n) W = n;
    if (W == 0) {                      /* no slip can pay: keep this fist      */
        free(pool); free(cbuf);
        return n - bestgr / 2;         /* step 12 */
    }

    /* step 9b (REPAIR R1+R3 of steps 9 and 11): the horse may be set down
       again after each slip, and in either furrow, so the family of
       arrangements is every arrangement with at most W slips. They are
       weighed house by house instead of thread by thread (knots are additive
       along the road), which is the only way this enlarged family is
       walkable. One breath = one anti-diagonal k = i+j: those houses no
       longer lean on each other, so eight are marked per note.               */
    for (i = -PAD; i <= n + PAD; i++) { A0[i] = GR_BIG; A1[i] = GR_BIG; A2[i] = GR_BIG; }
    p2 = A2; p1 = A1; cur = A0;
    {
#if defined(__AVX2__)
        const __m256i vmis = _mm256_set1_epi32(GR_MIS);
        const __m256i vgap = _mm256_set1_epi32(GR_GAP);
#endif
        for (k = 0; k <= 2 * n; k++) {
            t  = k - W;
            lo = (t <= 0) ? 0 : ((t + 1) >> 1);      /* band: |j-i| <= W */
            hi = (k + W) >> 1;
            if (lo < k - n) lo = k - n;
            if (hi > n) hi = n;
            if (hi > k) hi = k;
            if (k <= W) {                            /* the road's rim */
                cur[0] = GR_GAP * k;                 /* whole upper row empty  */
                cur[k] = GR_GAP * k;                 /* whole lower row empty  */
                ilo = 1; ihi = k - 1;
            } else {
                ilo = lo; ihi = hi;
            }
            base = n - k;
            i = ilo;
#if defined(__AVX2__)
            for (; i + 7 <= ihi; i += 8) {
                __m128i ca = _mm_loadl_epi64((const __m128i *)(up + i - 1));
                __m128i cb = _mm_loadl_epi64((const __m128i *)(lowrev + base + i));
                __m256i sm = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(ca, cb));
                __m256i dg = _mm256_add_epi32(
                                 _mm256_loadu_si256((const __m256i *)(p2 + i - 1)),
                                 _mm256_andnot_si256(sm, vmis));
                __m256i gp = _mm256_add_epi32(_mm256_min_epi32(
                                 _mm256_loadu_si256((const __m256i *)(p1 + i - 1)),
                                 _mm256_loadu_si256((const __m256i *)(p1 + i))), vgap);
                _mm256_storeu_si256((__m256i *)(cur + i), _mm256_min_epi32(dg, gp));
            }
#endif
            for (; i <= ihi; i++) {
                int d = p2[i - 1] + ((up[i - 1] == lowrev[base + i]) ? 0 : GR_MIS);
                int g = (p1[i - 1] < p1[i] ? p1[i - 1] : p1[i]) + GR_GAP;
                cur[i] = (d < g) ? d : g;
            }
            cur[lo - 1] = GR_BIG;                    /* off the road, below */
            for (z = 1; z <= 8; z++) cur[hi + z] = GR_BIG;   /* and above   */
            tmp = p2; p2 = p1; p1 = cur; cur = tmp;
        }
    }

    /* step 12: fling every other thread to the flying fish, carry back the
       knots of the one kept fist, in the contract's currency.                */
    cost = p1[n];
    free(pool); free(cbuf);
    return n - cost / 2;
}
