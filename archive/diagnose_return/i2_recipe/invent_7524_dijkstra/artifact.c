#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define BLK      64      /* chalk squares summarized by one margin note */
#define BLKSH    6
#define DARK     (-1)    /* blank square: no chalked number, never lit      */
#define CROUCHED (-2)    /* stroke in the corner: already crouched beside   */

/* machine-level helpers: the recipe's "look over ... and find the smallest",
   done on contiguous doubles.  Both return exactly what a scalar scan would. */
static inline double vmin_range(const double *a, int lo, int hi)
{
    int j = lo;
    double mn = INFINITY;
#if defined(__AVX2__)
    if (hi - lo >= 4) {
        __m256d v = _mm256_set1_pd(INFINITY);
        for (; j + 4 <= hi; j += 4)
            v = _mm256_min_pd(v, _mm256_loadu_pd(a + j));
        double t[4];
        _mm256_storeu_pd(t, v);
        mn = t[0] < t[1] ? t[0] : t[1];
        if (t[2] < mn) mn = t[2];
        if (t[3] < mn) mn = t[3];
    }
#endif
    for (; j < hi; ++j) if (a[j] < mn) mn = a[j];
    return mn;
}

static inline int vfind_le(const double *a, int lo, int hi, double t)
{
    int j = lo;
#if defined(__AVX2__)
    __m256d tv = _mm256_set1_pd(t);
    for (; j + 4 <= hi; j += 4) {
        int msk = _mm256_movemask_pd(
                      _mm256_cmp_pd(_mm256_loadu_pd(a + j), tv, _CMP_LE_OQ));
        if (msk) return j + (int)__builtin_ctz((unsigned)msk);
    }
#endif
    for (; j < hi; ++j) if (a[j] <= t) return j;
    return -1;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    if (m < 0) m = 0;

    /* step 1: hang one jewel per place; knot between each pair that has a road
       a cord cut exactly as long as the road (cord_len = weight).  Where there
       is no road, no cord exists.  The net is stored as CSR, each jewel's cords
       contiguous, so step 5 can run a finger along them at memory speed.
       (Corrected reading: a cord is followed only in its road's direction.) */
    const size_t me = (size_t)(m > 0 ? m : 1);
    int    *head     = (int *)   malloc((size_t)(n + 1) * sizeof(int));
    int    *cursor   = (int *)   malloc((size_t)(n + 1) * sizeof(int));
    int    *cord_to  = (int *)   malloc(me * sizeof(int));
    double *cord_len = (double *)malloc(me * sizeof(double));
    int    *pos      = (int *)   malloc((size_t)n * sizeof(int));  /* square state */
    int    *fid      = (int *)   malloc((size_t)n * sizeof(int));  /* lit+uncrouched */
    double *fkey     = (double *)malloc((size_t)n * sizeof(double));
    const int nbcap  = (n + BLK - 1) / BLK + 1;
    double *bm       = (double *)malloc((size_t)nbcap * sizeof(double));

    if (!head || !cursor || !cord_to || !cord_len || !pos || !fid || !fkey || !bm) {
        for (int i = 0; i < n; ++i) dist_out[i] = INFINITY;
        if (source >= 0 && source < n) dist_out[source] = 0.0;
        free(head); free(cursor); free(cord_to); free(cord_len);
        free(pos); free(fid); free(fkey); free(bm);
        return;
    }

    memset(head, 0, (size_t)(n + 1) * sizeof(int));
    for (int i = 0; i < m; ++i) head[src[i] + 1]++;
    for (int u = 0; u < n; ++u) head[u + 1] += head[u];
    memcpy(cursor, head, (size_t)(n + 1) * sizeof(int));
    for (int i = 0; i < m; ++i) {
        int p = cursor[src[i]]++;
        cord_to[p]  = dst[i];
        cord_len[p] = weight[i];
    }

    /* step 2: scratch the grid, one square per jewel, every square left blank;
       set the throwing-stone in the square of the jewel where the traveler
       stands.  Blank == INFINITY + state DARK.  Nothing is lit yet, so the
       lit-and-uncrouched set is empty and every margin note is blank. */
#pragma omp parallel for schedule(static) if (n > 65536)
    for (int i = 0; i < n; ++i) { dist_out[i] = INFINITY; pos[i] = DARK; }
    for (int b = 0; b < nbcap; ++b) bm[b] = INFINITY;
    int fn = 0;
    const int stone = (source >= 0 && source < n) ? source : 0;

    /* step 3: on that one square only, chalk the number nought -- the stone's
       own square catches fire at no cost of waiting -- and so it becomes the
       first lit, uncrouched square. */
    dist_out[stone] = 0.0;
    fid[0] = stone; fkey[0] = 0.0; pos[stone] = 0; fn = 1;
    bm[0] = 0.0;

    while (fn > 0) {
        /* step 4: look over the lit squares, keep only those not yet crouched
           beside (exactly the frontier), and choose the smallest chalked
           number.  Ties: "take either -- it makes no difference".  If there
           are none at all the loop ends and we go to step 9. */
        const int nb = (fn + BLK - 1) >> BLKSH;
        const double best = vmin_range(bm, 0, nb);
        const int bb = vfind_le(bm, 0, nb, best);
        int lo, hi;
        if (bb >= 0) { lo = bb << BLKSH; hi = lo + BLK; if (hi > fn) hi = fn; }
        else         { lo = 0; hi = fn; }            /* fully literal scan */
        const double bv = vmin_range(fkey, lo, hi);
        const int    bi = vfind_le(fkey, lo, hi, bv);
        const int    u  = fid[bi];
        const double du = fkey[bi];                  /* its chalked number */

        /* step 5: crouch beside that jewel and follow with your fingers every
           cord knotted to it, one cord at a time, reading the cord's length
           and the chosen jewel's own chalked number (du). */
        const int e0 = head[u], e1 = head[u + 1];
        for (int e = e0; e < e1; ++e) {
            const int    v = cord_to[e];
            const double L = cord_len[e];

            /* step 6: add the cord's length to the chalked number -- the
               instant the glint arrives at the far jewel by this route. */
            const double sum = du + L;

            /* step 7: compare the sum against the far jewel's square.
               dark (INFINITY)  -> first flare, chalk it in;
               carries <= sum   -> a late flare down a slacker cord, discard;
               carries  > sum   -> rub it out and chalk the sum in its place. */
            if (sum < dist_out[v]) {
                dist_out[v] = sum;
                const int p = pos[v];
                if (p >= 0) {                        /* rub out, chalk in */
                    fkey[p] = sum;
                    const int b = p >> BLKSH;
                    if (sum < bm[b]) bm[b] = sum;
                } else {                             /* first flare: light it */
                    const int i = fn++;
                    fid[i] = v; fkey[i] = sum; pos[v] = i;
                    const int b = i >> BLKSH;
                    if ((i & (BLK - 1)) == 0) bm[b] = sum;
                    else if (sum < bm[b])     bm[b] = sum;
                }
            }
        }

        /* step 8: every cord read -- mark the jewel crouched-beside with a
           stroke in the corner of its square, so you never crouch there
           twice (it leaves the lit-and-uncrouched set), then return to
           step 4.  The two disturbed chalk rows get their margin notes
           re-read from the squares themselves. */
        pos[u] = CROUCHED;
        const int last = --fn;
        if (bi != last) {
            fid[bi] = fid[last]; fkey[bi] = fkey[last]; pos[fid[bi]] = bi;
        }
        {
            const int b1 = bi >> BLKSH, b2 = last >> BLKSH;
            int l1 = b1 << BLKSH, h1 = l1 + BLK; if (h1 > fn) h1 = fn;
            bm[b1] = (l1 >= fn) ? INFINITY : vmin_range(fkey, l1, h1);
            if (b2 != b1) {
                int l2 = b2 << BLKSH, h2 = l2 + BLK; if (h2 > fn) h2 = fn;
                bm[b2] = (l2 >= fn) ? INFINITY : vmin_range(fkey, l2, h2);
            }
        }
    }

    /* step 9: no lit square is left uncrouched, so no glint is still
       travelling in the net.  Cross out with a single scratch every square
       still dark -- those places no glint ever reached.  The scratch is the
       INFINITY the contract asks for. */
    for (int i = 0; i < n; ++i) if (pos[i] == DARK) dist_out[i] = INFINITY;

    /* step 10: read the grid -- each chalked number is the length of the
       shortest way from the traveler's jewel to that place, each scratched
       square a place with no way at all.  That grid is dist_out itself. */
    free(head); free(cursor); free(cord_to); free(cord_len);
    free(pos); free(fid); free(fkey); free(bm);
}
