#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* The native's recipe, literally: places = nodes, roads = edges, embers = dist[],
   one "throw" = one land-wide synchronous relaxation of every road at once.      */
void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    if (m < 0) m = 0;

    /* step 1: lay out one heap per place, each empty; cold ash (no ember) everywhere.
       At the traveler's disc, lay the stick notched to nothing at all and kindle it. */
    double *ember     = (double *)malloc((size_t)n * sizeof(double));
    double *banked    = (double *)malloc((size_t)n * sizeof(double));
    int    *roadstart = (int    *)malloc(((size_t)n + 1) * sizeof(int));
    int    *cursor    = (int    *)malloc((size_t)n * sizeof(int));
    int    *nearend   = (m > 0) ? (int    *)malloc((size_t)m * sizeof(int))    : NULL;
    double *notch     = (m > 0) ? (double *)malloc((size_t)m * sizeof(double)) : NULL;

    if (!ember || !banked || !roadstart || !cursor ||
        (m > 0 && (!nearend || !notch))) {
        for (int v = 0; v < n; v++) dist_out[v] = INFINITY;
        if (source >= 0 && source < n) dist_out[source] = 0.0;
        free(ember); free(banked); free(roadstart);
        free(cursor); free(nearend); free(notch);
        return;
    }

#pragma omp parallel for schedule(static) if(n > 8192)
    for (int v = 0; v < n; v++) { ember[v] = INFINITY; banked[v] = INFINITY; }
    if (source >= 0 && source < n) ember[source] = 0.0;   /* the kindled ember */

    /* step 2: walk every road once and notch its own length into a stick kept at the
       roadside, so each road holds its measure permanently. The roadside sticks are
       filed by the road's far end, so a heap can later find all its own roads.       */
    for (int v = 0; v <= n; v++) roadstart[v] = 0;
    for (int e = 0; e < m; e++)  roadstart[dst[e] + 1]++;
    for (int v = 0; v < n; v++)  roadstart[v + 1] += roadstart[v];
    memcpy(cursor, roadstart, (size_t)n * sizeof(int));
    for (int e = 0; e < m; e++) {
        int p = cursor[dst[e]]++;
        nearend[p] = src[e];      /* which heap this road starts at   */
        notch[p]   = weight[e];   /* the road's permanently notched length */
    }
    free(cursor);
    cursor = NULL;

    int changed = 1;
    int throws  = 0;
    const int throw_cap = (n > 1) ? (n - 1) : 1;   /* step 10: never more than places-1 */

    while (changed) {
        changed = 0;

#pragma omp parallel for schedule(guided) reduction(|:changed) if(m > 8192)
        for (int v = 0; v < n; v++) {
            const int lo = roadstart[v], hi = roadstart[v + 1];
            double sliver = INFINITY;

            /* step 3: begin a throw. For every road in the land at once, look at the
               heap at the road's near end. An emberless heap's stick is notched to
               INFINITY, so that road does nothing this throw. Otherwise take a fresh
               stick, notch it to (ember + road length), carry it down the road and lay
               it on the heap at the far end -- this heap.                            */
            /* step 4: only when every road has been walked do the heaps throw together
               in a single land-wide throw. Enforced by reading ONLY pre-throw embers
               (`ember`) and writing only to the banked pile (`banked`): no new ember is
               visible anywhere until the whole throw is over. The sticks cross and
               settle into the pyramid as they land.                                  */
#pragma omp simd reduction(min:sliver)
            for (int p = lo; p < hi; p++) {
                double s = ember[nearend[p]] + notch[p];
                sliver = s < sliver ? s : sliver;
            }

            /* step 5: read the pyramid straight across a single row -- this row holds
               one settled sliver from every heap in the same breath, so every place is
               judged in the same moment and none before another.                     */
            const double old_ember = ember[v];

            /* step 6: in each heap, compare the slivers against one another and against
               that heap's current ember; keep the single shortest. Cold ash + shortest
               sliver becomes an ember; a shorter sliver replaces the old ember (which is
               burned); otherwise the ember stands unchanged.                          */
            const double keep = (sliver < old_ember) ? sliver : old_ember;

            /* step 7: burn every sliver that was not kept -- nothing is retained, each
               heap carries exactly one thing forward: its ember, banked to a coal.    */
            banked[v] = keep;

            /* step 8 (a): note whether any ember anywhere in the land changed. */
            if (keep < old_ember) changed = 1;
        }

        { double *t = ember; ember = banked; banked = t; }   /* the banked coals */

        /* step 8 (b): if even one ember changed, go back to step 3 and throw again with
           the new embers in place.                                                    */
        /* step 9: if a whole throw passed and not one ember changed, stop -- the embers
           have settled and no further throw can shorten them (the `while` condition).  */

        /* step 10: count the throws. You may never need more than places-minus-one.
           With every road's notch non-negative no ring of roads can give back more than
           it takes, so the "unsound heap" branch is provably unreachable here and after
           this many throws the embers are final.                                       */
        throws++;
        if (throws >= throw_cap) break;
    }

    /* step 11: every heap still holding cold ash and no ember was never reached by any
       spark -- a dead field the elephants never walked. Leave its distance forever
       unlit.                                                                          */
#pragma omp parallel for schedule(static) if(n > 8192)
    for (int v = 0; v < n; v++) {
        if (!(ember[v] < INFINITY)) ember[v] = INFINITY;
    }

    /* step 12: the hooded figure reads the whole land's embers aloud in one pass. */
    memcpy(dist_out, ember, (size_t)n * sizeof(double));

    free(ember); free(banked); free(roadstart); free(nearend); free(notch);
}
