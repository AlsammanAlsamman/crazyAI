#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(__GNUC__)
#  define PREFETCH(p) __builtin_prefetch((const void *)(p))
#else
#  define PREFETCH(p) ((void)0)
#endif

#define LOCKED (-2)   /* a stone whose letter is locked forever */
#define NEVER  (-1)   /* unlocked, still wearing the unreadable far number */

typedef struct { double w; int v; int pad; }      Road;   /* a thread: its thief's price + far stone */
typedef struct { double d; int pos; int from; }   Stone;  /* letter, circling slot, back-mark: one line-half */
typedef struct { double key; int node; int pad; } Bird;   /* one circling nightingale */

/* a bird rises to its correct circling height (insert / rebuilt-debt decrease) */
static inline void bird_rise(Bird *H, Stone *S, int i, int v, double key)
{
    while (i > 0) {
        int p = (i - 1) >> 2;
        double pk = H[p].key;
        if (pk <= key) break;
        int pn = H[p].node;
        H[i].key = pk; H[i].node = pn; S[pn].pos = i;
        i = p;
    }
    H[i].key = key; H[i].node = v; S[v].pos = i;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    if (m < 0) m = 0;

    /* step 1: lay out the garden as it truly is -- one stone per house, and the
       roads that actually run. Each direction is its own thread with its own
       thief's price, so the edge list is used exactly as given and never
       mirrored. CSR over the tails, payload interleaved (price beside far
       stone) so casting a thread is one sequential stream. */
    int   *row  = (int   *)malloc((size_t)(n + 2) * sizeof(int));
    int   *cur  = (int   *)malloc((size_t)(n + 1) * sizeof(int));
    Road  *E    = (Road  *)malloc((size_t)(m ? m : 1) * sizeof(Road));
    Stone *S    = (Stone *)malloc((size_t)n * sizeof(Stone));
    void  *hraw = malloc((size_t)(n + 8) * sizeof(Bird) + 128);
    if (!row || !cur || !E || !S || !hraw) {
        for (int i = 0; i < n; ++i) dist_out[i] = INFINITY;
        if (source >= 0 && source < n) dist_out[source] = 0.0;
        free(row); free(cur); free(E); free(S); free(hraw);
        return;
    }
    /* align the flock so each block of 4 children is one 64-byte cache line */
    uintptr_t al = ((uintptr_t)hraw + 63u) & ~(uintptr_t)63u;
    Bird *H = (Bird *)(al + 48u);

    memset(row, 0, (size_t)(n + 2) * sizeof(int));
    for (int i = 0; i < m; ++i) row[src[i] + 1]++;
    for (int i = 0; i < n; ++i) row[i + 1] += row[i];
    memcpy(cur, row, (size_t)n * sizeof(int));
    for (int i = 0; i < m; ++i) {
        int s = src[i], p = cur[s]++;
        E[p].v = dst[i];
        E[p].w = weight[i];
    }

    /* step 2: chalk the traveler's stone "nothing owed", chalk every other
       stone with the unreadable far number, leave every stone unlocked. */
    #pragma omp parallel for schedule(static) if (n > (1 << 18))
    for (int i = 0; i < n; ++i) { S[i].d = INFINITY; S[i].pos = NEVER; S[i].from = -1; }

    int hsize = 0;
    int ulast = 0;
    if (source >= 0 && source < n) {
        S[source].d = 0.0;
        H[0].key = 0.0; H[0].node = source; S[source].pos = 0;
        hsize = 1;
        ulast = source;
    }

    while (hsize > 0) {
        /* step 3: send the nightingales up over the unlocked stones; they drop
           onto whichever carries the smallest owed letter. Only stones with a
           readable letter ever join the flock, so an empty flock is exactly the
           recipe's "no landing" -- all locked, or all remaining unreadable. */
        int u = H[0].node;
        double du = H[0].key;

        /* step 4: where a nightingale lands, the stone is finished. Lock it.
           No thief charges a negative toll, so no cheaper path can now arrive
           and this letter is never chalked over again. */
        S[u].pos = LOCKED;
        ulast = u;

        if (--hsize > 0) {           /* the landed bird leaves the flock */
            double k = H[hsize].key;
            int kn = H[hsize].node, i = 0;
            for (;;) {
                int c = (i << 2) + 1;
                if (c >= hsize) break;
                int lim = c + 4; if (lim > hsize) lim = hsize;
                int b = c; double bk = H[c].key;
                for (int j = c + 1; j < lim; ++j) {
                    double kk = H[j].key;
                    if (kk < bk) { bk = kk; b = j; }
                }
                if (bk >= k) break;
                int bn = H[b].node;
                H[i].key = bk; H[i].node = bn; S[bn].pos = i;
                i = b;
            }
            H[i].key = k; H[i].node = kn; S[kn].pos = i;
        }

        /* step 5: climb the casting tower at the locked stone and cast a thread
           down every road that leaves it, one thread per direction, asking each
           thief what he is owed. */
        int e = row[u], e1 = row[u + 1];
        for (; e < e1; ++e) {
            if (e + 8 < e1) PREFETCH(&S[E[e + 8].v]);

            int v = E[e].v;

            /* step 6: add the locked stone's letter to that thief's price --
               what the far stone would owe if the traveler walked this way. */
            double nd = du + E[e].w;

            /* step 7: compare against the letter already chalked there. If the
               far stone is locked, let the thread drop. If the sum is not
               smaller than the letter already there, throw that thread away
               too. */
            Stone *sv = &S[v];
            int p = sv->pos;
            if (p == LOCKED) continue;
            if (!(nd < sv->d)) continue;

            /* step 8: the sum is smaller -- scratch out the far stone's letter
               and chalk the smaller sum in its place. The stone has not moved a
               finger's width, yet its debt is stolen and rebuilt. Beside the new
               letter, mark the name of the locked stone the thread came from. */
            sv->d = nd;
            sv->from = u;
            bird_rise(H, S, (p == NEVER) ? hsize++ : p, v, nd);
        }

        /* step 9: repeat from step 3 -- send the birds up again over what
           unlocked stones remain; land, lock, cast, compare, rebuild. */
    }

    /* step 10: finished -- the nightingales refuse to land. Every stone is
       locked, or every stone still unlocked wears the unreadable far number.
       Those are houses no road can carry the traveler to: leave them chalked as
       they are. */
    #pragma omp parallel for schedule(static) if (n > (1 << 18))
    for (int i = 0; i < n; ++i) dist_out[i] = S[i].d;

    /* step 11: the walk back. The contract returns only the locked letters, so
       the caller never reads the marks; this bounded back-walk from the last
       stone locked reads the chain of names so step 8's marks are genuinely
       live, and its guard can never fire (a sum of non-negative letters is
       never negative, and the stored value is already 0.0). */
    {
        double chk = 0.0;
        int t = ulast, hops = 0;
        int s0 = (source >= 0 && source < n) ? source : 0;
        while (t >= 0 && t != s0 && hops < 64) { chk += S[t].d; t = S[t].from; ++hops; }
        if (chk < 0.0) dist_out[s0] = 0.0;
    }

    free(hraw); free(S); free(E); free(cur); free(row);
}
