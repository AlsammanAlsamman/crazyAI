## DICTIONARY

| Recipe object | Computational object |
|---|---|
| the garden | the directed weighted graph |
| a stone / a house | a node index `0..n-1` |
| a road, a thread | one directed edge `src[i] -> dst[i]` |
| the thief on a road, his price | `weight[i]`, the edge cost, `>= 0` |
| "two directions = two threads, two prices, even where the road looks single" | the edge list is used **as given**, directed; never symmetrized. A CSR built on `src` only |
| the stone's chalked letter | `S[v].d`, the current distance label |
| "nothing owed" | `0.0` |
| the unreadable far number | `INFINITY` |
| unlocked stone | `S[v].pos >= 0` (circling) or `S[v].pos == NEVER` (`-1`, never yet reached) |
| locked stone / settling | `S[v].pos == LOCKED` (`-2`); the label is final |
| the nightingales circling the unlocked stones | the priority structure over unlocked stones with readable letters (a 4-ary implicit heap, `Bird` entries) |
| a nightingale landing | extracting the minimum-label unlocked stone |
| the casting tower | the CSR row `row[u] .. row[u+1]` of the just-locked stone |
| casting a thread down a road | reading one out-edge `E[e]` |
| the sum (letter + price) | `nd = du + E[e].w` |
| "let the thread drop and throw it away" | `continue` — no write, nothing enqueued |
| "the stone's debt is stolen and rebuilt" | `S[v].d = nd` plus insert-or-decrease-key |
| the name marked beside the letter | `S[v].from`, the predecessor |
| birds refuse to land | heap empty |
| stones left unreadable | nodes left at `INFINITY` in `dist_out` |

**Ambiguity, most literal reading, stated openly.** Step 3 says the birds *circle the unlocked stones* and land on the smallest. The flat-footed reading is a linear scan of all unlocked stones each round (`O(n^2 + m)`), and that is exactly what my v1 was. But step 3 legislates **which stone is chosen** (the globally smallest owed letter among unlocked stones), not how the birds find it — the flight is not part of the answer. So from v2 on the *selection rule is preserved bit-for-bit* while the search is done with a heap. I am not replacing a step; I am choosing "how it is carried out at the machine level," which is what I was allowed to choose. The scan version is my own baseline, and it is slower, not faster.

**Is the recipe ever wrong?** No. I checked the one place it could be: step 4's lock is justified by "no thief charges a negative toll," which is exactly the non-negativity precondition; zero-weight roads are fine because step 7's `not smaller` test rejects equal sums. Step 10's two exit conditions collapse into one (`heap empty`) because a stone is only ever enqueued when its letter becomes readable. **No step required correction.** Step 11 is not asked for by the contract (only distances are returned), but the recipe demands the back-marks exist, so `S[v].from` is written in step 8 and step 11 is compiled in as a real bounded back-walk so the marks are not dead stores.

## ARTIFACT

Four improvement rounds, all inside the recipe: (1) literal linear scan over unlocked stones; (2) the circling set as a 4-ary implicit heap with decrease-key, cache-line-aligned so each 4-child block is exactly one 64 B line; (3) **one cache line per stone** — letter, circling slot and back-mark fused into a 16 B `Stone`, so step 7's two tests plus step 8's two writes touch a single line instead of three arrays; (4) 16 B interleaved CSR (`price` beside `far stone`, one stream) plus an 8-edge-ahead software prefetch of the far stone's line, since step 7 is a pure random-access probe and that is where all the stalls live.

```c
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
```

## PREDICTION

PREDICTION: speedup_vs_heap = 1.8

Reasoning before any measurement: the algorithm is identical to a heap Dijkstra, so every gain must come from memory traffic. Against a textbook binary-heap CSR baseline I expect ~1.15x from the 4-ary aligned heap (shallower tree, one cache line per child block), ~1.3x from fusing label/slot/back-mark into one 16 B line (step 7 + step 8 become one random line touch instead of three), and ~1.15x from the interleaved CSR plus prefetch — compounding to roughly 1.7-1.9x on sparse graphs with `m/n` around 4-16. If the baseline heap is a lazy-deletion one that pushes `O(m)` entries, I would expect 2.2-2.8x instead. If the graph is dense or tiny (`n < ~10^4`) I expect ~1.0-1.2x, because CSR construction and the `O(m)` scan dominate and there is nothing left to win. My own literal v1 (linear scan over unlocked stones) should be *slower* than the heap baseline by roughly `n/(4 log n)`, i.e. badly slower at `n = 10^6`.

## MEASUREMENT

Reported plainly: **no measurement was taken.** `dijkstra_bench` and `dijkstra_contract` were not available in this session — the tool surface I was given contains only Gmail, Drive, Slack and Docs connectors (two of which are unauthorized), and no benchmark harness. I did not measure, so I will not report a number as if I had, and I did not tune anything against feedback: the four improvement rounds above are *reasoned*, not *validated*. The 1.8 is a prediction standing naked.

What the pipeline should check, and what would falsify me:
- Exactness: `dist_out` bit-equal to the reference on reachable nodes (this kernel does the same additions in the same settling order as any Dijkstra, so it should be exact, not merely within tolerance), and `INFINITY` on unreachable ones.
- Degenerate inputs I reasoned through but could not run: `m == 0`; `source` isolated; zero-weight edges; duplicate parallel edges; self-loops (rejected by step 7's `LOCKED` test); a single node.
- If the measured speedup lands below ~1.2x, the most likely cause is that CSR construction dominates, and the honest next move would be a parallel counting sort — not more heap micro-tuning.

## VERDICT

The recipe is Dijkstra, stated without a single error, and unusually precise about the two things people get wrong: it insists the two directions of a road are separate threads with separate prices (directed edges, no symmetrization), and it gives the correct *reason* for locking a settled stone ("no thief anywhere charges a negative toll") rather than just the rule. Nothing needed correcting. The one real interpretive choice was step 3: the nightingales' flight is the priority queue, and the literal "circle every unlocked stone" reading is an `O(n^2)` scan. I built that first, kept its *selection rule* exactly — always the globally smallest owed letter among unlocked stones — and moved the search into a heap, which is a change of flight path, not of algorithm. Step 11 is real code, not a comment, because the recipe demands the back-marks and I would rather pay a bounded walk than let the compiler quietly delete a step. The prediction of 1.8x is unverified and may well be wrong; the artifact is complete and ready to be judged by the harness rather than by me.