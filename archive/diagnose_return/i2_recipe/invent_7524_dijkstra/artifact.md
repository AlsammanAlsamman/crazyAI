## DICTIONARY

| Recipe object | Computational object |
|---|---|
| a place the traveler might reach | node `0..n-1` |
| polished jewel hung for that place | its entry in the CSR node table |
| the traveler's jewel / where he stands | `source` |
| a road from one place to another | one directed edge `src[i] -> dst[i]` |
| blood-dark cord, cut exactly as long as the road | `weight[i]` (a short taut cord = small weight, long slack cord = large weight) |
| "where there is no road, knot no cord" | absent edge; no CSR entry |
| the net of jewels and cords | CSR: `head[n+1]`, `cord_to[m]`, `cord_len[m]` |
| chalk grid, one square per jewel | `dist_out[0..n-1]` |
| blank / **dark** square | `dist_out[v] == INFINITY` **and** `pos[v] == DARK(-1)` (never lit) |
| chalked number / **lit** square | finite `dist_out[v]` |
| flat throwing-stone | the integer `source` (marks which square gets the nought) |
| chalked nought, "catches fire at no cost of waiting" | `dist_out[source] = 0.0` |
| the set of lit squares not yet crouched beside | compact frontier arrays `fid[]` (jewel) ∥ `fkey[]` (its chalked number), size `fn` |
| "crouched beside" (stroke in the square's corner) | `pos[u] = CROUCHED(-2)`, i.e. **settled**; removed from the frontier |
| where a square lives in that set | `pos[v] >= 0` = its frontier slot (so step 7 can rub a number out in O(1)) |
| the glint crawling down a cord | one edge relaxation |
| "the instant the glint would arrive" (sum in your hand) | `sum = fkey[bi] + cord_len[e]` |
| first flare into a dark square | first-discovery push into the frontier |
| "late flare come along a slacker cord, wasted color" | `sum >= dist_out[v]` → discarded |
| "you had guessed a slow road before you found the quick one" | decrease-key: `dist_out[v] = sum`, `fkey[pos[v]] = sum` |
| margin summary of each chalk row (my only added layout) | `bm[b]` = exact min of the 64 frontier squares in block `b` — a *summary of the same numbers*, so step 4 selects exactly what the naive scan selects |
| square crossed out with a single scratch | `INFINITY` in `dist_out` |
| reading the grid like a thrown stone | `dist_out` as returned |

**Ambiguities, read as literally as possible**

1. **Step 4 is not a scan of the whole pavement.** It says *"look over all the **lit** squares and find those whose jewel you have not yet crouched beside. Among **only those** …"*. The dark squares and the crouched ones are excluded by the recipe itself, so the literal object being scanned is the lit-uncrouched set — which I therefore keep *as* a compact array. This is a data-layout choice, not a method change: no heap, no ordering invariant, just the recipe's own scan over the recipe's own set. The per-row margin minima (`bm`) are a machine-level accelerator of that same scan and return the same element.
2. **"If two tie, take either — it makes no difference."** The recipe explicitly frees the tie-break, which licenses any SIMD scan order.
3. **Step 9's "cross out"** = write the `INFINITY` sentinel the contract asks for.
4. **Required correction, step 5.** Literally *"every cord knotted to it"* would include cords arriving from the other side, i.e. an undirected net — that gives the wrong answer on a *directed* graph. Smallest fix, and the one the seed itself implies (the glint is *"released from the traveler's jewel"* and *"crawls down"*): follow each cord only in the road's direction, away from the crouched jewel. So `head[u]..head[u+1]` holds only the outgoing cords.
5. **Step 7 needs no correction.** As written it would even re-chalk a *crouched* square; with non-negative cord lengths that branch is provably unreachable (`d_u + L >= d_u >= d_v` for settled `v`), so the literal text is already correct.
6. Step 4→9 exit ("if there are none at all") is the loop condition `fn > 0`; no glint is still travelling.

## ARTIFACT

```c
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
```

## PREDICTION

PREDICTION: speedup_vs_heap = 1.8

Cost model behind it, stated before any measurement: per crouch, the recipe's step-4 scan costs `fn/BLK` margin reads + two 64-square re-reads, all contiguous AVX2 min-reductions at ~8 doubles/cycle, ≈ `F/512 + ~90` cycles; steps 5–7 cost one sequential CSR walk plus one random `dist_out[v]` touch per cord, the same random traffic a heap pays, minus all sift/pointer work. I expect a clear win for `n` up to ~10⁵ (frontier small, everything in L2, scan ≈ 200 cycles/crouch vs ~17 cache-missing heap comparisons per edge) and a **loss** at `n ≳ 10⁶` with a dense frontier, where `n·F/512` overtakes `m log n`. 1.8 is my single-number bet on a mid-sized benchmark; if the harness is dominated by very large sparse graphs I expect ~0.4 and I will say so.

## MEASUREMENT

**Not performed. No tools were available in this session** — `dijkstra_bench` and `dijkstra_contract` could not be called, so the prediction above is unvalidated and the "at most four improvements" loop never ran against real numbers. What I state honestly:

- **Measured speedup: unknown.** Nothing here has been compiled or timed by me.
- The three optimizations folded into this single artifact (compact lit-uncrouched frontier instead of a whole-pavement scan; per-64 margin minima; AVX2 min/select) were chosen by the cost model above, **not** by measurement. Any of them could be a wash in practice; the margin-minima layer is the one I would test first by `#define BLK 1`-style ablation, since it is the only piece whose benefit depends on frontier size.
- Correctness I do claim on argument, not on the bench: the selection performed in step 4 is provably the same element a naive scan would pick (margin notes are maintained exactly — assign on block-open, min on decrease, full re-read of both disturbed rows on removal — and the `<=`/paranoid-fallback path cannot select a non-minimal square), and steps 6–7 are unmodified Dijkstra relaxation, exact for non-negative weights. Expected match with the reference: exact bit-for-bit on the same summation order, since every `dist` value is a sum along one specific path taken in one specific order.

## VERDICT

The recipe is Dijkstra's algorithm, told honestly and completely: lit = discovered, dark = undiscovered, crouched-beside = settled, first-flare-only = the decrease-key rule, and step 4's repeated smallest-chalked-number choice is the extract-min. Nothing had to be replaced. Exactly one correction was required — step 5's *"every cord knotted to it"* must be read as outgoing cords only, or the directed graph is silently treated as undirected.

What I want to flag as the interesting part of the translation: the recipe does **not** describe the textbook O(n²) grid sweep that people usually project onto this story. Step 4 says to look over *the lit squares*, not *the squares* — the native already keeps the search confined to the discovered-but-unsettled set. Taking that literally hands you the frontier as a first-class object, and a contiguous frontier plus a margin note per chalk row turns the min-selection into a SIMD stream instead of either a pointer-chasing heap or a full-array sweep. That is a real structure the story gave me, not something I imported.

Where it will fail: on a graph large and dense enough that the frontier stays a sizeable fraction of `n`, the scan is asymptotically worse than a heap, and no amount of vector width fixes `n·F`. The recipe has no cure for that, and I did not bolt one on. Also, the algorithm is inherently sequential — step 4 crouches beside exactly one jewel at a time — so `-fopenmp` buys only the initialization pass; I will not pretend otherwise.

Given no measurement was possible, treat the prediction as a bet, not a result.