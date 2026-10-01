## DICTIONARY

| Recipe object | Computational object |
|---|---|
| the land / a cross-marked board | a graph node `0..n-1` |
| a road leaving a board | a directed out-edge `src[i] -> dst[i]` |
| strides of road beneath a bird / hoarseness | `weight[i]` (non-negative), hoarseness grows with distance |
| a note; quieter = fainter | a `double` distance value; **quieter = numerically smaller** |
| the silence, "no strides at all, faintest sound there is" | `0.0` |
| blank slot (uncarved) | `INFINITY` |
| **the rigid lattice of old memory**, one fixed slot per board | `double dist[n]` — written directly into `dist_out` (no copy) |
| a slot's carving | `dist[v]`, the current tentative distance |
| **the second, smaller lattice**, one plain bead per slot | `unsigned char settled[n]` |
| bead unsettled / settled | `settled[v] == 0` / `== 1` |
| settling a bead (step 4) | permanently finalizing `dist[u]`; node never revisited |
| **a nightingale** | one relaxation of one out-edge; born at a settled board, dies at the far board |
| "sings itself to death at the first board its road reaches" | each bird traverses exactly one edge — no propagation past `dst` |
| **a cow's head / the woman milking the smaller cow** | the two-input adder: `dist[u] + weight[e]` |
| **her milk** | the candidate distance `nd` |
| thinner / fatter milk | smaller / larger `double` |
| "spill it in the desert" | discard `nd`, no write |
| a **dead road** | an edge whose far board is already settled (step 9's definition; see below) |
| the desert, "blank as the pink-brown desert with no landmark" | `INFINITY` in the output = unreachable |
| *(my data layout, not a step)* the set in step 3 — "slots unsettled and not blank" | a compacted candidate array `fidx[]`/`fdist[]` + `pos[n]`, holding exactly that set |
| *(my data layout)* "every road leaving it" | CSR out-adjacency `head[]`, `adj_v[]`, `adj_w[]` |

**Ambiguities, resolved by the most literal reading**

- *quieter / thinner* = smaller; *fatter* = larger. The whole recipe is a min-selection over sums, so "faintness" is magnitude.
- **Step 3 ties** are unspecified. Any minimum is a valid Dijkstra choice; I take the first one found in the candidate array.
- **Step 8** ("keep nothing else. What's thrown away stays thrown away") I read as an explicit prohibition on auxiliary state: **no predecessor array, no second-best, no re-insertion.** It is therefore a zero-instruction block, which is the point of it.
- **Step 9** marks dead any road whose bird brought nothing thinner *and* whose far board is already settled. Because a settled carving is final and weights are ≥ 0, `dist[v] ≤ dist[u] ≤ dist[u]+w` for any settled `v`, so **every** road into a settled board brings nothing thinner. Hence "dead road" ≡ "road whose far board is settled", and step 5's own instruction *"send no bird down a road already marked dead"* becomes `if (settled[v]) continue;`. This is the recipe's test, placed where the recipe places it.
- "never a second bird down a road already flown" is automatic: each node is settled once (step 4), so each out-edge is relaxed exactly once. No `dead[m]` bitmap is needed or consulted.
- **No step is wrong.** This is exactly Dijkstra with linear-scan selection, and it is correct for non-negative weights. No step required modification.

**What I deliberately did *not* do:** step 3 is a *scan* of the candidate slots. I did not replace it with a binary heap. That is the native's idea and the whole performance question here, so it stays a scan — hardened with AVX2 and an optional threaded reduction, and narrowed to exactly the slots the recipe names (unsettled ∧ non-blank), never all `n`.

## ARTIFACT

```c
/* Dijkstra by the Nightingale recipe: linear-scan selection over the
   candidate slots, AVX2-hardened. No heap anywhere. */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

/* --- machine-level helpers for step 3 (the scan for the quietest carving) --- */

static double lat_min(const double *fd, int len)
{
    double best = INFINITY;
    int i = 0;
#if defined(__AVX2__)
    if (len >= 16) {
        __m256d b0 = _mm256_set1_pd(INFINITY), b1 = b0, b2 = b0, b3 = b0;
        for (; i + 16 <= len; i += 16) {
            b0 = _mm256_min_pd(b0, _mm256_loadu_pd(fd + i));
            b1 = _mm256_min_pd(b1, _mm256_loadu_pd(fd + i + 4));
            b2 = _mm256_min_pd(b2, _mm256_loadu_pd(fd + i + 8));
            b3 = _mm256_min_pd(b3, _mm256_loadu_pd(fd + i + 12));
        }
        b0 = _mm256_min_pd(b0, b1);
        b2 = _mm256_min_pd(b2, b3);
        b0 = _mm256_min_pd(b0, b2);
        __m128d lo = _mm256_castpd256_pd128(b0);
        __m128d hi = _mm256_extractf128_pd(b0, 1);
        lo = _mm_min_pd(lo, hi);
        lo = _mm_min_sd(lo, _mm_unpackhi_pd(lo, lo));
        best = _mm_cvtsd_f64(lo);
    }
#endif
    for (; i < len; i++) if (fd[i] < best) best = fd[i];
    return best;
}

static double lat_min_par(const double *fd, int len)
{
#ifdef _OPENMP
    if (len >= (1 << 16)) {            /* only when the scan is genuinely long */
        double g = INFINITY;
        #pragma omp parallel
        {
            int nt = omp_get_num_threads(), id = omp_get_thread_num();
            long long a = (long long)len * id / nt;
            long long b = (long long)len * (id + 1) / nt;
            double lm = lat_min(fd + a, (int)(b - a));
            #pragma omp critical
            { if (lm < g) g = lm; }
        }
        return g;
    }
#endif
    return lat_min(fd, len);
}

static int lat_find(const double *fd, int len, double best)
{
    int p = 0;
#if defined(__AVX2__)
    {
        __m256d bb = _mm256_set1_pd(best);
        for (; p + 4 <= len; p += 4) {
            __m256d v = _mm256_loadu_pd(fd + p);
            int mk = _mm256_movemask_pd(_mm256_cmp_pd(v, bb, _CMP_EQ_OQ));
            if (mk) return p + (int)__builtin_ctz((unsigned)mk);
        }
    }
#endif
    for (; p < len; p++) if (fd[p] == best) return p;
    return 0;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;

    /* ---------------- step 1 ----------------
       Lay out the rigid lattice of old memory: one fixed slot for every
       cross-marked board, no more and no fewer, each cut blank (= INFINITY).
       Beside it the second, smaller lattice: same slots, one plain bead each,
       unsettled (= 0).
       Layout only (not a step of its own): the land is also laid out as CSR
       out-adjacency so step 5 can take "one nightingale for each road leaving"
       a board, and a compacted view `fidx/fdist/pos` holds exactly the slot set
       step 3 looks over (unsettled and not blank). */
    double *dist = dist_out;                                  /* the lattice   */
    unsigned char *settled = (unsigned char *)malloc((size_t)n);
    int    *pos   = (int *)malloc((size_t)n * sizeof(int));
    int    *fidx  = (int *)malloc((size_t)n * sizeof(int));
    double *fdist = (double *)malloc((size_t)n * sizeof(double));
    int    *head  = (int *)malloc((size_t)(n + 2) * sizeof(int));
    int    *adj_v = (int *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *adj_w = (double *)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    if (!settled || !pos || !fidx || !fdist || !head || !adj_v || !adj_w) {
        for (int i = 0; i < n; i++) dist[i] = INFINITY;
        free(settled); free(pos); free(fidx); free(fdist);
        free(head); free(adj_v); free(adj_w);
        return;
    }

    #pragma omp parallel for schedule(static) if(n > 65536)
    for (int i = 0; i < n; i++) { dist[i] = INFINITY; settled[i] = 0; pos[i] = -1; }

    memset(head, 0, (size_t)(n + 2) * sizeof(int));
    for (int i = 0; i < m; i++) head[src[i] + 2]++;
    for (int u = 2; u <= n + 1; u++) head[u] += head[u - 1];
    for (int i = 0; i < m; i++) {                 /* head[u+1] used as cursor  */
        int q = head[src[i] + 1]++;
        adj_v[q] = dst[i];
        adj_w[q] = weight[i];
    }                    /* now head[u]..head[u+1) are the roads leaving u     */
    int fsize = 0;

    /* ---------------- step 2 ----------------
       Into the slot for the traveler's own board carve the silence: the note of
       no strides at all, the faintest sound there is. Every other slot blank.
       (It is now unsettled-and-not-blank, so it enters the step-3 view.) */
    if (source >= 0 && source < n) {
        dist[source] = 0.0;
        fidx[0] = source; fdist[0] = 0.0; pos[source] = 0; fsize = 1;
    }

    for (;;) {
        /* ---------------- step 3 ----------------
           Look over all slots whose bead is still unsettled and whose carving
           is not blank; take the one whose carving is quietest. If no such slot
           exists, go to step 11. (Ties: the first such slot found.) */
        if (fsize == 0) break;                           /* -> step 11 */
        double best = lat_min_par(fdist, fsize);
        int p = lat_find(fdist, fsize, best);
        int u = fidx[p];
        double du = best;

        /* ---------------- step 4 ----------------
           Turn that slot's bead to settled. A settled bead is never turned
           back and its board is never returned to: so it leaves the view of
           step 3 forever (last candidate swapped into its place). */
        settled[u] = 1;
        {
            int last = fsize - 1;
            int mv = fidx[last];
            fidx[p] = mv; fdist[p] = fdist[last]; pos[mv] = p;
            pos[u] = -1; fsize = last;
        }

        /* ---------------- step 5 ----------------
           Stand at that settled board. Take one nightingale for each road
           leaving it and loose them all at once; each sings itself to death at
           the first board its road reaches. Send no bird down a road already
           marked dead (step 9: a road into an already-settled board), and never
           a second bird down a road already flown (automatic: u is settled once,
           so each of its roads is flown exactly once). */
        for (int e = head[u], e1 = head[u + 1]; e < e1; e++) {
            int v = adj_v[e];
            if (settled[v]) continue;                    /* dead road */

            /* ---------------- step 6 ----------------
               At the board where the bird falls, open a cow's head: the woman
               draws the settled board's own carved note with that dead bird's
               hoarseness added onto it. */
            double nd = du + adj_w[e];

            /* ---------------- step 7 ----------------
               Compare her milk against what is carved in that fallen board's
               slot. Blank, or thinner than the carving: scrape clean and cut
               her milk in its place. Same or fatter: spill it in the desert. */
            if (nd < dist[v]) {
                dist[v] = nd;
                int q = pos[v];
                if (q < 0) { q = fsize++; fidx[q] = v; pos[v] = q; }
                fdist[q] = nd;
            }

            /* ---------------- step 8 ----------------
               Where two dying notes reach the same board, the comparing above
               has already kept the fainter and let the louder rot. Keep nothing
               else: no predecessor, no second-best, no record of the discarded.
               What's thrown away stays thrown away. (Zero instructions.) */

            /* ---------------- step 9 ----------------
               Mark dead any road whose bird brings nothing thinner and whose
               far board is already settled, and send no bird down it again.
               Since a settled carving is final and weights are >= 0, that is
               exactly "every road into a settled board" -- carried in the
               `settled` beads and consulted by step 5's guard above. No bird
               ever revisits a road anyway. */
        }
        /* ---------------- step 10 ----------------
           Return to step 3. */
    }

    /* ---------------- step 11 ----------------
       Finished: no unsettled slot holds a carving -- no nightingale is left to
       sing and no cow yields a thinner milk. Every reachable board bears its
       one permanent carving, the shortest way there. Every slot still blank is
       a place past reach, blank as the desert (INFINITY). */

    /* ---------------- step 12 ----------------
       Read the traveler the whole lattice in order, slot by slot: each board,
       its carved note, and the desert for the blank ones. The lattice was
       carved in `dist_out` itself, already in slot order, so it is read out as
       it stands. */
    free(settled); free(pos); free(fidx); free(fdist);
    free(head); free(adj_v); free(adj_w);
}
```

## PREDICTION

PREDICTION: speedup_vs_heap = 0.6

Reasoning stated before any measurement: step 3 is a scan, so selection costs `Σ|candidates|` instead of a heap's `O(log n)` per pop. Narrowing the scan to exactly the recipe's set (unsettled ∧ non-blank) plus AVX2 buys ~8 doubles/cycle, which should beat a pointer-chasing binary heap outright on small-to-medium `n` (crossover I estimate around `n ≈ 10^5`) and on geometric graphs where the candidate set stays `O(n^{2/3})`. On a large random sparse graph the candidate set is a constant fraction of `n` through the middle of the run, giving `~0.4n²` scanned elements, and the scan loses by roughly an order of magnitude. Weighting those cases, I expect to land just under parity — hence 0.6, with a wide honest band of roughly 0.05–4 depending entirely on the benchmark's `n` and graph structure.

## MEASUREMENT

**Not measured.** `dijkstra_bench` and `dijkstra_contract` were listed as available but this session was invoked with no tool access, so I could not compile, verify against the reference, or run a single timing. I am reporting that plainly rather than inventing numbers: there are zero measurements behind this artifact, and the four permitted improvement rounds were never spent. The optimizations present (compacted candidate view, AVX2 two-pass argmin, threaded reduction above 65536 candidates, dead-road skip, in-place output) are design-time choices justified by operation counts, not by evidence.

What a run should check, in order: (1) exact agreement with the reference on reachable nodes and `INFINITY` on unreachable ones; (2) `Σ|candidates|` — instrument it, because it alone decides the verdict; (3) whether the `1<<16` parallel threshold ever fires and whether it helps or hurts when it does.

## VERDICT

The recipe is Dijkstra, stated with unusual precision — the settled/unsettled bead pair, the once-only edge relaxation, the `< ` versus `≤` comparison in step 7, the prohibition in step 8 on keeping anything discarded, and step 11's characterization of blank slots as unreachable are all exactly right, and step 9's "dead road" turns out to be a derivable consequence rather than extra bookkeeping. **No step needed correcting.**

The one thing the recipe does *not* contain is a priority queue. Step 3 says *look over all slots and take the quietest*, and I implemented that literally rather than substituting the textbook heap. That is the entire performance story, and it is a real gamble: the native's selection rule is asymptotically worse (`O(n² + m)` vs `O((n+m) log n)`), and whether the constant factors rescue it depends on facts about the benchmark graph I was unable to obtain. My honest expectation is that it comes in slightly below the heap on a typical large sparse benchmark; if `n` is modest or the graph is geometric, it should win comfortably. Unverified either way.