## APPROACH

**Mapping the disguised solution onto the real problem.**

| Sunny Hollow | Real implementation |
|---|---|
| tree-houses / rope bridges with one-way signs | `n` nodes, `m` directed edges `src[i] -> dst[i]` of weight `weight[i] >= 0` |
| scratch-guess list, Oak = 0, everyone else "unknown, forever" | tentative-distance array, `0` at `source`, `INFINITY` elsewhere |
| "never trust a shortcut *through* a house until its time is carved in stone" | a node is only relaxed *out of* after it is settled — classic Dijkstra settle-then-relax |
| **"walk past every not-yet-finalized house, compare side by side, pick the smallest"** | **linear array scan for the argmin — no heap at all.** This is the mechanism; I keep it |
| "carve it in stone, then cross its outgoing bridges and lower neighbours' guesses" | write `dist_out[u]`, iterate its CSR adjacency slice, relax |
| "re-scan the whole list each round, gets slower with more houses" | the O(n²) cost I attack — by making the scanned list *smaller* and the scan *wider*, never by adding a priority queue |

**Implementing the same mechanism better (three changes, all faithful to "scan the list"):**

1. **Scan a compacted list, not all `n` slots.** Ricky's own words are "every *not-yet-finalized* house on his list". So I hold the unsettled nodes in a dense parallel pair `aidx[] / ad[]` with an inverse map `pos[]`, and swap-remove on settle. The scanned run shrinks by one every round — it halves the comparisons versus a `done[]`-masked full sweep, and, more importantly, the scan is over **contiguous doubles**, which is what makes step 3 possible.
2. **Don't scan houses marked "no idea yet".** A node still at `INFINITY` can never be the minimum unless *everything* left is `INFINITY` — in which case nothing more is reachable and the custom is finished. So the live list is partitioned `[0, nfin)` = finite guesses, `[nfin, nact)` = untouched. Relaxing a node out of `INFINITY` promotes it across the boundary by one swap; settling removes it by two. The scan length is therefore the **frontier size**, not `n`. On graphs where the reachable frontier is small this collapses O(n²) to near-linear, and the "must explore the whole graph" assumption dies for free: unreachable nodes are never scanned and the loop exits the instant `nfin == 0`.
3. **Vectorise the comparison.** "Compares their current guessed times side by side" is literally a SIMD argmin. AVX-512 path (2×8 lanes) and AVX2 path (4×4 lanes) with independent min/index accumulators so the `vminpd` latency chain doesn't serialise; index carried in the double lanes so `blendv`/`mask_blend` does the bookkeeping. Scalar tail and scalar fallback for `n < 16` keep short scans overhead-free.

CSR is built once by counting sort. No OpenMP: a parallel region costs ~1–2 µs and this kernel does one scan *per settled node*, so threading only pays when the frontier is already large enough that this whole family of algorithms has lost anyway — adding it would buy noise and risk regressions.

**Honest risk:** this mechanism is a genuine win on dense or moderate graphs and on graphs with a modest frontier. If the benchmark is a very large *sparse* graph with a wide frontier (say `n ≳ 200k` with tens of thousands of simultaneously-reached nodes), no amount of SIMD saves an argmin scan and the binary heap wins. I'm told not to switch mechanisms, so I've spent the budget on shrinking what gets scanned instead.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

/* Index of the minimum of a[0..n), n >= 1. No NaNs are ever stored in a[]. */
static int scan_argmin(const double *a, int n)
{
    int i = 0, bi = 0;
    double best = INFINITY;

#if defined(__AVX512F__)
    if (n >= 16) {
        const __m512d inc = _mm512_set1_pd(16.0);
        __m512d m0 = _mm512_set1_pd(INFINITY), m1 = m0;
        __m512d j0 = _mm512_setzero_pd(),      j1 = _mm512_setzero_pd();
        __m512d c0 = _mm512_set_pd(7.0,6.0,5.0,4.0,3.0,2.0,1.0,0.0);
        __m512d c1 = _mm512_add_pd(c0, _mm512_set1_pd(8.0));
        for (; i + 16 <= n; i += 16) {
            __m512d v0 = _mm512_loadu_pd(a + i);
            __m512d v1 = _mm512_loadu_pd(a + i + 8);
            __mmask8 k0 = _mm512_cmp_pd_mask(v0, m0, _CMP_LT_OQ);
            __mmask8 k1 = _mm512_cmp_pd_mask(v1, m1, _CMP_LT_OQ);
            m0 = _mm512_min_pd(m0, v0);
            m1 = _mm512_min_pd(m1, v1);
            j0 = _mm512_mask_blend_pd(k0, j0, c0);
            j1 = _mm512_mask_blend_pd(k1, j1, c1);
            c0 = _mm512_add_pd(c0, inc);
            c1 = _mm512_add_pd(c1, inc);
        }
        {
            double mb[16], ib[16];
            _mm512_storeu_pd(mb,     m0); _mm512_storeu_pd(mb + 8, m1);
            _mm512_storeu_pd(ib,     j0); _mm512_storeu_pd(ib + 8, j1);
            for (int t = 0; t < 16; t++)
                if (mb[t] < best) { best = mb[t]; bi = (int)ib[t]; }
        }
    }
#elif defined(__AVX2__)
    if (n >= 16) {
        const __m256d inc = _mm256_set1_pd(16.0);
        __m256d m0 = _mm256_set1_pd(INFINITY), m1 = m0, m2 = m0, m3 = m0;
        __m256d j0 = _mm256_setzero_pd(), j1 = j0, j2 = j0, j3 = j0;
        __m256d c0 = _mm256_set_pd(3.0,2.0,1.0,0.0);
        __m256d c1 = _mm256_add_pd(c0, _mm256_set1_pd(4.0));
        __m256d c2 = _mm256_add_pd(c0, _mm256_set1_pd(8.0));
        __m256d c3 = _mm256_add_pd(c0, _mm256_set1_pd(12.0));
        for (; i + 16 <= n; i += 16) {
            __m256d v0 = _mm256_loadu_pd(a + i);
            __m256d v1 = _mm256_loadu_pd(a + i + 4);
            __m256d v2 = _mm256_loadu_pd(a + i + 8);
            __m256d v3 = _mm256_loadu_pd(a + i + 12);
            __m256d k0 = _mm256_cmp_pd(v0, m0, _CMP_LT_OQ);
            __m256d k1 = _mm256_cmp_pd(v1, m1, _CMP_LT_OQ);
            __m256d k2 = _mm256_cmp_pd(v2, m2, _CMP_LT_OQ);
            __m256d k3 = _mm256_cmp_pd(v3, m3, _CMP_LT_OQ);
            m0 = _mm256_min_pd(m0, v0); m1 = _mm256_min_pd(m1, v1);
            m2 = _mm256_min_pd(m2, v2); m3 = _mm256_min_pd(m3, v3);
            j0 = _mm256_blendv_pd(j0, c0, k0); j1 = _mm256_blendv_pd(j1, c1, k1);
            j2 = _mm256_blendv_pd(j2, c2, k2); j3 = _mm256_blendv_pd(j3, c3, k3);
            c0 = _mm256_add_pd(c0, inc); c1 = _mm256_add_pd(c1, inc);
            c2 = _mm256_add_pd(c2, inc); c3 = _mm256_add_pd(c3, inc);
        }
        {
            double mb[16], ib[16];
            _mm256_storeu_pd(mb,      m0); _mm256_storeu_pd(mb + 4,  m1);
            _mm256_storeu_pd(mb + 8,  m2); _mm256_storeu_pd(mb + 12, m3);
            _mm256_storeu_pd(ib,      j0); _mm256_storeu_pd(ib + 4,  j1);
            _mm256_storeu_pd(ib + 8,  j2); _mm256_storeu_pd(ib + 12, j3);
            for (int t = 0; t < 16; t++)
                if (mb[t] < best) { best = mb[t]; bi = (int)ib[t]; }
        }
    }
#endif
    for (; i < n; i++)
        if (a[i] < best) { best = a[i]; bi = i; }
    return bi;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    if (m < 0) m = 0;

    size_t msz = (size_t)(m > 0 ? m : 1);

    /* ---- CSR by counting sort ---- */
    int    *off  = (int *)calloc((size_t)n + 1, sizeof(int));
    int    *fill = (int *)malloc((size_t)n * sizeof(int));
    int    *edst = (int *)malloc(msz * sizeof(int));
    double *ew   = (double *)malloc(msz * sizeof(double));

    for (int e = 0; e < m; e++) off[src[e] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    memcpy(fill, off, (size_t)n * sizeof(int));
    for (int e = 0; e < m; e++) {
        int u = src[e];
        int p = fill[u]++;
        edst[p] = dst[e];
        ew[p]   = weight[e];
    }

    /* ---- the scratch-list ----
       aidx[0..nact) : still-unfinalized nodes
       ad  [0..nact) : their current guesses
       slots [0,   nfin) hold finite guesses  (the only ones ever scanned)
       slots [nfin,nact) hold "no idea yet"   (INFINITY)
       pos[v] = slot of v, or -1 once carved in stone                        */
    int    *aidx = (int *)malloc((size_t)n * sizeof(int));
    int    *pos  = (int *)malloc((size_t)n * sizeof(int));
    double *ad   = (double *)malloc((size_t)n * sizeof(double));

    for (int i = 0; i < n; i++) { aidx[i] = i; pos[i] = i; ad[i] = INFINITY; }

    int nact = n, nfin = 0;
    {   /* put the source in slot 0 with guess 0 */
        int ps = pos[source];
        int a0 = aidx[0];
        aidx[ps] = a0;  pos[a0] = ps;
        aidx[0]  = source; pos[source] = 0;
        ad[ps] = INFINITY;      /* order matters when source == 0 */
        ad[0]  = 0.0;
        nfin = 1;
    }

    while (nfin > 0) {
        int    p  = scan_argmin(ad, nfin);
        double dv = ad[p];
        int    u  = aidx[p];

        dist_out[u] = dv;                      /* carved in stone */

        /* remove u: close the finite run, then close the active run */
        {
            int last = nfin - 1;
            if (p != last) {
                int b = aidx[last];
                aidx[p] = b; ad[p] = ad[last]; pos[b] = p;
            }
            nfin = last;
            int la = nact - 1;
            if (nfin != la) {
                int b = aidx[la];
                aidx[nfin] = b; ad[nfin] = ad[la]; pos[b] = nfin;
            }
            nact = la;
            pos[u] = -1;
        }

        /* walk its outgoing bridges */
        {
            int e0 = off[u], e1 = off[u + 1];
            for (int e = e0; e < e1; e++) {
                int v = edst[e];
                int q = pos[v];
                if (q < 0) continue;           /* already final */
                double nd = dv + ew[e];
                if (nd < ad[q]) {
                    if (q >= nfin) {           /* first real guess: promote */
                        int b = aidx[nfin];
                        aidx[q]    = b;  ad[q]    = ad[nfin]; pos[b] = q;
                        aidx[nfin] = v;  ad[nfin] = nd;       pos[v] = nfin;
                        nfin++;
                    } else {
                        ad[q] = nd;
                    }
                }
            }
        }
    }
    /* nfin == 0 -> every remaining node is unreachable; dist_out already INF */

    free(off); free(fill); free(edst); free(ew);
    free(aidx); free(pos); free(ad);
}
```

## PREDICTION

PREDICTION: 3.0

(Speedup factor of this kernel over the reference binary-heap Dijkstra in the problem statement. Reasoning behind the number: the heap version spends ~`m` pushes at ~50–100 cycles each including cache-missing sift-ups on a `2m`-entry array; this version spends ~`m` branch-predictable in-place relaxations plus a SIMD argmin over the frontier at 4–8 doubles/cycle from L1/L2. I expect a large win — 5×+ — on dense/moderate graphs, roughly parity on wide-frontier sparse graphs, so 3.0 is my blended point estimate.)

## MEASUREMENT

**Not measured. Reporting that plainly rather than inventing numbers.** This session was invoked with no tools available — `dijkstra_bench` and `dijkstra_contract` are listed in the task but not callable here, so I could neither run the contract check nor take a single timing, and I therefore used zero of my four improvement rounds. Everything above is static reasoning; the prediction stands unvalidated.

What I did verify by hand, since I could not verify it by machine:

- **Contract:** signature, argument order, and `dist_out` semantics (length `n`, `INFINITY` for unreachable) match exactly. `dist_out` is fully initialised before any early return.
- **Partition invariant:** after each settle, `nact-1 >= nfin`, so the slot vacated at index `nfin` is always backfilled from the `INFINITY` region — the two runs stay contiguous. Traced the three aliasing cases by hand: `p == last`, `la == nfin`, and `q == nfin` in the promotion swap (where `b` aliases `v`; the final two stores still land the correct `nd`).
- **Source-in-slot-0 aliasing:** `ad[ps] = INFINITY` must precede `ad[0] = 0.0` or `source == 0` silently starts the search with every distance infinite. That ordering is load-bearing and commented.
- **Argmin fallback:** an all-`INFINITY` scan returns index 0, which is in range — though the partition makes that state unreachable.
- Degenerate `n <= 0`, `m == 0` (zero-size `malloc`), and out-of-range `source` are guarded.

## VERDICT

An unmeasured artifact, offered as such. I believe it is correct — the invariants are simple enough to check by hand and I did — and I believe it is faster than the heap on the graph shapes this problem was framed around, but "I believe" is not "I measured," and the difference matters. If the harness runs it, the two numbers to watch are the contract check (pass/fail, no partial credit) and whether the benchmark graph's peak frontier is small relative to `n`; the second one decides whether 3.0 was optimistic or conservative.

The one thing I'd flag as a real, not rhetorical, risk: on a large sparse graph with a wide frontier this mechanism is structurally slower than the heap and no implementation effort recovers it. I was instructed to improve the mechanism rather than replace it, so I pushed on the two things that shrink the scanned set — dense compaction and excluding never-reached nodes — which is where the leverage is. If the measurement comes back below 1.0, the honest conclusion is that the disguised solution's mechanism was the wrong one for that input, not that this implementation of it was sloppy.