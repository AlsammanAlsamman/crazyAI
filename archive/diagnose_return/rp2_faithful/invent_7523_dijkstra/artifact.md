## MAPPING

| Native's world-object | Problem-object | |
|---|---|---|
| **SEED 1** — *"A runner walks each road carrying its notch and lays a new stick on the far heap equal to the home ember's length plus that notch."* | | |
| place | node `v` | |
| road | directed edge `src[i] -> dst[i]` | |
| notch on the stick | `weight[i]` | |
| ember at a place | `dist[u]` (current upper bound) | |
| runner going out | one relaxation `dist[u] + w` | |
| "runners down every road **at once**" | all edges relaxed in one unordered sweep | |
| new stick laid on far heap | a *candidate* value deposited at `v`, **not** written into `dist[v]` yet | |
| **Breaks:** *"a road can only be considered once its starting place is fully settled"* and *"each place's distance must be finalized before its neighbors are explored."* A runner leaves from a merely-lit ember, never a settled one. | | |

| Native's world-object | Problem-object |
|---|---|
| **SEED 2** — *"Every heap throws its sticks together in one land-wide throw, and the row read straight across picks each heap's shortest sliver as its new ember while the rest are burned for fuel."* | |
| heap of sticks at a place | the set of **in-edges** of `v`: candidates `dist[u]+w` |
| "one throw for the whole land at once" | one synchronous round over all active heaps |
| "read **straight across a single row**, one sliver from **every heap in the same breath**" | vectorise **across heaps, not within one heap**: slice `k` of four adjacent heaps read as one SIMD lane-group → sliced-ELLPACK (chunk = 4 rows, column-major within the chunk) |
| shortest sliver becomes the new ember | `dist[v] = min(dist[v], min_e(dist[u_e]+w_e))` — a **(min,+) sparse mat-vec**, pull-style, no scatter, no atomics |
| longer slivers burned | the non-winning candidates are simply never stored |
| **Breaks:** *"a priority structure must be consulted before every relaxation."* There is no priority structure at all; the ordering work is replaced by a lane-wise `vminpd`. |

| Native's world-object | Problem-object |
|---|---|
| **SEED 3** — *"A garrison seals a heap once its ember stops changing between throws, fixing its distance for good and halting its runners."* | |
| garrison sealing a heap | `sealed[v] = 1` → `v` is final, permanently removed from the work set |
| "its ember will not shorten again **no matter what still arrives**" | a **finality certificate**, proved locally, not by ranking |
| "roads into garrisoned places — those sticks the runners drop by the wayside" | in the push pass, edges whose head is sealed are skipped (dead-edge pruning) |
| "I feed every dropped stick straight into the nearest hearth… a fire that isn't given fuel goes out" | the saved work is the fuel: sealed rows are never thrown again, so each round is cheaper than the last; the throw keeps burning only because sealing keeps feeding it |
| "its own runners stop going out for good" | a sealed node never re-enters the frontier |
| "a heap whose ash never once catches a spark" | `dist[v] == INFINITY` at quiescence → unreachable |
| the hooded figure reading all embers in one pass | final linear copy of the working ember array into `dist_out` |
| **Breaks:** *"the next place to finalize is found by comparing against every remaining place."* Finalisation here is a **threshold sweep that seals many places at once**, never an argmin over the remaining set — and the threshold is read off the *newest embers only* (the frontier), not off the whole land. |

## CHOSEN SEED

**SEED 3 — the garrison.** It is the one of the three that breaks the preferred assumption (*"the next place to finalize is found by comparing against every remaining place"*), and it is the furthest from the known way: Dijkstra's whole identity is *one* finalisation per `extract-min`, chosen by global ranking. The garrison finalises in *batches*, by certificate, with no ranking anywhere in the program. Seeds 1 and 2 are the substrate the garrison rides over (the same native's same land), so they are built too — but the garrison is the mechanism under test.

**One honest deviation, stated up front.** The native's *test* for the garrison — "its ember matched exactly what it held after the throw before" — is **unsound**, and I can show it: `s→v` weight 10, `s→a` 1, `a→b` 1, `b→v` 1. After throw 1, `dist[v]=10`; after throw 2, `dist[v]=10` again (unchanged) → the native seals `v` at 10; throw 3 would have brought 3. So I kept the garrison's *function and its claim* ("will not shorten again no matter what still arrives") and replaced only its *trigger* with one derived from the same world and actually provable:

> **Garrison rule (sound):** let `mu` = the shortest of the embers that *just changed in this throw*, and `wmin` = the shortest road in the land. Nothing shorter than `mu + wmin` can ever still arrive anywhere. So every unsealed heap whose ember is `<= mu + wmin` is sealed, all at once.

*Proof:* every value ever stored is the length of a real path, so `dist >= true_dist` always. Suppose `v` is not yet final; take its shortest path `P` and let `z` be the last vertex on `P` with `dist[z] == true_dist[z]`, `x` its successor. Then `dist[x] > dist[z] + w(z,x)`, so edge `(z,x)` has not been relaxed since `dist[z]` took its value — which can only happen if `z` changed *in this very throw* (a node that changed in an earlier throw had all its out-edges relaxed in the next one, by construction of the runner pass; skipped edges only ever point at already-sealed heads, which are final by induction). Hence `z` is in the frontier, and `true_dist[v] >= true_dist[x] = dist[z] + w(z,x) >= mu + wmin`. ∎

Note what this rule is *not*: it is not an argmin over the remaining places. It is a min over the **frontier** (the newly lit embers) plus a threshold sweep that seals an arbitrary number of heaps in one ride. The banned assumption stays broken.

## ASSUMPTION BROKEN

*"The next place to finalize is found by comparing against every remaining place."* — replaced by a batch finality certificate. Also broken, as a consequence of the substrate: *"a priority structure must be consulted before every relaxation"* (there is no heap), *"a road can only be considered once its starting place is fully settled"* (every lit ember sends runners), and *"each place's distance must be finalized before its neighbors are explored."*

## ARTIFACT

Which code implements which part of the native:

| Native | Code |
|---|---|
| one ember at the traveler, cold ash elsewhere | `D[source]=0`, rest `INFINITY`; `D` is padded to a whole number of chunks so the ghost heaps at the edge of the land need no special case |
| roads out of each place (runners) | forward CSR `foff/fadj` |
| the heap of sticks standing at each place | reverse adjacency — **sliced ELLPACK** `coff/esrc/ew`, chunk of 4 rows, column-major |
| **the throw**, "one sliver from every heap in the same breath" | the AVX2 inner loop: `_mm256_i32gather_pd` of four *different heaps'* `k`-th sticks, `_mm256_add_pd` the four notches, `_mm256_min_pd` into a 4-lane accumulator. Four heaps' winners fall out with **no horizontal reduction at all** — that is exactly "read straight across a single row" |
| shortest sliver becomes the ember, the rest burned | `_mm256_min_pd(acc,cur)` + `_mm256_movemask_pd` to learn which lanes shortened; losers are never stored |
| runners sent, sticks laid | the push pass marking receiving **chunks** (`inq/chunks`) — a hamlet of four heaps throws together |
| sticks dropped by the wayside at garrisoned places | `if (sealed[v]) continue;` in the push pass |
| **the garrison** | the `mu`/`thr` block + the `unsealed[]` compaction sweep — seals in batches, amortised so the ride only happens when enough new embers have settled to be worth it |
| dropped sticks fed to the hearth so the fire keeps burning | sealed heaps leave `unsealed[]` and can never be re-activated, so every later throw is strictly cheaper — the pruning *is* the fuel |
| ash that never catches a spark | nodes still at `INFINITY` at quiescence |
| hooded figure reads the whole land in one pass | final `memcpy`-style copy `D -> dist_out` |
| **regime: is this a land where the runners are on every road anyway?** | `fedges*4 >= m` → skip the marking and throw the **whole land** at once (dense / wide-frontier rounds) |
| **regime: are the heaps ragged or even?** | padded size `> 2m+8n` (or no AVX2) → fall back to ragged CSR rows with 4 independent scalar accumulators instead of ELLPACK |
| **regime: is this a long thin land the throw can't warm?** | the fuel counter `work`/`rounds` — if the throws burn more than 16 sweeps' worth of sticks, or 2048 throws pass unfinished, the native gives up and **calls the hooded figure for the slow reading** (heap Dijkstra), correct by construction |
| too small a land to be worth runners | `n < 64` → slow reading directly |

Thread parallelism is deliberately **absent**: the instruction is to reach for SIMD first, and the throw's races would make the garrison's `mu` nondeterministic. The whole speed claim rests on the lane-wise throw, not on cores.

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#define HAVE_AVX2 1
#endif

/* ================= the hooded figure's slow reading: heap Dijkstra ================= */
typedef struct { double d; int u; } HeapItem;

static void hf_push(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++;
    h[i].d = d; h[i].u = u;
    while (i > 0) { int p = (i - 1) >> 1; if (h[p].d <= h[i].d) break;
                    HeapItem t = h[p]; h[p] = h[i]; h[i] = t; i = p; }
}
static HeapItem hf_pop(HeapItem *h, int *hs) {
    HeapItem top = h[0]; (*hs)--; h[0] = h[*hs];
    int i = 0;
    for (;;) {
        int l = 2 * i + 1, r = l + 1, s = i;
        if (l < *hs && h[l].d < h[s].d) s = l;
        if (r < *hs && h[r].d < h[s].d) s = r;
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t; i = s;
    }
    return top;
}
static void slow_reading(int n, int m, const int *src, const int *dst,
                         const double *weight, int source, double *dist_out)
{
    int *off  = (int *)calloc((size_t)n + 1, sizeof(int));
    int *adj  = (int *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *aw = (double *)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    int *fill = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    memcpy(fill, off, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) { int p = fill[src[i]]++; adj[p] = dst[i]; aw[p] = weight[i]; }
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;
    char *done = (char *)calloc((size_t)n, 1);
    HeapItem *h = (HeapItem *)malloc((size_t)(m + 2) * sizeof(HeapItem));
    int hs = 0; hf_push(h, &hs, 0.0, source);
    while (hs > 0) {
        HeapItem t = hf_pop(h, &hs);
        int u = t.u;
        if (done[u]) continue;
        done[u] = 1;
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = adj[e]; double nd = dist_out[u] + aw[e];
            if (nd < dist_out[v]) { dist_out[v] = nd; hf_push(h, &hs, nd, v); }
        }
    }
    free(off); free(adj); free(aw); free(fill); free(done); free(h);
}

/* ============================== the native's land ================================= */
void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    if (source < 0 || source >= n) { for (int i = 0; i < n; i++) dist_out[i] = INFINITY; return; }
    if (m <= 0) { for (int i = 0; i < n; i++) dist_out[i] = INFINITY; dist_out[source] = 0.0; return; }
    if (n < 64) { slow_reading(n, m, src, dst, weight, source, dist_out); return; }  /* too small a land */

    const int C = 4;
    int nch  = (n + C - 1) / C;
    int npad = nch * C;

    /* ---- roads out of every place (runners), and the shortest road in the land ---- */
    int *foff  = (int *)calloc((size_t)n + 2, sizeof(int));
    int *indeg = (int *)calloc((size_t)npad + 1, sizeof(int));
    double wmin = INFINITY;
    for (int i = 0; i < m; i++) {
        foff[src[i] + 1]++;
        indeg[dst[i]]++;
        if (weight[i] < wmin) wmin = weight[i];
    }
    if (!(wmin >= 0.0)) wmin = 0.0;
    for (int i = 0; i < n; i++) foff[i + 1] += foff[i];
    int *fadj = (int *)malloc((size_t)m * sizeof(int));
    int *fill = (int *)malloc((size_t)n * sizeof(int));
    memcpy(fill, foff, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) fadj[fill[src[i]]++] = dst[i];

    /* ---- the heap of sticks at every place: are the heaps even or ragged? ---- */
    int *clen = (int *)malloc((size_t)nch * sizeof(int));
    long long total = 0;
    for (int c = 0; c < nch; c++) {
        int b = c * C, mx = 0;
        for (int l = 0; l < C; l++) { int d = indeg[b + l]; if (d > mx) mx = d; }
        clen[c] = mx; total += (long long)mx * C;
    }
    int use_ell = 0;
#ifdef HAVE_AVX2
    if (total <= 2LL * m + 8LL * n) use_ell = 1;      /* even enough heaps -> lane-wise throw */
#endif

    size_t *coff = NULL; int *esrc = NULL; double *ew = NULL;
    int *roff2 = NULL, *rsrc = NULL; double *rw = NULL;

    if (use_ell) {
        coff = (size_t *)malloc(((size_t)nch + 1) * sizeof(size_t));
        coff[0] = 0;
        for (int c = 0; c < nch; c++) coff[c + 1] = coff[c] + (size_t)clen[c] * (size_t)C;
        size_t tot = coff[nch];
        esrc = (int *)malloc((tot ? tot : 1) * sizeof(int));
        ew   = (double *)malloc((tot ? tot : 1) * sizeof(double));
        for (int c = 0; c < nch; c++) {                 /* neutral padding slivers */
            int b = c * C, L = clen[c]; size_t base = coff[c];
            for (int l = 0; l < C; l++)
                for (int k = indeg[b + l]; k < L; k++) {
                    size_t idx = base + (size_t)C * (size_t)k + (size_t)l;
                    esrc[idx] = 0; ew[idx] = INFINITY;
                }
        }
        int *rfill = (int *)calloc((size_t)npad, sizeof(int));
        for (int i = 0; i < m; i++) {
            int v = dst[i], c = v >> 2, l = v & 3, k = rfill[v]++;
            size_t idx = coff[c] + (size_t)4 * (size_t)k + (size_t)l;
            esrc[idx] = src[i]; ew[idx] = weight[i];
        }
        free(rfill);
    } else {
        roff2 = (int *)malloc(((size_t)npad + 1) * sizeof(int));
        roff2[0] = 0;
        for (int i = 0; i < npad; i++) roff2[i + 1] = roff2[i] + indeg[i];
        rsrc = (int *)malloc((size_t)m * sizeof(int));
        rw   = (double *)malloc((size_t)m * sizeof(double));
        int *rfill = (int *)malloc((size_t)npad * sizeof(int));
        memcpy(rfill, roff2, (size_t)npad * sizeof(int));
        for (int i = 0; i < m; i++) { int v = dst[i], p = rfill[v]++; rsrc[p] = src[i]; rw[p] = weight[i]; }
        free(rfill);
    }

    /* ---- embers, garrisons, runners ---- */
    double *D = (double *)malloc((size_t)npad * sizeof(double));
    for (int i = 0; i < npad; i++) D[i] = INFINITY;
    D[source] = 0.0;
    char *sealed   = (char *)calloc((size_t)npad, 1);
    int  *unsealed = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) unsealed[i] = i;
    int nuns = n, pending = 0;
    int *fr = (int *)malloc((size_t)npad * sizeof(int));
    int *nx = (int *)malloc((size_t)npad * sizeof(int));
    int *chunks = (int *)malloc((size_t)nch * sizeof(int));
    char *inq   = (char *)calloc((size_t)nch, 1);
    int nf = 1; fr[0] = source;

    long long work = 0, budget = 16LL * m + 32LL * n;   /* the fuel counter */
    int rounds = 0, bailed = 0;

    while (nf > 0) {
        rounds++;
        long long fedges = 0;
        for (int i = 0; i < nf; i++) { int u = fr[i]; fedges += foff[u + 1] - foff[u]; }

        int nc = 0;
        if (fedges * 4 >= (long long)m) {          /* runners on every road anyway: whole land throws */
            for (int c = 0; c < nch; c++) chunks[nc++] = c;
            work += nf;
        } else {                                    /* send runners, mark the receiving hamlets */
            for (int i = 0; i < nf; i++) {
                int u = fr[i];
                for (int e = foff[u]; e < foff[u + 1]; e++) {
                    int v = fadj[e];
                    if (sealed[v]) continue;        /* stick dropped by the wayside */
                    int c = v >> 2;
                    if (!inq[c]) { inq[c] = 1; chunks[nc++] = c; }
                }
            }
            work += fedges;
        }

        int nn = 0;
        if (use_ell) {
#ifdef HAVE_AVX2
            for (int j = 0; j < nc; j++) {
                int c = chunks[j]; inq[c] = 0;
                int L = clen[c];
                if (L == 0) continue;
                const int    *es = esrc + coff[c];
                const double *ws = ew   + coff[c];
                __m256d acc = _mm256_set1_pd(INFINITY);
                for (int k = 0; k < L; k++, es += 4, ws += 4) {   /* one sliver from every heap */
                    __m128i ix  = _mm_loadu_si128((const __m128i *)es);
                    __m256d du  = _mm256_i32gather_pd(D, ix, 8);
                    __m256d wv  = _mm256_loadu_pd(ws);
                    acc = _mm256_min_pd(acc, _mm256_add_pd(du, wv));
                }
                int b = c * 4;
                __m256d cur = _mm256_loadu_pd(D + b);
                __m256d nd  = _mm256_min_pd(acc, cur);
                int mask = _mm256_movemask_pd(_mm256_cmp_pd(nd, cur, _CMP_LT_OQ));
                if (mask) {
                    _mm256_storeu_pd(D + b, nd);
                    if (mask & 1) nx[nn++] = b;
                    if (mask & 2) nx[nn++] = b + 1;
                    if (mask & 4) nx[nn++] = b + 2;
                    if (mask & 8) nx[nn++] = b + 3;
                }
                work += 4LL * L;
            }
#endif
        } else {
            for (int j = 0; j < nc; j++) {
                int c = chunks[j]; inq[c] = 0;
                int b = c * 4;
                for (int l = 0; l < 4; l++) {
                    int v = b + l;
                    int e = roff2[v], en = roff2[v + 1];
                    if (e == en) continue;
                    work += (en - e);
                    double a0 = INFINITY, a1 = INFINITY, a2 = INFINITY, a3 = INFINITY;
                    for (; e + 3 < en; e += 4) {
                        double t0 = D[rsrc[e]]     + rw[e];
                        double t1 = D[rsrc[e + 1]] + rw[e + 1];
                        double t2 = D[rsrc[e + 2]] + rw[e + 2];
                        double t3 = D[rsrc[e + 3]] + rw[e + 3];
                        if (t0 < a0) a0 = t0;
                        if (t1 < a1) a1 = t1;
                        if (t2 < a2) a2 = t2;
                        if (t3 < a3) a3 = t3;
                    }
                    for (; e < en; e++) { double t = D[rsrc[e]] + rw[e]; if (t < a0) a0 = t; }
                    if (a1 < a0) a0 = a1;
                    if (a3 < a2) a2 = a3;
                    if (a2 < a0) a0 = a2;
                    if (a0 < D[v]) { D[v] = a0; nx[nn++] = v; }
                }
            }
        }

        /* ---- the garrison rides in ---- */
        if (nn > 0) {
            double mu = INFINITY;
            for (int i = 0; i < nn; i++) { double d = D[nx[i]]; if (d < mu) mu = d; }
            double thr = mu + wmin;                 /* nothing shorter can still arrive */
            pending += nn;
            if ((long long)pending * 8 >= (long long)nuns) {
                int k2 = 0;
                for (int i = 0; i < nuns; i++) {
                    int v = unsealed[i];
                    if (D[v] <= thr) sealed[v] = 1; else unsealed[k2++] = v;
                }
                work += nuns; nuns = k2; pending = 0;
            }
        }

        { int *t = fr; fr = nx; nx = t; }
        nf = nn;
        if (nf > 0 && (work > budget || rounds >= 2048)) { bailed = 1; break; }
    }

    if (!bailed) for (int i = 0; i < n; i++) dist_out[i] = D[i];   /* read the land aloud */

    free(foff); free(indeg); free(fadj); free(fill); free(clen);
    if (use_ell) { free(coff); free(esrc); free(ew); }
    else         { free(roff2); free(rsrc); free(rw); }
    free(D); free(sealed); free(unsealed); free(fr); free(nx); free(chunks); free(inq);

    if (bailed) slow_reading(n, m, src, dst, weight, source, dist_out);  /* long thin land */
}
```

## PREDICTION

**PREDICTION: speedup_vs_heap = 2.0**

Reasoning stated before any measurement: on graphs with small hop-diameter the throw converges in a handful of rounds, so total edge traffic is ~6–8× `m` of branch-free, gather-add-min work at 4 heaps per instruction, against the reference's `m` relaxations plus ~`m` sift operations of pointer-chasing with unpredictable branches. I expect 2–3× on random sparse graphs, more on dense ones (where the hop-diameter is 2–3 and the throw is nearly pure SIMD), and ~0.6–0.9× on any graph that trips the bail-out, since that work is wasted before Dijkstra runs.

## MEASUREMENT

**Not performed — and I will not pretend otherwise.** This session was invoked with no tools available: `dijkstra_bench` and `dijkstra_contract` could not be called, so the prediction above is unfalsified. The code is written against the fixed contract signature exactly as given and is self-contained (`gcc -O3 -march=native -fopenmp -lm`), but *no compile, no correctness check against a reference, and no timing has actually happened here.* Treat every number above as a claim awaiting the pipeline, not a result. If the harness reports a mismatch, the first places to look are the ELLPACK index arithmetic (`coff[c] + 4*k + l`) and the garrison threshold, in that order.

## VERDICT

The garrison is a real mechanism and it survives translation — but only after its trigger was repaired. The native's literal rule ("unchanged for one throw ⇒ sealed") is demonstrably wrong, and I gave the four-node counterexample rather than quietly softening it. What I kept is the *shape* of his idea, which is the valuable part and which the textbook does not have: **finalisation as a batch certificate rather than a ranking**. Dijkstra seals exactly one place per `extract-min` and pays `log n` for the privilege; the garrison seals an unbounded batch per throw and pays one comparison per unsealed place, amortised. The `mu + wmin` threshold is derived from the native's own furniture (the newest embers and the shortest road), not imported from Δ-stepping's vocabulary.

The sharpest thing the native gave me is not the garrison at all, though — it is *"one sliver from every heap in the same breath."* That single phrase forbids the obvious implementation (reduce one node's in-edges, then the next) and demands the transposed one (take slice `k` of four nodes at once). That is sliced-ELLPACK, and it is genuinely better: it turns short, latency-bound `min` chains into four independent lanes with **no horizontal reduction anywhere in the hot loop**. I would not have written it that way on my own.

Risks I named and then addressed rather than shipped bare: (1) *only helps when the hop-diameter is small* → the fuel counter and round cap bail out to heap Dijkstra, bounded waste; (2) *ELLPACK explodes on skewed in-degree* → padded size is measured before commitment and a ragged-CSR path takes over; (3) *overhead on tiny graphs* → `n < 64` goes straight to the slow reading; (4) *wide vs. narrow frontier* → decided per round by whether the runners already own a quarter of the roads. Thread parallelism was considered and dropped: the throw races would make `mu` nondeterministic and I have no measurement here to justify the risk.

The honest bottom line: the mechanism is sound, the regimes are guarded, the metaphor was followed rather than replaced — and the speed claim is still a prediction, not a finding.