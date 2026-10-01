No tools were reachable in this session (including `dijkstra_bench`), so everything below the PREDICTION line is reported as *unmeasured*. I state that plainly rather than inventing numbers.

---

# DICTIONARY

| Recipe object | Computational object |
|---|---|
| a **place** in the land | a node index `v ∈ [0,n)` |
| a **road** | a directed edge `e`: `src[e] -> dst[e]` |
| the road's **near end** / **far end** | `src[e]` / `dst[e]` |
| a **heap of sticks** at a place | the multiset of candidate distances arriving at `v` during the current throw |
| a **stick notched to a number** | one `double` value |
| the **ember** of a heap | `ember[v]` — the current best known distance |
| **cold ash, no ember** | `ember[v] == INFINITY` (distance still unknown) |
| the **round wooden disc etched with one character** | the `source` argument |
| **kindling** the zero-notched stick | `ember[source] = 0.0` |
| the road's **notched roadside stick** (step 2) | `notch[p]` — every edge weight materialized once into in-CSR order, paired with `nearend[p]` |
| **one throw** | one synchronous Bellman–Ford relaxation round over all `m` edges |
| **all heaps throw together, single land-wide throw** | the round barrier: *every* stick in the throw is computed from the pre-throw `ember[]` only — enforced by double buffering (`ember` read, `banked` written) |
| the **pyramid** | the throw's candidate structure; sticks landing on a heap cross and settle into that heap's one shortest sliver |
| **a row read straight across** (one sliver per heap, same breath) | the length-`n` candidate vector swept in one parallel phase; no place is judged before another because no new ember is visible until the sweep ends |
| **settling** | `min`-combination of the sticks that landed on a heap |
| **keeping the single shortest** (step 6) | `keep = min(ember[v], sliver)` |
| **burning the slivers** (step 7) | discarding the transient candidates — nothing but the ember survives the throw |
| **banked to a coal** | the `ember ↔ banked` pointer swap |
| **noting whether any ember changed** | `changed` flag, OR-reduced across threads |
| **counting throws**, cap of *places minus one* | round counter capped at `n-1` |
| a **ring of roads that gives back more than it takes** | a negative cycle |
| an **unsound heap** | node whose ember is untrustworthy — unreachable under this contract (`weight >= 0`) |
| **unlit heap / dead field the elephants never walked** | `INFINITY` in `dist_out` |
| the **hooded figure reading all embers aloud in one pass** | the final single copy of `ember[]` into `dist_out[]` |

**Ambiguity resolutions (most literal reading taken in each case):**

1. **Steps 4–5, the pyramid.** Step 5 says the row holds *"one sliver from every heap"* while step 6 says *"compare the slivers against one another"* (plural, per heap). The only consistent reading is that the crossing/settling of a heap's many sticks **is** the pyramid's formation, and the row exposes each heap's already-settled shortest sliver. So the pyramid is not a DRAM array of `m` sticks: a heap's sticks combine as they land, and the row is the length-`n` sliver vector. That vector is loop-fused into the same sweep, so each heap's sliver lives in a register. The *semantic* content of step 4 — the barrier, "none before another" — is preserved exactly by the double buffer, so the result is bit-identical to materialising the pyramid.
2. **Step 3, "if that heap has no ember, the road does nothing."** Implemented branchlessly: an emberless heap has `ember = INFINITY`, so its stick is notched to `INFINITY + w = INFINITY` and can never be the shortest kept. Arithmetically identical to skipping the road, but vectorizable. (No NaN is possible: all weights are finite and non-negative.)
3. **Step 3, walking order.** I walk each road exactly once per throw, but grouped by its **far** end (in-CSR) rather than in arbitrary order. Same roads, same count, same throw — only the visiting order, i.e. data layout, which I'm allowed to choose. This removes every atomic.
4. **Step 10, the negative ring.** Under this contract all weights are `>= 0`, so no ring can give back more than it takes and the "mark as unsound" branch is provably dead. I implement step 10 as the throw cap only (`n-1`), which is where its real force lies. Nothing is marked, because nothing can be unsound.
5. **Step 11.** Since "cold ash" is already encoded as `INFINITY`, the explicit pass over unlit heaps is a no-op; I keep it as a real pass anyway, for fidelity — it costs one streaming read of `n` doubles.

**No step was added, removed, or replaced.** In particular I did **not** add a per-heap "changed" mask / active frontier / delta-stepping bucket structure. The recipe permits exactly one skip (step 3's emberless road) and exactly one piece of cross-throw bookkeeping (step 8's single land-wide flag), and that is all this kernel has.

---

# ARTIFACT

```c
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
```

Four machine-level choices I applied while writing it (design iterations, **all unmeasured** — see MEASUREMENT):

1. **Pull layout (in-CSR by far end) instead of push + atomic-min on the heaps.** A literal push throw needs an atomic `min` on a `double`, i.e. a CAS loop per road, with heavy contention on high-in-degree heaps. Grouping the roadside sticks by far end makes every heap the sole writer of its own ember: zero atomics, zero contention.
2. **Pyramid collapsed to one sliver per heap, held in a register** rather than an `m`-long stick array in DRAM. Saves ~`16m` bytes of write+read traffic *per throw* with no change in the answer.
3. **Branchless `INFINITY` arithmetic + `omp simd reduction(min:)`** so the inner road walk compiles to `vgatherdpd` + `vminpd` instead of a data-dependent branch per road.
4. **`schedule(guided)`** (in-degree is usually skewed, so static chunking strands one thread on the hubs) and **`if(m > 8192)` / `if(n > 8192)`** so tiny lands don't pay team-launch cost per throw.

---

# PREDICTION: speedup_vs_heap = 0.6

Reasoning, stated before any measurement: the recipe is *synchronous Bellman–Ford*. Its cost is `R · m` where `R` is the hop-depth of the shortest-path tree plus one; a heap Dijkstra's cost is `m + n log n` but with expensive, serial, cache-missing heap operations. So the trade is a factor `R` of extra work against a factor of maybe 5–15× from parallel streaming edge relaxation.

- Random graph, `n ~ 10^6`, `m ~ 10^7`: weighted SP trees are deeper than the unweighted diameter, so I expect `R ≈ 20–35`. Against a ~10× per-edge throughput win, that lands **near 0.3–0.8**.
- Low-hop-diameter or dense graph (`R ≈ 5–10`) on 16 cores: **1.5–3×**.
- Grid / road-network-like land (`R` in the hundreds or thousands): **0.01–0.1×** — a rout.

I am predicting a single number for the likely mid-case, and I expect the recipe to **lose** to the heap. The honest reason is structural, not implementational: step 3 forbids any notion of an *active* heap. It permits skipping only the roads out of a heap with **no** ember, never the roads out of a heap whose ember didn't change last throw. So the kernel re-walks all `m` roads every throw even when a handful of embers moved. That is the entire gap, and it cannot be closed without adding a step the native didn't give.

---

# MEASUREMENT

**Not performed — `dijkstra_bench` and `dijkstra_contract` were not reachable in this session** (no tools were available at all). I will not report numbers I did not take, and I have not back-fitted the prediction above to anything.

What the pipeline should find when it compiles and runs this:

- **Correctness:** should match the reference exactly, not merely within tolerance, on all-distinct-shortest-path instances. Each ember is formed as `ember[u] + w` from a *settled* `u`, which is the same floating-point addition sequence along the same path that Dijkstra performs, so the bits agree. Where ties exist, a different equal-cost path may be selected and the sum may differ in the last ulp — inside tolerance.
- **Speed:** I expect `speedup_vs_heap` well under 1 on any graph with hop depth above ~15, and above 1 only on shallow, wide graphs with enough cores. If the measured value is far below 0.3, the diagnostic is `R`: instrument `throws` and check it against `m`-relaxations-per-throw. If the measured value came out above 1.5 on a large sparse graph, I'd distrust my own reasoning before celebrating and would check that the reference heap isn't pathologically slow.
- **Risk items I could not compile-check:** `reduction(min:sliver)` on a `double` requires OpenMP ≥ 4.0 (fine for any modern gcc); if that `#pragma omp simd` line is rejected, deleting it changes nothing but vectorization quality.

---

# VERDICT

The recipe is a **complete and correct** algorithm for the target problem, and it is not the textbook one: it is synchronous, barrier-per-round Bellman–Ford with double buffering, a global change flag, an `n-1` round cap, negative-ring detection, and explicit unreachable handling. Every object in the native's world mapped onto something concrete with no leftovers and no gaps — the roadside notched sticks are a real preprocessing pass, the pyramid's "row read in one breath" is a real and load-bearing constraint (it is exactly what forbids Gauss–Seidel in-place updates and makes the answer deterministic), and the "dead field the elephants never walked" is exactly `INFINITY`. The one place the recipe is genuinely under-determined — whether a heap's many sticks settle on landing or on reading — I resolved to *on landing*, which is the reading that makes steps 5 and 6 consistent with each other.

Where the recipe is weak, and I won't paper over it: **step 3 has no notion of a heap whose ember stood still.** A heap with a stale ember re-notches and re-carries sticks down every one of its roads, throw after throw, and they are all burned in step 7. For a typical sparse land this wastes the large majority of all work, and no data layout or vector instruction recovers it. Dijkstra's heap exists precisely to touch each road once. The single smallest change that would fix it — a per-heap "my ember moved last throw" mark, gating step 3 on it (an active frontier) — would likely swing this from ~0.6 to well above 1, and a delta-stepping bucket structure would go further. **I did not make that change**, because step 3 says the road does nothing only when the near heap "has no ember," and step 8 gives exactly one flag for the whole land; adding a per-heap flag would be substituting my method for the native's. If the measured speedup is poor, that is the recipe's honest score, not a bug in the translation.

*(Note: the `claude.ai` PubMed connector shown in this environment requires authorization via your claude.ai connector settings before its tools can be used; it was not needed here.)*