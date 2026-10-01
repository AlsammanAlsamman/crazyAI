I had no tools in this session (no `alignment_bench`, no `alignment_contract`, no shell), so nothing below is measured. I say so plainly in MEASUREMENT rather than inventing numbers.

---

# DICTIONARY

| Thing in the recipe's world | Concrete computational thing |
|---|---|
| The scratched **road of lines and dots** | The alignment grid's index axis; house *m* = a column position. Its houses are the DP cells' row index `i`. |
| **Upper furrow** (pressed pawns, "never moves again") | `a`, read-only, copied once into a padded buffer `up` (padding only so 8-byte SIMD loads can't walk off a page). |
| **Lower furrow** (pawns "loose enough to shove") | `b`. Stored *reversed* as `lowrev` so that, on one anti-diagonal, the pawns facing consecutive upper houses are contiguous in memory: `b[j-1] = lowrev[(n-k)+i]` for `j = k-i`. |
| A **pawn's carved crown-beast** | One nucleotide byte, `{A,C,G,T}`. |
| **Beasts face each other in a house** | One byte compare — `_mm_cmpeq_epi8`. |
| **Lotus petal** (same beast) | Cost contribution 0. |
| **Red knot** (differ, or pawn facing an empty house) | Cost contribution. *Weighted* (repair R2): differing pair = 4 grains, house facing an empty one = 5 grains. |
| **The knotted thread of a trial** | The total cost of one alignment arrangement. |
| **Weighing a fist of knots** | Integer sum; "lightest" = `min`. |
| **The horse's-head piece** | Insertion of exactly one gap at one position = one unit slip of a row. |
| **The horse standing nowhere** | Offset 0: the plain diagonal / Hamming arrangement. |
| **Horse set down in house k, pawns shuffle one forward** | The alignment `a[0..k) ~ b[0..k)`, then `a[k]` vs gap, then `a[m] ~ b[m-1]`; the pawn shoved off the road's end is `b[n-1]` vs gap. Two gap columns, i.e. `g=1` in each string — a legal alignment. |
| **The flute-player's one low note per house** | One dependency tick, *not* a ban on simultaneity: the notes order houses so that no house is marked before the houses it leans on. The set of houses that no longer lean on each other is exactly one **anti-diagonal** `i+j=k`, so eight of them share one breath (one AVX2 register). This is my reading of the ambiguity in step 5. |
| **Time** | The anti-diagonal index `k = i+j`, from 0 to 2n. |
| **Memory that flows** | Three rotating anti-diagonal buffers `A0,A1,A2` (`i`-indexed), holding diagonals `k, k-1, k-2`. |
| **Memory that stays still** | `up`, `lowrev`, and the suffix-knot tables. |
| **Processor** | One core; the 8 int32 lanes of one AVX2 register are eight walkers marking eight houses on one breath. No OpenMP (see VERDICT). |
| **Off the road / outside the furrow** | Sentinel `GR_BIG = 2^27`, written just below `lo` and just above `hi` of every diagonal. Provably enough because `lo` is non-decreasing and `hi` grows by ≤1 per diagonal. |
| **Flinging the other threads past the eaves to the flying fish** | `free()` of the scratch pool — literally, the discarded threads are released memory. |
| **"How far the two strings disagree"** | grains; converted to the contract's score by `score = n - grains/2`. |

**Repairs (step 3 of the brief: the recipe as written gives wrong answers).**

- **R1 — step 2 + step 9 (the real bug).** The recipe permits *one* slip, only in the lower furrow. That is a heuristic, not the alignment score. Counter-example for the one-slip restriction's direction: a row may need slipping in the *upper* furrow; and for long strings, two slips in opposite directions can beat any single slip. Smallest change: **the horse may be set down again after each slip, and in either furrow.** The family of arrangements then becomes exactly the set of alignments — and it is affordable only because of R3.
- **R2 — step 5 (currency).** One uniform knot cannot express the contract. With `M = n-g` pairs and `x` mismatches, `score = n - 2x - 5g`, so a mismatch and an empty house are *not* worth the same. Smallest change: knots have weights — 4 grains for a differing pair, 5 grains for a house facing an empty one — and step 12 carries back `n - grains/2`.
- **R3 — step 11 (weighing moved earlier).** Re-walking every arrangement and weighing only at the road's end is unaffordable once R1 enlarges the family. Knots are additive along the road, so the lightest thread reaching a given house-pair has lightest prefixes (Bellman): weigh house-by-house instead of at the end. That fusion of step 11 into step 5 *is* the dynamic program. I am not hiding this: R1+R3 is where the recipe becomes an aligner.

**What genuinely fell out of the literal reading.** The recipe insists on walking the unslipped trial *first*, keeping its thread, and only then moving the horse. That ordering is not decoration — it is a **pruning certificate**. Any arrangement with `g` slips weighs at least `10g` grains, so if the lightest thread found so far weighs `Bgrains`, every optimal arrangement obeys `g ≤ ⌊B/10⌋`. Hence the horse provably never needs to wander further than `W = ⌊B/10⌋` houses, and the walk is a band of half-width `W` — **exact, not heuristic**. Better still, the recipe's own step-9 family (all one-slip trials, both furrows) can be weighed in `O(n)` with prefix/suffix knot counts, giving a tighter `B` before the big walk starts. Consequences: identical or near-identical strings (`x₀ ≤ 2`) give `W = 0` and the answer in `O(n)` with no walk at all; random DNA (`x₀ ≈ 0.75n`) gives `W ≈ 0.3n`, i.e. 40% of the cells never touched.

---

# ARTIFACT

```c
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
```

---

# PREDICTION

**PREDICTION: speedup_vs_dp = 9**

Reasoning, before any measurement: against a scalar full-matrix DP at ~1 cell/ns, two independent factors multiply — band area `(2W+1)/n ≈ 0.6` for random DNA (1.7×) and AVX2 int32 anti-diagonal lanes at an effective 5× (8 lanes minus short-diagonal and bookkeeping losses). If the benchmark uses *similar* sequences (mutated copies), `W` collapses and the number should be 30–150× instead, with `W = 0` giving an `O(n)` answer. If it uses very short sequences (n ≤ 256), setup and the two `O(n)` passes dominate and I expect 2–4×.

---

# MEASUREMENT

**Not measured.** `alignment_bench` and `alignment_contract` were not available in this session, so I ran zero compilations and zero timings, and I made zero of my four permitted improvement rounds. The artifact above is a *design*, verified only by hand:

- Hand-checked by hand-trace: `n=1` both cases; `"AT"/"TA"` → −2 (gaps would give −3); `"ACAC"/"CACA"` → −1 via `W=1`, where the diagonal gives −4; the `k ≤ 2` rim cells reproduce `C[1][1] = min(sub, 10)`, i.e. score ±1.
- Proved rather than tested: the band bound (`10g ≤ grains`, so every optimum has `g ≤ ⌊B/10⌋`); grain-parity at `(n,n)` (`4x + 10g` is even, so `cost/2` is exact); sentinel sufficiency (`lo` non-decreasing, `hi` grows by ≤1, so all reads land on live cells or on written `GR_BIG`); and `GR_BIG = 2^27` poison never undercutting a legitimate `≤ 5n` cost while growing by at most `10n`.
- **Unverified and therefore the honest risk list:** (1) the reference aligner's exact boundary convention — I assumed textbook `H[i][0] = -2i`; (2) whether the bench's `n` is large enough for int32 lanes to be the right choice; (3) any off-by-one in the rim branch `k ≤ W` that my hand-traces (`k ≤ 2`) did not reach; (4) the `-march=native` machine actually having AVX2 (there is a scalar fallback, which would land near 1.7× — band only).

What would falsify the prediction: a measured speedup < 1.7× on random DNA would mean the band or the SIMD is not doing what I claim; a *wrong answer* would most likely indict the rim branch or R2's currency conversion, not the band (which is proved).

---

# VERDICT

The recipe is **not** an aligner, and I will not pretend otherwise. Its step 9 — one slip, lower furrow only — is a heuristic that fails whenever the best arrangement needs two slips or a slip of the fixed row, and its step 5 counts all knots alike, which cannot express match +1 / mismatch −1 / gap −2. Those are the two places I changed it (R1, R2), plus moving the weighing from thread-end to house-by-house (R3) so the enlarged family is walkable at all. With R1+R3 the walk *is* a dynamic program; I am not going to dress that up as something else.

What survived translation, and earned its keep: the native's ordering. Walking the unslipped trial **first** and keeping its thread is exactly a pruning certificate — `W = ⌊grains/10⌋` bounds how far the horse can ever pay to wander, and that bound is *provable*, not heuristic, so the banded walk is exact for every input. The recipe's own one-slip family, which looks like wasted effort next to a DP, becomes the cheap `O(n)` way to tighten that certificate before the expensive walk, and for identical or near-identical strings it removes the walk entirely. A textbook implementation would never compute the Hamming distance first; the native's insistence on it is the whole reason this kernel can be sublinear-area. The other survivor is the flute: "one note per house" read as *dependency order* rather than as serialism is precisely the anti-diagonal wavefront, which is what makes the inner loop vectorizable at all.

The four improvement rounds I did not get to run, in the order I would run them: (1) int16 lanes with a clamp-to-ceiling per step for `n ≤ 4000` — 16 lanes instead of 8, my best guess at a further 1.6×, deliberately omitted here because I could not test the overflow discipline and an untested overflow is a wrong answer, not a slow one; (2) processing two anti-diagonals per pass to halve the per-diagonal bookkeeping, which dominates when `W` is small; (3) a blocked wavefront over the band for OpenMP — plain `#pragma omp parallel for` inside the diagonal loop would pay ~2n barriers and lose, so it is absent by decision, not oversight; (4) an iterated certificate (run the band at a small `W`, use the resulting score to shrink `W`, repeat) which could cut random-DNA band area further. Until someone runs `alignment_bench`, the number in PREDICTION is a claim, not a result.