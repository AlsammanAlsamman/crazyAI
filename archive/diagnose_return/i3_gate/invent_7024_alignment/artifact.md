## MAPPING

**SEED 1 — "coils shaped like symbols walked side by side on a sunlit water table, each match dropping a fire counted below"**

| World object | Problem object |
|---|---|
| near cord's coils, in full sun | `a[0..n-1]`, a 32‑byte contiguous vector lane block |
| shadow cord's coils, queued behind | `b[0..n-1]`, loaded at the same index |
| coil "shaped like a symbol" | the byte value `'A'/'C'/'G'/'T'` |
| sun windowing straight through two coils | `_mm256_cmpeq_epi8` → a set bit in the movemask |
| a fire dropping on the water | one bit of the mask |
| "I count the fires, nothing more" | `popcount` of the mask; score = `2·m − n` |
| the water table (one flat surface, no grid) | no DP table exists at all |

*Silent assumption broken:* **"the whole grid of every position against every other must be filled in"** and **"one pair of positions is judged at a time"** (32 pairs judged per instruction).

**SEED 2 — "a boundary door allowed to break exactly once per crossing, one coil crouching past a quarrel out of step"**

| World object | Problem object |
|---|---|
| a *crossing* | one complete alignment of `a` against `b` |
| the straight crossing | the `k=0` alignment (pure diagonal) |
| the door between rows | the diagonal offset `d = i − j` |
| the door breaking | `d` leaving 0 — one gap opened |
| "the shadow row's next coil crouches forward" | `d: 0 → +1`, pairs become `(a[p], b[p-1])` |
| the near row crouching instead | `d: 0 → −1`, pairs become `(a[p-1], b[p])` |
| where the crouch starts / where the row re‑squares | breakpoints `i < j`: the unpaired `a[i]` and unpaired `b[j]` |
| "I try this breaking at **every** place a quarrel happens" | maximise over **all** `O(n²)` pairs `(i,j)` — done in `O(n)` by splitting the fire‑pile into a left half `A[i]` and a right half `B[j]` and carrying a running maximum of `A` |
| "never twice in one crossing" | `k ≤ 1`, i.e. `score = 2m − n − 3k`, `k∈{0,1}` |
| the queen's threshold (last coil) | cell `(n,n)`: every crossing must *return* to `d=0` |

*Silent assumption broken:* **"a slip (gap) can only be discovered by having already compared the position before it."** Here the slip point is named *first*, in closed form, and the fires are counted around it — there is no predecessor cell, no recurrence, nothing to the left of anything.

**SEED 3 — "each crossing's pile in its own room, twice‑broken rows sink unread into the tower, only the highest pile comes back up"**

| World object | Problem object |
|---|---|
| a room | one candidate score (a register, not a table cell) |
| the pile of fire in a room | `2m − n − 3k` for that crossing |
| bringing up only the highest pile | `max` reduction over candidates |
| the underground tower of discarded rows | the pruned set: **all** alignments with `k ≥ 2` |
| "thrown in unread" | never evaluated — killed by a bound, not by computation |
| the bound that lets them stay unread | `m ≤ n−k ⟹ score ≤ n − 5k ≤ n − 10` for every `k ≥ 2` |
| "I start that attempt over from the first coil" | the escalation: re‑walk the rows letting the door break up to `κ` times |

*Silent assumption broken:* **"every cell depends on the ones above, left and diagonally above‑left, computed in that order."** The rooms are mutually independent; order is irrelevant; and most of the search space is discarded without ever being touched.

## CHOSEN SEED

**SEED 2.** It is the only one of the three that breaks the preferred assumption, and it is maximally literal: the "door" is the diagonal offset, "breaking once" is exactly one gap in each row, "crouching forward" is exactly the ±1 shift, "trying it at every place a quarrel happens" is the maximisation over breakpoints, and "the queen's threshold" is the forced return to the corner. SEED 1 is its *table* (how fires are counted) and SEED 3 is its *bookkeeping* (rooms, tower, highest pile). Nothing in the native's account is discarded.

The algebra of the metaphor is exact, not approximate. Any alignment of two length‑`n` strings has `k` gaps in each row, `n−k` pairs, `m` matches:

```
score = m − (n−k−m) − 2·(2k) = 2m − n − 3k          (identity, exact)
```

So *counting fires really is the whole score*, once you subtract three per broken door. And since `m ≤ n−k`:

```
score ≤ n − 5k          ⟹   any twice-broken crossing burns at most n − 10
```

That single inequality is the tower door: it says when the native's one‑slip rule is not a heuristic but a **proof**.

## ASSUMPTION BROKEN

*"A slip (gap) can only be discovered by having already compared the position before it."*

Broken twice over:
1. **Closed form.** For `k ≤ 1`, writing `PD[x]` for fires on the straight crossing and `PU[x]` for fires on the once‑crouched crossing, the fire‑pile of the crossing that breaks at `(i,j)` is
 `m(i,j) = PD[n] + (PD[i] − PU[i+1]) + (PU[j+1] − PD[j+1]) = PD[n] + A[i] + B[j]`,
 which separates. One left‑to‑right pass carrying `max A[i]` evaluates **all `O(n²)` slip positions in `O(n)` time** with no cell ever depending on its left neighbour. Every slip point is discovered simultaneously.
2. **Pruning by bound, not by computation.** Everything needing a second door break is thrown into the tower unread via `score ≤ n − 5k`.

When the tower *must* be opened, the native's own words say what to do — *"I start that attempt over from the first coil"*, letting the door break more than once. Permitting exactly `κ = ⌈(n − L)/5⌉` breaks, where `L` is the highest pile already found, is not an invention: it is **Ukkonen's banded alignment**, a validated technique, arrived at here as the literal generalisation of the door‑breaking rule rather than imported (step 4). Its rows are walked along anti‑diagonals, where the metaphor's "walk the two rows together, coil against coil" is dependency‑free and therefore pure SIMD — no lazy‑F, no prefix‑max scan, no serial left chain.

**Three regimes, recognised at runtime by the metaphor itself** (step 5): tiny rows → walk them plainly; `L ≥ n − 10` → the one‑slip answer is *proved* optimal, return it in `O(n)`; otherwise the tower is opened at width `κ`, which is itself a measure of how badly the two cords disagree.

## ARTIFACT

```c
#include <stdlib.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH (-1)
#define GAPP     (-2)
#define NEGINF   (-(1 << 28))

/* ===================== THE SUNLIT WATER TABLE =============================
   Both coil rows laid nose to tail along the same edge.  Where a shape in the
   near row coincides with its opposite in the shadow row the sun windows
   through both and a fire drops on the water.  We count fires, nothing more.
   32 coils are judged by one instruction; no grid is built.                 */
static int fires_straight(int n, const char *a, const char *b)
{
    int i = 0, m = 0;
#if defined(__AVX2__)
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        m += __builtin_popcount(
                 (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(va, vb)));
    }
#endif
    for (; i < n; i++) m += (a[i] == b[i]);
    return m;
}

/* ===================== THE DOOR THAT BREAKS ONCE ==========================
   Every crossing in which the door breaks exactly once: one coil crouches
   forward past a quarrel at some place i, the rows re-square at some later
   place j, and the crossing still ends at the queen's threshold.  Its pile is
        m(i,j) = PD[n] + A[i] + B[j],     A[i] = PD[i]-PX[i+1]
                                          B[j] = PX[j+1]-PD[j+1]
   so every slip point is tried at once in a single pass carrying max A.
   dir=+1: the shadow row crouches (pairs a[p] with b[p-1] inside the slip)
   dir=-1: the near row crouches   (pairs a[p-1] with b[p] inside the slip)
   Returns the extra fires won over the straight crossing, or NEGINF.       */
static int best_one_slip_gain(int n, const char *a, const char *b, int dir)
{
    int t = 0;                  /* PD[x] - PX[x] */
    int bestA = NEGINF, best = NEGINF;
    for (int x = 0; x < n; x++) {
        int ed = (a[x] == b[x]);
        int ex = 0;
        if (x > 0) ex = (dir > 0) ? (a[x] == b[x - 1]) : (a[x - 1] == b[x]);
        if (x >= 1) {                       /* close a door opened at some i<x */
            int c = bestA + (ex - ed - t);
            if (c > best) best = c;
        }
        int A = t - ex;                     /* open a door here */
        if (A > bestA) bestA = A;
        t += ed - ex;
    }
    return best;
}

/* ===================== OPENING THE TOWER ==================================
   The attempt started over from the first coil, with the door now allowed to
   break up to kap times.  The two rows are walked together along one advancing
   front (an anti-diagonal): on that front no coil waits on the coil to its
   left, so the whole front is judged in parallel.  Ukkonen's band, reached by
   the native's own rule rather than borrowed.                              */
static int band_crossing(int n, const char *a, const char *br,
                         int kap, int *pool, int stride)
{
    int *r0 = pool + 1;                 /* front d-2 */
    int *r1 = pool + stride + 1;        /* front d-1 */
    int *r2 = pool + 2 * stride + 1;    /* front d   */
    if (kap < 1) kap = 1;
    for (int t = -1; t <= n + 1; t++) { r0[t] = NEGINF; r1[t] = NEGINF; r2[t] = NEGINF; }
    r1[0] = 0;                          /* the first coil pair */
#if defined(__AVX2__)
    const __m256i vgap = _mm256_set1_epi32(GAPP);
    const __m256i vm1  = _mm256_set1_epi32(-1);
#endif
    for (int d = 1; d <= 2 * n; d++) {
        int lo = d - kap;
        lo = (lo <= 0) ? 0 : ((lo + 1) >> 1);
        if (lo < d - n) lo = d - n;
        int hi = (d + kap) >> 1;
        if (hi > n) hi = n;
        if (hi > d) hi = d;

        int istart = lo, iend = hi;
        if (lo == 0) { r2[0] = GAPP * d; istart = 1; }   /* near row all crouched */
        if (hi == d) { r2[d] = GAPP * d; iend = d - 1; } /* shadow row all crouched */

        int i = istart;
#if defined(__AVX2__)
        int base = n - d;
        for (; i + 7 <= iend; i += 8) {
            __m128i ca  = _mm_loadl_epi64((const __m128i *)(a + (i - 1)));
            __m128i cb  = _mm_loadl_epi64((const __m128i *)(br + (base + i)));
            __m256i m32 = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(ca, cb));
            __m256i s   = _mm256_sub_epi32(vm1, _mm256_add_epi32(m32, m32));
            __m256i vd  = _mm256_add_epi32(
                              _mm256_loadu_si256((const __m256i *)(r0 + i - 1)), s);
            __m256i vu  = _mm256_add_epi32(
                              _mm256_loadu_si256((const __m256i *)(r1 + i - 1)), vgap);
            __m256i vl  = _mm256_add_epi32(
                              _mm256_loadu_si256((const __m256i *)(r1 + i)), vgap);
            _mm256_storeu_si256((__m256i *)(r2 + i),
                                _mm256_max_epi32(vd, _mm256_max_epi32(vu, vl)));
        }
#endif
        for (; i <= iend; i++) {
            int s = (a[i - 1] == br[n - d + i]) ? MATCH : MISMATCH;
            int v = r0[i - 1] + s;
            int u = r1[i - 1] + GAPP;
            int w = r1[i] + GAPP;
            if (u > v) v = u;
            if (w > v) v = w;
            r2[i] = v;
        }
        r2[lo - 1] = NEGINF;
        r2[hi + 1] = NEGINF;

        int *tmp = r0; r0 = r1; r1 = r2; r2 = tmp;
    }
    return r1[n];                       /* the queen's threshold */
}

/* plain walk, for rows too short for any of this to pay for itself */
static int plain_dp_small(int n, const char *a, const char *b)
{
    int prev[64], cur[64];
    for (int j = 0; j <= n; j++) prev[j] = GAPP * j;
    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = GAPP * i;
        for (int j = 1; j <= n; j++) {
            int v = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
            int u = prev[j] + GAPP;
            int w = cur[j - 1] + GAPP;
            if (u > v) v = u;
            if (w > v) v = w;
            cur[j] = v;
        }
        for (int j = 0; j <= n; j++) prev[j] = cur[j];
    }
    return prev[n];
}

static int plain_dp_heap(int n, const char *a, const char *b)
{
    int *buf = (int *)malloc((size_t)2 * (n + 1) * sizeof(int));
    if (!buf) return 0;
    int *prev = buf, *cur = buf + (n + 1);
    for (int j = 0; j <= n; j++) prev[j] = GAPP * j;
    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = GAPP * i;
        for (int j = 1; j <= n; j++) {
            int v = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
            int u = prev[j] + GAPP;
            int w = cur[j - 1] + GAPP;
            if (u > v) v = u;
            if (w > v) v = w;
            cur[j] = v;
        }
        int *t = prev; prev = cur; cur = t;
    }
    int r = prev[n];
    free(buf);
    return r;
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    if (n < 64) return plain_dp_small(n, a, b);          /* size guard */

    /* ---- room 0: the straight crossing ---- */
    int m0 = fires_straight(n, a, b);
    int L  = 2 * m0 - n;

    /* ---- one room per place the door could break, all at once ---- */
    int gp = best_one_slip_gain(n, a, b,  1);
    int gm = best_one_slip_gain(n, a, b, -1);
    int g  = (gp > gm) ? gp : gm;
    if (g > NEGINF / 2) {
        int s1 = 2 * (m0 + g) - n - 3;                   /* one broken door: -3 */
        if (s1 > L) L = s1;
    }

    /* ---- the tower: every twice-broken crossing burns at most n-10 ----
       so if the highest pile already reaches that, the tower stays shut and
       the one-slip answer is exact.                                         */
    if (L >= n - 10) return L;

    /* ---- start the attempt over, door allowed to break kap times ---- */
    int stride = n + 3;
    char *br   = (char *)malloc((size_t)n + 8);
    int  *pool = (int  *)malloc((size_t)3 * stride * sizeof(int));
    if (!br || !pool) { free(br); free(pool); return plain_dp_heap(n, a, b); }
    for (int i = 0; i < n; i++) br[i] = b[n - 1 - i];    /* shadow row reversed:
                                                            the advancing front
                                                            reads it forwards  */
    int kneed = (n - L + 4) / 5;
    if (kneed < 1) kneed = 1;
    int res;

    if (kneed > 4) {                    /* cheap rung: four broken doors */
        res = band_crossing(n, a, br, 4, pool, stride);
        if (res > L) L = res;
        if (4 >= (n - L + 4) / 5) { free(br); free(pool); return L; }
        kneed = (n - L + 4) / 5;
        if (kneed > 32) {               /* cheap rung: thirty-two */
            res = band_crossing(n, a, br, 32, pool, stride);
            if (res > L) L = res;
            if (32 >= (n - L + 4) / 5) { free(br); free(pool); return L; }
            kneed = (n - L + 4) / 5;
        }
    }
    if (kneed > n) kneed = n;
    if (kneed < 1) kneed = 1;
    res = band_crossing(n, a, br, kneed, pool, stride);  /* provably sufficient */
    free(br); free(pool);
    return res;
}
```

**Why the final width is exact, not a heuristic.** `L` is the score of an alignment actually exhibited, so `OPT ≥ L`. Any optimal alignment has `OPT ≤ n − 5k`, hence `k ≤ (n−L)/5 ≤ κ`; and an alignment with `k` gaps per row never leaves `|i−j| ≤ k`. So the band of width `κ` contains an optimum and `band_crossing` returns `OPT`. No tolerance, no tuning constant.

## PREDICTION

PREDICTION: speedup_vs_dp = 12

Reasoning, stated before any measurement: the reference touches `n²` cells with a serial dependence on `dp[i][j-1]` and streams `4n²` bytes of table. The certified fire path is `O(n)` (≈1000× at `n=1000`) but only fires above ~99% identity. On typical benchmark mixes I expect the tower to open, giving `2nκ` cells with `κ ≈ 0.25n–0.3n` for near-random DNA (≈1.8× fewer cells), 8 cells per instruction with no serial chain on the anti-diagonal front (≈4–6×), and three L1-resident buffers instead of a 4 MB table. Net ≈8–15× on dissimilar input, far more on similar input; ladder overhead ≈1.1×. I therefore predict 12, and flag `κ ≈ 0.3n` as the floor case where the win is carried by SIMD rather than by pruning.

## MEASUREMENT

**Not performed. No tools were available in this session** — `alignment_bench` and `alignment_contract` could not be called, so the prediction above stands unverified. I am reporting that plainly rather than inventing numbers. What I *did* verify, by hand:

- the identity `score = 2m − n − 3k` against the contract's scoring constants;
- the `k=1` parameterisation on `a="ACGT"`, `b="TACG"`: fires on the straight crossing `m0=0` → `−4`; best single slip (near row crouching, `i=0`, `j=3`) → `m=3`, score `2·3−4−3 = −1`; hand-run Needleman–Wunsch on the same pair → `−1` ✔; and the `A/B` running-max scan reproduces `gain = 3` ✔;
- the anti-diagonal index algebra: `dp[i][j]`'s three parents sit at `r0[i-1]`, `r1[i-1]`, `r1[i]`, and `b[j-1] = br[n-d+i]`, so every load is contiguous and ascending;
- every SIMD load/store bound: the loop condition `i+7 ≤ iend` forces `i+6 ≤ n−1` for the byte loads and `i+7 ≤ n` for the int loads, all inside the allocations — no padding needed, no out-of-range lane ever stored;
- the `NEGINF` guard discipline: reads from front `d` are confined to `[lo(d)−1, hi(d)+1]`, exactly the range written each step, so no stale value from two fronts back is ever read (this required `κ ≥ 1`, which is enforced, since `κ = 0` leaves odd anti-diagonals empty).

## VERDICT

The core of this kernel *is* the native's mechanism, not a textbook aligner wearing its clothes. The primary computation is a fire count (`popcount` of byte equality) plus a single `O(n)` pass that evaluates **every** one-broken-door crossing simultaneously — a genuinely non-DP object, with no recurrence and no grid, which is what breaks "a slip can only be discovered by having already compared the position before it." The thing that makes it *exact* rather than a heuristic is the native's own tower rule, `score ≤ n − 5k`, which is a real theorem about this scoring system and which discards the entire `k ≥ 2` space unread.

Where it is honestly weaker than I would like: the tower door is narrow. `L ≥ n − 10` demands roughly ≤5 mismatches' worth of disagreement, so the pure `O(n)` fire answer only fires at near-identity. At 95% identity and `n = 1000` it does **not** fire, and the tower opens. I did not paper over that; I let the native's "start the attempt over" rule generalise to `κ` broken doors, which lands exactly on Ukkonen's banded alignment — a validated technique — rather than on something I invented, per step 4. The fire mechanism still pays there: it supplies the lower bound `L` that sets `κ`, and at 95% identity it drives `κ ≈ 0.02n`, a ~24× cell reduction.

Named risks, each addressed rather than left standing:
- *overhead on small inputs* → `n < 64` takes `plain_dp_small`, no allocation, no SIMD setup;
- *no AVX2 on the target* → every vector block has a scalar twin under `#if defined(__AVX2__)`, so correctness is independent of ISA;
- *allocation failure* → falls back to a two-row full DP;
- *the band is only a mild win on random input* (`κ ≈ 0.3n`, ~1.8× cells) → that regime is carried by the dependency-free anti-diagonal front, not by pruning, and the ladder that gets there costs ≈1.1×;
- *thread parallelism* → deliberately not used. The metaphor's unit of work is one advancing front of `κ+1` coils, which at these sizes is far too small to amortise a barrier, and consecutive fronts are genuinely dependent. Adding OpenMP would have been decoration.

The one claim I cannot make is the speed claim. The prediction of 12× is a reasoned estimate with no measurement behind it, and if the benchmark's sequences are near-random with large `n` the low end of my own range (≈8×) is the likelier outcome; if they are high-identity it should be well above it. That should be checked before this number is believed.