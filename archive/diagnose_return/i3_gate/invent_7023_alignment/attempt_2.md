## 1. SEED → PROBLEM MAPPINGS

**SEED A — "the low sun throws one doubled shadow where two knots share a dye"**

| world | problem |
|---|---|
| two grass-ropes, knots dyed 4 berry-colors | the two length-*n* strings over {A,C,G,T} |
| a knot | one character |
| a crossing | a DP cell (i,j) |
| the low sun striking both ropes at once | one SIMD compare of a broadcast character against a whole window of the other rope |
| two dyes agreeing → **one** shadow instead of two | `a^b == 0`, i.e. `_mm_cmpeq_epi*` — equality is "the xor throws no second shadow" |
| finding every treasure without walking the room | the whole row's match mask produced in one instruction, never per-pair |

Breaks: **"one pair of positions is judged at a time."**

**SEED B — "the bare floor's cost is read off how many knots forward each rope must go, never walked"**

| world | problem |
|---|---|
| bare floor between two treasures | the run of non-match cells the gap/left-chain walks through |
| "how many knots forward on the wall-rope, how many on the floor-rope" | the index difference `j − k` |
| corridor cost = that count, read, not paced | gap cost = `2·(j−k)` — a **closed-form arithmetic function of the coordinates only** |
| counting doorframes instead of pacing | fold `2j` into the coordinate: define `y[j] = D[i][j] + 2j`; then the whole left-corridor collapses to `y[j] = max(W[j], y[j−1])` — a plain running max with **no decay term at all** |
| "the corridor's cost was never a secret — only its length was" | the gap penalty is known a priori; only the distance is data — so it can be pre-paid by a change of frame |

Breaks: **"every cell depends on the ones above, to the left, and diagonally above-left, computed in that order."** The left dependency — the single thing that makes Needleman–Wunsch un-vectorizable along a row — is *dissolved*, not respected. It also breaks the grid-filling assumption in the precise sense the native means: the corridor between treasures is never stepped through, it is jumped by one `max`.

**SEED C — "keep only the cheapest arriving tally, slip spent or slip whole; drop the rest"**

| world | problem |
|---|---|
| a slip (one rope idles, the other runs ahead) | one gap column |
| slip spent / slip still whole | the two incoming states at a cell |
| "drop the bad thread-end, don't pick it back up" | dominance pruning |
| slips must be paid for and the ropes must fall back into step | each slip costs 5 net against the gapless score ⇒ `g ≤ (n − L)/5` is *provable*, so every crossing farther than that from the main diagonal is a tally already known to be dominated and is dropped without ever being formed |

Breaks: **"the whole grid of every position against every other must be filled in"** — but only via a bound, and only usefully when the ropes are nearly in step.

## 2. CHOSEN SEED

**SEED B.** It is the most literal (its object — "cost = knots forward, read not paced" — is a exact arithmetic identity, not an analogy), and the most different from the known way, which is *defined* by walking that corridor cell by cell in a fixed order. SEED A is real but is only a match-finder; SEED C is real but degenerates to a band.

Honesty note required by the brief: none of the three seeds breaks *"the whole grid must be filled in"* in a way that stays **exact** on its own. SEED C comes closest and I take it, but only in its provable form, as a secondary regime guard — not as the core. I say that plainly rather than pretending a sparse match-chaining core is viable: over a 4-letter alphabet the treasures are ~n²/4, so a literal treasure-only chain is *slower* than the grid, and I will not ship a mechanism I know is worse.

## 3. ASSUMPTION BROKEN

> *every cell depends on the ones above, to the left, and diagonally above-left, **computed in that order***

The left-neighbour dependency is removed by a change of frame. Since the corridor's *price* (2 per knot) was never secret and only its *length* is data, pre-pay it into the coordinate:

$$y_i[j] \;=\; D[i][j] + 2j$$

$$y_i[j] \;=\; \max\Big(\underbrace{y_{i-1}[j-1] + (2+s)}_{\text{diagonal}},\; \underbrace{y_{i-1}[j] - 2}_{\text{vertical}},\; \underbrace{y_i[j-1]}_{\textbf{corridor — free}}\Big)$$

with `2+s ∈ {3,1}`. The left term has **no penalty left in it**. A whole row is therefore: an elementwise map (vectorizes trivially) followed by a **prefix maximum** (a log-depth SIMD scan). The answer is `y_n[n] − 2n`. This is exactly the scan-formulation used in validated GPU Smith–Waterman work (Khajeh-Saeed et al.), so the mechanism *arrives at* a known-good technique rather than inventing one.

## 4. WORLD → MACHINE

| world | machine |
|---|---|
| the two ropes | `const char *a`, `const char *b` (b pre-widened to int16 once) |
| the empty room / crossings | never materialised — no n² table exists |
| what stays still | the two 16-bit rope-rows `y_prev`, `y_cur` (O(n), L1-resident) |
| what flows | `carry` — the single running "best tally so far" sliding rightward |
| the processor | one XMM register = 8 crossings judged simultaneously |
| time | the row index *i*; each tick advances the wall-rope one knot |
| the low sun | `_mm_cmpeq_epi16(b_window, broadcast(a_i))` — one instruction, 8 shadows |
| the corridor, read not paced | `3 × (alignr + max)` = log₂8 steps for 8 cells of corridor |
| dropping the worse tally | `_mm_max_epi16` |
| the cat counting main-diagonal treasures | SSE `cmpeq_epi8` + `movemask` + `popcount` → the slip budget `w = (n − (2M₀ − n))/5` |

Regimes (required, since known\_way names two): the cat's treasure count on the main diagonal *is* the regime detector. Near-identical ropes → `w` collapses to ~0 → the corridor is one knot wide. Random ropes → `w ≈ 0.3n` → ~60 % of the grid, still exact. `w` is clamped to `n`, so full-grid is not a separate code path, just `w = n`. Size guard: 16-bit lanes are provably safe only for `n ≤ 10000` (real `y ∈ [−2n, 3n]`, sentinel −32000); outside that the kernel falls back to a scalar two-row version of the *same* y-space recurrence. No OpenMP: rows are strictly serial in this formulation and the per-row work is a few microseconds — threading it would be overhead, so it is dropped rather than shipped unguarded.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>

#if defined(__SSSE3__) || defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#define ROPE_SIMD 1
#endif

/* ------------------------------------------------------------------ *
 *  y-space:  y_i[j] = D[i][j] + 2j
 *      y_i[j] = max( y_{i-1}[j-1] + (2+s), y_{i-1}[j] - 2, y_i[j-1] )
 *  The corridor (left) term carries NO penalty: the gap price was
 *  pre-paid into the coordinate.  A row is therefore
 *      map  ->  prefix-maximum.
 *  Answer: y_n[n] - 2n.
 * ------------------------------------------------------------------ */

static int rope_scalar(int n, const char *restrict a, const char *restrict b)
{
    int *p = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *c = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int i, j, res;
    if (!p || !c) { free(p); free(c); return 0; }
    for (j = 0; j <= n; j++) p[j] = 0;              /* y_0[j] = 0 */
    for (i = 1; i <= n; i++) {
        char ai = a[i - 1];
        int run = -2 * i;                            /* y_i[0] */
        c[0] = run;
        for (j = 1; j <= n; j++) {
            int d = p[j - 1] + ((ai == b[j - 1]) ? 3 : 1);
            int u = p[j] - 2;
            int wv = d > u ? d : u;
            if (wv > run) run = wv;                  /* drop the worse tally */
            c[j] = run;
        }
        { int *t = p; p = c; c = t; }
    }
    res = p[n] - 2 * n;
    free(p); free(c);
    return res;
}

#ifdef ROPE_SIMD
static int rope_simd(int n, const char *restrict a, const char *restrict b, int w)
{
    const short NEG = -32000;
    size_t alloc = (size_t)n + 32;
    short *ya  = (short *)malloc(alloc * sizeof(short));
    short *yb  = (short *)malloc(alloc * sizeof(short));
    short *b16 = (short *)malloc(alloc * sizeof(short));
    short *prev, *cur;
    int i, j, res;

    if (!ya || !yb || !b16) { free(ya); free(yb); free(b16); return rope_scalar(n, a, b); }

    for (j = 0; j < n; j++) b16[j] = (short)(unsigned char)b[j];
    for (j = n; j < (int)alloc; j++) b16[j] = 0;          /* never matches ACGT */

    prev = ya; cur = yb;
    for (j = 0; j < (int)alloc; j++) { prev[j] = NEG; cur[j] = NEG; }
    { int jh0 = (w < n) ? w : n;
      for (j = 0; j <= jh0; j++) prev[j] = 0; }           /* row 0 inside band */

    {
    const __m128i NEGV = _mm_set1_epi16(NEG);
    const __m128i TWO  = _mm_set1_epi16(2);
    const __m128i ONE  = _mm_set1_epi16(1);
    const __m128i BCM  = _mm_setr_epi8(14,15,14,15,14,15,14,15,
                                       14,15,14,15,14,15,14,15);

    for (i = 1; i <= n; i++) {
        int jlo = i - w; if (jlo < 0) jlo = 0;
        int jhi = i + w; if (jhi > n) jhi = n;
        int jst = (jlo < 1) ? 1 : jlo;
        __m128i carry, av;

        if (jlo == 0) { cur[0] = (short)(-2 * i); carry = _mm_set1_epi16((short)(-2 * i)); }
        else          { cur[jst - 1] = NEG;       carry = NEGV; }
        av = _mm_set1_epi16((short)(unsigned char)a[i - 1]);   /* the low sun */

        j = jst;
        for (; j + 15 <= jhi; j += 16) {
            __m128i p1a = _mm_loadu_si128((const __m128i *)(prev + j - 1));
            __m128i p0a = _mm_loadu_si128((const __m128i *)(prev + j));
            __m128i bva = _mm_loadu_si128((const __m128i *)(b16  + j - 1));
            __m128i p1b = _mm_loadu_si128((const __m128i *)(prev + j + 7));
            __m128i p0b = _mm_loadu_si128((const __m128i *)(prev + j + 8));
            __m128i bvb = _mm_loadu_si128((const __m128i *)(b16  + j + 7));
            __m128i ta, tb, c1, o0, o1;

            /* doubled shadow: equal dyes -> +3, else +1, 8 crossings at once */
            ta = _mm_max_epi16(
                   _mm_add_epi16(p1a, _mm_add_epi16(ONE,
                       _mm_and_si128(_mm_cmpeq_epi16(bva, av), TWO))),
                   _mm_sub_epi16(p0a, TWO));
            tb = _mm_max_epi16(
                   _mm_add_epi16(p1b, _mm_add_epi16(ONE,
                       _mm_and_si128(_mm_cmpeq_epi16(bvb, av), TWO))),
                   _mm_sub_epi16(p0b, TWO));

            /* corridor read, not paced: log-depth prefix max, NEG shifted in */
            ta = _mm_max_epi16(ta, _mm_alignr_epi8(ta, NEGV, 14));
            ta = _mm_max_epi16(ta, _mm_alignr_epi8(ta, NEGV, 12));
            ta = _mm_max_epi16(ta, _mm_alignr_epi8(ta, NEGV,  8));
            tb = _mm_max_epi16(tb, _mm_alignr_epi8(tb, NEGV, 14));
            tb = _mm_max_epi16(tb, _mm_alignr_epi8(tb, NEGV, 12));
            tb = _mm_max_epi16(tb, _mm_alignr_epi8(tb, NEGV,  8));

            o0 = _mm_max_epi16(ta, carry);
            c1 = _mm_max_epi16(carry, _mm_shuffle_epi8(ta, BCM));
            o1 = _mm_max_epi16(tb, c1);
            _mm_storeu_si128((__m128i *)(cur + j),     o0);
            _mm_storeu_si128((__m128i *)(cur + j + 8), o1);
            carry = _mm_max_epi16(c1, _mm_shuffle_epi8(tb, BCM));
        }
        for (; j <= jhi; j += 8) {
            __m128i p1 = _mm_loadu_si128((const __m128i *)(prev + j - 1));
            __m128i p0 = _mm_loadu_si128((const __m128i *)(prev + j));
            __m128i bv = _mm_loadu_si128((const __m128i *)(b16  + j - 1));
            __m128i t  = _mm_max_epi16(
                           _mm_add_epi16(p1, _mm_add_epi16(ONE,
                               _mm_and_si128(_mm_cmpeq_epi16(bv, av), TWO))),
                           _mm_sub_epi16(p0, TWO));
            t = _mm_max_epi16(t, _mm_alignr_epi8(t, NEGV, 14));
            t = _mm_max_epi16(t, _mm_alignr_epi8(t, NEGV, 12));
            t = _mm_max_epi16(t, _mm_alignr_epi8(t, NEGV,  8));
            t = _mm_max_epi16(t, carry);
            _mm_storeu_si128((__m128i *)(cur + j), t);
            carry = _mm_shuffle_epi8(t, BCM);
        }
        { int ge = jhi + 10; if (ge > (int)alloc - 1) ge = (int)alloc - 1;
          for (j = jhi + 1; j <= ge; j++) cur[j] = NEG; }   /* re-blank the walls */

        { short *t = prev; prev = cur; cur = t; }
    }
    res = (int)prev[n] - 2 * n;
    }

    free(ya); free(yb); free(b16);
    return res;
}
#endif

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
#ifdef ROPE_SIMD
    if (n >= 32 && n <= 10000) {          /* int16 lanes provably safe here */
        int m0 = 0, i = 0;
        for (; i + 16 <= n; i += 16) {    /* the cat counts main-diagonal treasures */
            __m128i va = _mm_loadu_si128((const __m128i *)(a + i));
            __m128i vb = _mm_loadu_si128((const __m128i *)(b + i));
            m0 += __builtin_popcount(
                    (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(va, vb)));
        }
        for (; i < n; i++) m0 += (a[i] == b[i]);
        {
            long lb = 2L * m0 - n;               /* score of the no-slip alignment */
            long wl = ((long)n - lb) / 5;        /* every alignment: D <= n - 5g   */
            int  w  = (wl >= n) ? n : (int)wl;   /* clamped -> full grid if needed */
            return rope_simd(n, a, b, w);
        }
    }
#endif
    return rope_scalar(n, a, b);
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 8**

Reasoning, stated before any measurement: the reference materialises an (n+1)² int table (67 MB at n=4096) and is latency-chained at ~2 cycles/cell on the `cur[j-1]` dependency. This kernel touches O(n) memory, judges 8 crossings per instruction, and has a carry chain of ~1 `max` per 16 cells. Throughput bound ≈ 2.4 uops/cell → ~0.6 cycles/cell, times ~0.6 of the grid from the slip-budget band ⇒ roughly 6–12×; I commit to 8.

## MEASUREMENT

**Not performed.** No tools were available in this session — `alignment_bench` and `alignment_contract` could not be invoked, so I have no measured number and I will not manufacture one. The prediction above stands unverified; the pipeline's measurement is the verdict. What I *can* assert without the bench is exactness, and I verified the transform by hand: `n=1 A/A → +1`, `n=1 A/C → −1`, `n=2 AA/TA → D[1][1]=−1, D[2][1]=−3, D[2][2]=0`, all matching the reference recurrence; and the band bound `g ≤ (n−L)/5` follows from `D = 2M − n − 3g ≤ n − 5g` with `L = 2M₀ − n` an achievable score, so an optimal path always lies inside.

## VERDICT

The core here is the native's second seed taken literally: *the corridor's price was never a secret, only its length was.* Paying the gap penalty into the coordinate (`y = D + 2j`) turns the left-neighbour term into a bare `max` with no arithmetic in it, which converts the one genuinely serial part of Needleman–Wunsch into a prefix maximum — a log-depth SIMD scan. That is the whole mechanism, and it is not banding and not an anti-diagonal wavefront; the band is a bounded, provable *secondary* guard that the metaphor's third seed supplies, clamped to the full grid so it can never be a separate risky path.

Where this could be worse than the known way, and what I did about it:
- **Small n** — SIMD setup and three mallocs dominate. Guarded: `n < 32` takes the scalar path.
- **Large n** — 16-bit lanes overflow above `3n ≈ 32767`. Guarded: `n > 10000` takes the scalar two-row path, which is still table-free and so still beats the reference, just without the vector win.
- **Non-x86 / no SSSE3** — guarded by `#ifdef` with the same scalar fallback.
- **Threading** — dropped outright rather than shipped: the recurrence is serial in *i* and a row is microseconds of work, so OpenMP would be pure overhead. I would rather report that honestly than add a knob that looks parallel.

The one thing I would flag as unresolved: if the benchmark's sequences are unrelated random DNA, the slip budget `w` lands near `0.3n` and the band buys only ~40 %. The speedup then rests almost entirely on the scan reformulation and on not allocating an n² table — which is the honest accounting of where the win comes from.