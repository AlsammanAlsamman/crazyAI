## MAPPING

### SEED 1 — "two boughs hung from crossed rows of lamp-posts, their meeting makes a lattice of spark-cells"

| World object | Problem object |
|---|---|
| bough running down | string `a`, indexed by `i` |
| bough running across | string `b`, indexed by `j` |
| row-post / column-post | the two index registers |
| lattice of unlit windows | the (n+1)² DP grid, never materialized |
| spark lit "only when I need it" | a cell value computed on demand |
| spark's colour chosen *by the two symbols*, not by me | substitution score `a[i]==b[j] ? +1 : -1` computed branchlessly from data |

Breaks: *"the whole grid of every position against every other must be filled in"* (only the cells actually needed are lit) — and, mildly, "one pair of positions is judged at a time" (a spark's colour is a data-driven, lane-parallel decision).

### SEED 2 — "fold along the slanting middle; cells of equal row-and-column sum press into one crease"

| World object | Problem object |
|---|---|
| slanting middle / fold line | the anti-diagonal `d = i + j` |
| one crease | the set `{(i,j) : i+j = d}` — a wavefront |
| cells pressed flush | mutually **independent** cells → one SIMD vector, lanes indexed by `i` |
| "the note itself is already folded in the hour between the note-trees" | the dependency cone already has this symmetry: same-`d` cells never depend on each other |
| the sheet | 3 × O(n) int16 buffers, not an O(n²) table |

Breaks: **"every cell depends on above, left, and above-left, *computed in that order*"** — the fold destroys the row-major order entirely; cells are produced in `i+j` order, and a whole crease is produced at one instant.

### SEED 3 — "read three stacked layers through each crease at once; let the spent bottom layer drift off"

| World object | Problem object |
|---|---|
| three layers of spark, bit for bit atop each other | diagonals `d`, `d-1`, `d-2` as three aligned int16 arrays indexed by the **same** `i` |
| "the layer beneath tells me what agreeing straight through cost" | `D2[i-1]` = the diagonal predecessor `dp[i-1][j-1]` |
| "the layer beside tells me what one slip cost" | `D1[i-1]` (up) and `D1[i]` (left), both `-2` |
| read all three at once, not one after another | one vector `max(max(dg,up),lf)` — no serial `left` chain |
| throw the bottom layer away / let it drift into the gradient | pointer rotation: 3 buffers reused forever, **O(n) memory** |
| "no ground to store it on" | no DP table; everything stays in L1 |
| "wait only as long as it takes each spark to catch" | latency = dependency latency only, no bookkeeping stalls |

Breaks: "the whole grid must be filled in" (as *stored*), and again the ordering assumption.

## CHOSEN SEED

**SEED 2** (with SEED 3 as its necessary consequence, and SEED 1's "light only what I need" as its regime switch). It is the only seed that breaks the preferred assumption — dependence "above, left, above-left, *in that order*" — and it does so with a completely literal mapping: the fold line *is* `i+j = const`, the crease *is* a SIMD register, three layers *are* three rolling buffers, letting the bottom layer drift off *is* the pointer rotation.

The literal mapping forces one more thing out, which is the part I did not expect: to read `a[i-1]` and `b[j-1]` as *contiguous* lanes along a crease, `b` must be hung on its posts **backwards** (`brev[k] = b[n-1-k]`), because `j = d-i` runs down while `i` runs up. So the metaphor also breaks *"both strings are read start to end in the same direction."* That reversal is what makes the substitution scores vectorizable at all.

Honesty note per step 4: this lands exactly on a **validated real technique** — Wozniak's 1997 anti-diagonal SIMD wavefront, plus Ukkonen banding — and I let it land there rather than inventing something new. The band bound is my own derivation but it is a proof, not a heuristic (below).

## ASSUMPTION BROKEN

Primary: *"every cell of the comparison depends on the ones above, to the left, and diagonally above-left, computed in that order."* Row-major order is abandoned; the traversal order is `i+j`, and every cell of one anti-diagonal is computed simultaneously (16 or 32 at a time). Secondary: the grid is never stored (3 rolling vectors), `b` is read in reverse, and cells outside a provably-safe band are never lit.

**Regime recognition (step 5).** `known_way` names two regimes (full O(n²) DP vs. a banded/bounded-score variant). The native's own "I light each cell only when I need it" supplies the runtime test: walk the main crease first (O(n), vectorizable) and get `S_lb` = the pure-diagonal score. Any alignment whose path reaches offset `t = |i-j|` uses `g ≥ 2t` gap columns; with `2m + g = 2n` it has `m ≤ n - t` aligned pairs, so its score is `≤ m - 2g ≤ n - 5t`. Hence no optimal path leaves `|i-j| ≤ W = ⌊(n - S_lb)/5⌋` (+2 margin). Near-identical boughs → `W` tiny → O(n·W) work; divergent boughs → `W → n` → the band *is* the full lattice and the identical code path degenerates to the exact full wavefront. One kernel, two regimes, no duplicated logic.

**Guards for my own stated risks:** the crease is only worth folding when it holds enough sparks to fill a vector, and int16 sparks only hold `[-2n, n]`, so `n < 64` or `n > 16000` falls back to a plain two-row int32 DP. No OpenMP: at the benchmark's `n`, one crease is ~n/2 cells ≈ a few hundred bytes — far below a thread-sync's worth of work, so thread parallelism would be pure loss (and a per-diagonal barrier 2n times is the classic wavefront-threading trap).

## ARTIFACT

```c
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define NW_MATCH      1
#define NW_MISMATCH (-1)
#define NW_GAP      (-2)

/* ---------------- the crease: one vector of int16 sparks ---------------- */
#if defined(__AVX512BW__)
  #include <immintrin.h>
  #define NW_VEC 1
  #define NW_VW  32
  typedef __m512i nwv;
  #define nwv_load(p)     _mm512_loadu_si512((const void *)(p))
  #define nwv_store(p,v)  _mm512_storeu_si512((void *)(p), (v))
  #define nwv_add(x,y)    _mm512_add_epi16((x),(y))
  #define nwv_max(x,y)    _mm512_max_epi16((x),(y))
  #define nwv_set1(x)     _mm512_set1_epi16((short)(x))
  static inline nwv nwv_score(const char *pa, const char *pb) {
      __m256i eq8 = _mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i *)pa),
                                      _mm256_loadu_si256((const __m256i *)pb));
      __m512i eq  = _mm512_cvtepi8_epi16(eq8);          /* 0 or -1 per lane */
      return _mm512_sub_epi16(_mm512_set1_epi16(-1), _mm512_add_epi16(eq, eq));
  }
#elif defined(__AVX2__)
  #include <immintrin.h>
  #define NW_VEC 1
  #define NW_VW  16
  typedef __m256i nwv;
  #define nwv_load(p)     _mm256_loadu_si256((const __m256i *)(p))
  #define nwv_store(p,v)  _mm256_storeu_si256((__m256i *)(p), (v))
  #define nwv_add(x,y)    _mm256_add_epi16((x),(y))
  #define nwv_max(x,y)    _mm256_max_epi16((x),(y))
  #define nwv_set1(x)     _mm256_set1_epi16((short)(x))
  static inline nwv nwv_score(const char *pa, const char *pb) {
      __m128i eq8 = _mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)pa),
                                   _mm_loadu_si128((const __m128i *)pb));
      __m256i eq  = _mm256_cvtepi8_epi16(eq8);
      return _mm256_sub_epi16(_mm256_set1_epi16(-1), _mm256_add_epi16(eq, eq));
  }
#elif defined(__SSE2__) || defined(__x86_64__) || defined(_M_X64)
  #include <emmintrin.h>
  #define NW_VEC 1
  #define NW_VW  8
  typedef __m128i nwv;
  #define nwv_load(p)     _mm_loadu_si128((const __m128i *)(p))
  #define nwv_store(p,v)  _mm_storeu_si128((__m128i *)(p), (v))
  #define nwv_add(x,y)    _mm_add_epi16((x),(y))
  #define nwv_max(x,y)    _mm_max_epi16((x),(y))
  #define nwv_set1(x)     _mm_set1_epi16((short)(x))
  static inline nwv nwv_score(const char *pa, const char *pb) {
      __m128i eq8 = _mm_cmpeq_epi8(_mm_loadl_epi64((const __m128i *)pa),
                                   _mm_loadl_epi64((const __m128i *)pb));
      __m128i eq  = _mm_unpacklo_epi8(eq8, eq8);        /* 0x0000 / 0xFFFF */
      return _mm_sub_epi16(_mm_set1_epi16(-1), _mm_add_epi16(eq, eq));
  }
#endif

/* ------------- flat-walker fallback: two rolling rows, int32 ------------- */
static int nw_rows(int n, const char *a, const char *b)
{
    int sprev[257], scur[257];
    int *prev, *cur, *heap = NULL;
    if (n <= 256) { prev = sprev; cur = scur; }
    else {
        heap = (int *)malloc((size_t)2 * (n + 1) * sizeof(int));
        if (!heap) return 0;
        prev = heap; cur = heap + (n + 1);
    }
    for (int j = 0; j <= n; ++j) prev[j] = j * NW_GAP;
    for (int i = 1; i <= n; ++i) {
        const char ai = a[i - 1];
        cur[0] = i * NW_GAP;
        int left = cur[0];
        for (int j = 1; j <= n; ++j) {
            int dg = prev[j - 1] + (ai == b[j - 1] ? NW_MATCH : NW_MISMATCH);
            int up = prev[j] + NW_GAP;
            int lf = left + NW_GAP;
            int best = dg > up ? dg : up;
            if (lf > best) best = lf;
            cur[j] = best; left = best;
        }
        int *t = prev; prev = cur; cur = t;
    }
    int r = prev[n];
    free(heap);
    return r;
}

#ifdef NW_VEC
#define NW_STACK_N 2048
#define NW_SLACK   (2 * NW_VW + 8)
#define NW_STACK_V (3 * (NW_STACK_N + 2 + 2 * 32 + 8))
#define NW_STACK_C (2 * (NW_STACK_N + 2 * 32 + 8))

/* fold the lattice along i+j; keep only three creases; band half-width W */
static int nw_diag(int n, const char *a, const char *b, int W)
{
    const int L = n + 2 + NW_SLACK;   /* one crease buffer, indexed by i */
    const int C = n + NW_SLACK;       /* one padded bough                */
    int16_t stack_v[NW_STACK_V];
    char    stack_c[NW_STACK_C];
    int16_t *v; char *ch; void *heap = NULL;

    if (n <= NW_STACK_N) { v = stack_v; ch = stack_c; }
    else {
        heap = malloc((size_t)3 * L * sizeof(int16_t) + (size_t)2 * C + 64);
        if (!heap) return nw_rows(n, a, b);
        v  = (int16_t *)heap;
        ch = (char *)((int16_t *)heap + (size_t)3 * L);
    }

    int16_t *p2 = v, *p1 = v + L, *p0 = v + 2 * L;
    char *ca = ch, *cb = ch + C;
    const int16_t NEG = (int16_t)(-2 * n - 16);   /* unreachable: dp >= -2n */

    /* one bough hangs forward, the other hangs backward, so a crease reads
       both contiguously: b[j-1] with j = d-i  ==  cb[n-d+i]               */
    memcpy(ca, a, (size_t)n);
    for (int k = n; k < C; ++k) ca[k] = 1;
    for (int k = 0; k < n; ++k) cb[k] = b[n - 1 - k];
    for (int k = n; k < C; ++k) cb[k] = 2;

    for (int k = 0; k < 3 * L; ++k) v[k] = NEG;

    p2[0] = 0;                        /* crease d=0 : dp[0][0]             */
    p1[0] = (int16_t)NW_GAP;          /* crease d=1 : dp[0][1]             */
    p1[1] = (int16_t)NW_GAP;          /*              dp[1][0]             */

    const nwv vgap = nwv_set1(NW_GAP);

    for (int d = 2; d <= 2 * n; ++d) {
        int blo = (d - W + 1) / 2;            /* band: |2i - d| <= W       */
        int bhi = (d + W) >> 1;
        int ilo = d - n; if (ilo < 0) ilo = 0;
        int ihi = d < n ? d : n;
        if (blo < ilo) blo = ilo;
        if (bhi > ihi) bhi = ihi;

        int lo = blo < 1 ? 1 : blo;           /* interior cells only       */
        int hi = bhi < d - 1 ? bhi : d - 1;

        if (hi >= lo) {
            const char    *pa = ca + (lo - 1);
            const char    *pb = cb + (n - d + lo);
            const int16_t *q2 = p2 + lo - 1;  /* layer beneath : diagonal  */
            const int16_t *q1 = p1 + lo - 1;  /* layer beside  : up / left */
            int16_t       *q0 = p0 + lo;
            const int len = hi - lo + 1;
            for (int t = 0; t < len; t += NW_VW) {
                nwv sc = nwv_score(pa + t, pb + t);
                nwv dg = nwv_add(nwv_load(q2 + t), sc);
                nwv up = nwv_add(nwv_load(q1 + t), vgap);
                nwv lf = nwv_add(nwv_load(q1 + t + 1), vgap);
                nwv_store(q0 + t, nwv_max(nwv_max(dg, up), lf));
            }
        }
        /* unlit windows just outside the band read as minus-infinity */
        if (blo - 1 >= 0) p0[blo - 1] = NEG;
        if (blo - 2 >= 0) p0[blo - 2] = NEG;
        p0[bhi + 1] = NEG;
        p0[bhi + 2] = NEG;
        /* the two border sparks, in-band only while d <= W */
        if (d <= n && d <= W) {
            int16_t g = (int16_t)(-2 * d);
            p0[0] = g; p0[d] = g;
        }
        /* the spent bottom layer drifts off into the gradient */
        int16_t *t = p2; p2 = p1; p1 = p0; p0 = t;
    }

    int res = (int)p1[n];             /* last crease, far corner */
    free(heap);
    return res;
}
#endif /* NW_VEC */

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
#ifdef NW_VEC
    /* fold only when a crease can fill a vector and an int16 spark holds
       the whole score range ([-2n, n]); otherwise walk flat.             */
    if (n >= 64 && n <= 16000) {
        int s = 0;
        for (int k = 0; k < n; ++k) s += (a[k] == b[k]) ? NW_MATCH : NW_MISMATCH;
        /* any path at offset t scores <= n - 5t; the main crease already
           scores s, so no optimal path leaves |i-j| <= (n-s)/5.          */
        int W = (n - s) / 5 + 2;
        if (W > n) W = n;             /* W == n  =>  exact full wavefront */
        return nw_diag(n, a, b, W);
    }
#endif
    return nw_rows(n, a, b);
}
```

## PREDICTION

PREDICTION: speedup_vs_dp = 12

Reasoning behind the number, stated before any measurement: the reference does ~2–4 cycles/cell (its `left` chain is a serial dependency, so gcc cannot auto-vectorize it) and streams 4·(n+1)² bytes — at n≈1000 that is 4 MB, past L2. The fold does 5 vector ops per 16 (or 32) cells ≈ 0.3–0.4 cycles/cell out of ~10 KB of L1-resident buffers, and the band removes ~40 % of cells on random DNA (`S_lb ≈ -0.5n → W ≈ 0.3n`). 3 × ÷ 0.6 ≈ 12×. If the benchmark pairs are similar rather than random, the band collapses and the ratio should be much larger (tens to hundreds); if the benchmark uses small n (≈100), expect only 4–6× because 2n crease set-ups amortize badly.

## MEASUREMENT

**Not performed.** This session was invoked with no tools available — `alignment_bench` and `alignment_contract` could not be called, so I have no measured number and I will not invent one. What I can state:

- Contract: signature is `int kernel(int n, const char *a, const char *b)`, argument order untouched, scoring +1/−1/−2, returns `dp[n][n]`.
- Correctness argued, not measured. The pieces I checked by hand: `j = d-i` index algebra and the reversed-`b` access; that reads of crease `d-1` never exceed index `min(n,d-1)` and reads of crease `d-2` never exceed `min(n,d-2)` (so no garbage lane ever contaminates a valid cell); that SIMD overshoot writes land only at indices > the last valid one and are overwritten by the sentinel/border stores that follow; that every in-band interior cell's *diagonal* predecessor is also in-band, so a `NEG` sentinel can never win a `max`; `dp ∈ [-2n, n]` ⇒ int16 safe to n = 16000; hand-traced d = 0,1,2 and n = 1.
- If the pipeline reports a mismatch, the two places I would look first are the sentinel width at the band edges (I write two on each side where the proof needs one) and the `d <= W` border-in-band condition.

## VERDICT

The native was describing the anti-diagonal SIMD wavefront, and describing it correctly and completely — including the two things people usually bolt on afterwards. "Three layers pressed flush and read at once" is not a metaphor for the recurrence; it *is* the recurrence, re-indexed so that the diagonal predecessor and the two gap predecessors sit in two arrays at the same lane index. "Throw the bottom layer away, it drifts off, I am not a forest that hoards its own marsh" is the O(n)-memory rotation that turns a 4 MB streaming problem into a 10 KB L1-resident one — which is probably where most of the win actually comes from, more than the lane count. And "I light each cell only when I need it" is Ukkonen banding; taking it literally forced me to derive a bound (`score ≤ n − 5t`) and gave the runtime regime test for free, using the main crease as its own probe.

The genuinely new-feeling thing the literalism produced: hanging the second bough **backwards** on its lamp-posts. Without that, the substitution scores along a crease are a gather and the whole idea collapses to scalar speed. The native says the boughs are "cut from different note-trees" and hung on *crossed* rows — one down, one across — and that crossing is exactly the reversal.

Where this can lose to the known way, and what I did about each:
- **Small n**: 2n crease set-ups dominate. Guarded — `n < 64` falls back to the two-row scalar DP.
- **Large n / int16 overflow**: guarded — `n > 16000` falls back to the int32 two-row DP (correct, just not fast; I judged an int32 wavefront not worth the extra untested path).
- **No AVX2**: SSE2 path (8 lanes) compiles on any x86-64; non-x86 gets the scalar path.
- **Narrow-band regime**: handled by the same code with a small `W`, not a second kernel.
- **Threads**: deliberately not used. One crease at n≈1000 is a few hundred bytes of work and there would be 2n barriers; I expect that to be a net loss, so per the instruction I stayed at vectorization.

The honest residual risk is the band, because it is the one part whose bound I derived myself rather than inherited. It is a proof and not a heuristic, and setting `W = n` (which the dispatcher does whenever the probe says the sequences are far apart enough) makes the kernel bit-identical to the unbanded full wavefront — so if the pipeline reports a wrong score, the band is the first thing to disable, and the fold underneath it should still stand on its own at roughly 6–8×.