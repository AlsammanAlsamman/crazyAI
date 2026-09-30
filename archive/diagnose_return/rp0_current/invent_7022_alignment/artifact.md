## MAPPING

### SEED 1 — "Two knot-cords laid crosswise over a sand-tray mark every place their symbols meet."

| World object | Problem object |
|---|---|
| cord running **north**, the horse's length | `a[0..n-1]`, laid out along the row index `i` |
| cord running **east**, the mountain's width | `b[0..n-1]`, laid out along the column index `j` — *perpendicular*, i.e. traversed in the opposite sense from `a` when you walk a slant |
| the crossing that "marks" the lattice | a single vector compare `a[i-1..] == brev[n-d+i ..]` — the two cords are physically laid so that one slice of each is contiguous and the meeting of a whole run of symbols is read **in one act** |
| furrow-point | DP cell `(i,j)` |
| "every place their symbols meet" (plural, simultaneous) | all cells on one anti-diagonal `d=i+j` judged in the same instruction |

**Breaks:** *one pair of positions is judged at a time.* The cords do not visit pairs; their crossing declares all agreements at once.

### SEED 2 — "A single ink-worm walks the eight desert winds and curls back across its own trail."

| World object | Problem object |
|---|---|
| **one** ink-worm, not a company of hooded memories | one SIMD register — one wide body, **not** many threads |
| "one cell wide" | the worm's body is exactly one anti-diagonal thick: 3 rolling diagonal buffers, nothing else |
| eight winds; straight along the grain / slant across the furrow | the three live moves N, W, NW (`up`, `left`, `diag`); the other winds are the doubling-back |
| "curls back across ground already crossed" | buffer rotation — the physical array that held `d-2` is overwritten to become `d`; the worm literally re-crawls its own sand |
| "keeps no ledger apart from the sand itself; a beaten mound is thrown away entirely" | no `(n+1)²` table, no traceback: 3·O(n) shorts total |

**Breaks:** *the whole grid of every position against every other must be filled in*, and *dependencies must be taken above→left→diagonal in that order*.

### SEED 3 — "The worm crosses a furrow sideways, spending no sand, to let one cord slip ahead."

| World object | Problem object |
|---|---|
| sideways crossing that **spends no sand** | the gap transition reads **no symbol**: its score change is a letter-blind constant `-2`, needing no character load, no compare |
| "the one permitted stumble" | one gap step; and — crucially — *how many* stumbles the worm can afford is fixed in advance by the height it already earned walking the straight grain |

**Breaks:** *a slip (gap) can only be discovered by having already compared the position before it.* The number of affordable slips is known **before** any cell is computed, from the straight-grain pass alone.

---

## CHOSEN SEED

**SEED 1** — the crosswise cords.

It is the only one of the three that breaks the preferred assumption ("one pair of positions is judged at a time"), and its mapping is the most literal: the *crosswise* lay-down is not decoration, it is the data layout. Laying `b` along the perpendicular edge means that when you read a slant, you read `a` forward and `b` **backward** — so materializing `brev[t] = b[n-1-t]` once makes both operands of a whole diagonal's worth of comparisons contiguous. That single physical fact is what makes "every place their symbols meet" a *one-instruction* event instead of a gather.

SEEDs 2 and 3 are not discarded; they constrain the same artifact (SEED 2 forbids threads and forbids the full table; SEED 3 supplies the band).

**Full computational mapping.** Memory = the sand (three anti-diagonal buffers, `p2`,`p1`,`cur`; nothing else persists — a beaten mound is thrown away). What flows = the worm, one anti-diagonal wide, sweeping `d = 0 … 2n`. What stays still = the two cords (`a`, and `brev`, written once). The processor = the worm's body, one SIMD register, **singular** — the native explicitly refuses "a hooded memory at each crossing," which is exactly a refusal of thread-level parallelism, so this kernel uses none. Time = the diagonal index `d`; every cell on one `d` is simultaneous, which is legal because `(i-1,j-1)`, `(i-1,j)`, `(i,j-1)` all lie on `d-1` or `d-2`.

**Arriving at validated technique, not invention.** Taken literally, the crosswise lay-down *is* Wozniak's (1997) anti-diagonal SIMD alignment, and SEED 3's "affordable stumbles" *is* Ukkonen-style adaptive banding (as in `edlib`). I let the metaphor land on those rather than inventing a substitute.

**Regime detection, in-world.** The native's worm first runs the straight grain and earns a standing height; that height fences the tray. Concretely: one O(n) vectorized pass gives the Hamming score `L = n − 2H`, a valid alignment score, hence `S* ≥ L`. Any path reaching offset `m` has `Ia = Ib = k ≥ m` and score `n − 2X − 5k ≤ n − 5m`; so it can be optimal only if `n − 5m ≥ n − 2H`, i.e. `m ≤ ⌊2H/5⌋`. Band half-width `W = ⌊2H/5⌋` is therefore **exact, not heuristic**. Near-identical cords ⇒ `W ≈ 0` (a thread of sand); random DNA ⇒ `H ≈ 0.75n`, `W ≈ 0.3n` (still ~2× fewer cells than the full tray). Both regimes, one mechanism, chosen at runtime.

**Guarded risks.** (i) Wavefront bookkeeping dominates on a tiny tray → `n < 96` falls back to plain rolling-row DP. (ii) 16-bit mounds would overflow on a huge tray → `n > 15000` falls back to a 32-bit wavefront (the bound `D ≥ −n − 2W ≥ −1.8n` with `NEG = −32000` makes the 16-bit path provably safe below the threshold). (iii) OOM → rolling-row DP.

## ASSUMPTION BROKEN

**"One pair of positions is judged at a time."** Replaced by: one crossing of two crosswise cords judges an entire anti-diagonal of pairs in a single compare. Secondarily broken: the above→left→diagonal ordering (all three predecessors are already finished two diagonals back), the full grid (3 diagonals of sand, everything else discarded), and gap-discovery-by-prior-comparison (the affordable slip count is fixed before the first cell).

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH (-1)
#define GAP      (-2)

/* ------------------------------------------------------------------------ *
 * "a hooded memory kneeling at each crossing in turn" -- the plain path.
 * Used for tiny trays (where worm bookkeeping dominates) and on allocation
 * failure.  Rolling row, exact Needleman-Wunsch.
 * ------------------------------------------------------------------------ */
static int nw_rowdp(int n, const char *a, const char *b)
{
    int stackrow[1025];
    int *row;
    int i, j, r;
    if (n <= 0) return 0;
    row = (n + 1 <= 1025) ? stackrow
                          : (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!row) return 0;
    for (j = 0; j <= n; j++) row[j] = j * GAP;
    for (i = 1; i <= n; i++) {
        int diagp = row[0];
        char ai = a[i - 1];
        row[0] = i * GAP;
        for (j = 1; j <= n; j++) {
            int up = row[j];
            int dg = diagp + (ai == b[j - 1] ? MATCH : MISMATCH);
            int uu = up + GAP;
            int ll = row[j - 1] + GAP;
            int best = dg > uu ? dg : uu;
            if (ll > best) best = ll;
            diagp = up;
            row[j] = best;
        }
    }
    r = row[n];
    if (row != stackrow) free(row);
    return r;
}

/* ------------------------------------------------------------------------ *
 * The ink-worm, 16-bit mounds.  One body, one cell wide, sweeping the
 * anti-diagonals d = i + j.  Cords laid crosswise: b is written out
 * backwards ONCE, so that along a slant both cords read forward and a whole
 * run of "places where symbols meet" is one compare.
 *
 *   cur[i] = D[i][d-i] = max( p2[i-1] + s , p1[i-1] + GAP , p1[i] + GAP )
 *   s      = (a[i-1] == brev[n-d+i]) ? +1 : -1
 *
 * Only cells with |i-j| <= W are visited (W from the straight-grain pass).
 * Two guard mounds per diagonal (at lo-1 and hi+1) hold NEG so the band edge
 * needs no branch.  Safe because every visited cell's diagonal predecessor is
 * itself in-band, so no computed value is ever guard-derived.
 * ------------------------------------------------------------------------ */
static int nw_wave16(int n, const char *a, const char *b, int W)
{
    const short NEGV = -32000;           /* below any real value (>= -1.8n) */
    const int   sz   = n + 48;
    short *buf = (short *)malloc((size_t)3 * (size_t)sz * sizeof(short));
    char  *brb = (char  *)malloc((size_t)n + 64);
    short *p2, *p1, *cur;
    int d, result = 0;
    size_t t;
    const int twon = 2 * n;

    if (!buf || !brb) { free(buf); free(brb); return nw_rowdp(n, a, b); }
    for (t = 0; t < (size_t)n; t++) brb[t] = b[n - 1 - (int)t];
    memset(brb + n, 0, 64);
    for (t = 0; t < (size_t)3 * (size_t)sz; t++) buf[t] = NEGV;

    p2  = buf + 16;
    p1  = buf + sz + 16;
    cur = buf + 2 * sz + 16;

    for (d = 0; d <= twon; d++) {
        int lo = d - W, hi, ilo, ihi, i, base;
        lo = (lo <= 0) ? 0 : ((lo + 1) >> 1);      /* ceil((d-W)/2), >= 0   */
        if (lo < d - n) lo = d - n;
        hi = (d + W) >> 1;                         /* floor((d+W)/2)        */
        if (hi > n) hi = n;
        if (hi > d) hi = d;

        base = n - d;
        ilo  = (lo < 1) ? 1 : lo;
        ihi  = (hi > d - 1) ? (d - 1) : hi;

        i = ilo;
#if defined(__AVX512BW__) && defined(__AVX512F__)
        {
            const __m512i vg  = _mm512_set1_epi16((short)GAP);
            const __m512i vm1 = _mm512_set1_epi16((short)-1);
            for (; i + 31 <= ihi; i += 32) {
                __m256i ca = _mm256_loadu_si256((const __m256i *)(a + i - 1));
                __m256i cb = _mm256_loadu_si256((const __m256i *)(brb + base + i));
                __m256i eq = _mm256_cmpeq_epi8(ca, cb);
                __m512i ew = _mm512_cvtepi8_epi16(eq);        /* -1 if equal */
                /* s = -1 - 2*ew :  equal -> +1 , differ -> -1 */
                __m512i s  = _mm512_sub_epi16(vm1, _mm512_add_epi16(ew, ew));
                __m512i D  = _mm512_loadu_si512((const void *)(p2 + i - 1));
                __m512i U  = _mm512_loadu_si512((const void *)(p1 + i - 1));
                __m512i L  = _mm512_loadu_si512((const void *)(p1 + i));
                __m512i r  = _mm512_max_epi16(_mm512_add_epi16(D, s),
                             _mm512_max_epi16(_mm512_add_epi16(U, vg),
                                              _mm512_add_epi16(L, vg)));
                _mm512_storeu_si512((void *)(cur + i), r);
            }
        }
#endif
#if defined(__AVX2__)
        {
            const __m256i vg = _mm256_set1_epi16((short)GAP);
            const __m256i vp = _mm256_set1_epi16((short)MATCH);
            const __m256i vm = _mm256_set1_epi16((short)MISMATCH);
            for (; i + 15 <= ihi; i += 16) {
                __m128i ca = _mm_loadu_si128((const __m128i *)(a + i - 1));
                __m128i cb = _mm_loadu_si128((const __m128i *)(brb + base + i));
                __m128i eq = _mm_cmpeq_epi8(ca, cb);
                __m256i ew = _mm256_cvtepi8_epi16(eq);
                __m256i s  = _mm256_blendv_epi8(vm, vp, ew);
                __m256i D  = _mm256_loadu_si256((const __m256i *)(p2 + i - 1));
                __m256i U  = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
                __m256i L  = _mm256_loadu_si256((const __m256i *)(p1 + i));
                __m256i r  = _mm256_max_epi16(_mm256_add_epi16(D, s),
                             _mm256_max_epi16(_mm256_add_epi16(U, vg),
                                              _mm256_add_epi16(L, vg)));
                _mm256_storeu_si256((__m256i *)(cur + i), r);
            }
        }
#endif
        {   /* scalar remainder; also the whole loop on non-x86 (auto-vec) */
            const short * __restrict q2 = p2;
            const short * __restrict q1 = p1;
            short       * __restrict cc = cur;
            const char  * __restrict aa = a;
            const char  * __restrict bb = brb;
            for (; i <= ihi; i++) {
                int s   = (aa[i - 1] == bb[base + i]) ? MATCH : MISMATCH;
                int dg  = (int)q2[i - 1] + s;
                int uu  = (int)q1[i - 1] + GAP;
                int ll  = (int)q1[i] + GAP;
                int bst = dg > uu ? dg : uu;
                if (ll > bst) bst = ll;
                cc[i] = (short)bst;
            }
        }

        if (lo == 0) cur[0] = (short)(GAP * d);    /* cell (0,d)  */
        if (hi == d) cur[d] = (short)(GAP * d);    /* cell (d,0)  */
        cur[lo - 1] = NEGV;                        /* guard mounds */
        cur[hi + 1] = NEGV;

        if (d == twon) { result = (int)cur[n]; break; }
        { short *tmp = p2; p2 = p1; p1 = cur; cur = tmp; }  /* the worm curls back */
    }
    free(buf); free(brb);
    return result;
}

/* Same worm, 32-bit mounds, for trays too large for 16-bit sand. */
static int nw_wave32(int n, const char *a, const char *b, int W)
{
    const int NEGV = -1000000000;
    const int sz   = n + 48;
    int  *buf = (int  *)malloc((size_t)3 * (size_t)sz * sizeof(int));
    char *brb = (char *)malloc((size_t)n + 64);
    int *p2, *p1, *cur;
    int d, result = 0;
    size_t t;
    const int twon = 2 * n;

    if (!buf || !brb) { free(buf); free(brb); return nw_rowdp(n, a, b); }
    for (t = 0; t < (size_t)n; t++) brb[t] = b[n - 1 - (int)t];
    memset(brb + n, 0, 64);
    for (t = 0; t < (size_t)3 * (size_t)sz; t++) buf[t] = NEGV;

    p2  = buf + 16;
    p1  = buf + sz + 16;
    cur = buf + 2 * sz + 16;

    for (d = 0; d <= twon; d++) {
        int lo = d - W, hi, ilo, ihi, i, base;
        lo = (lo <= 0) ? 0 : ((lo + 1) >> 1);
        if (lo < d - n) lo = d - n;
        hi = (d + W) >> 1;
        if (hi > n) hi = n;
        if (hi > d) hi = d;

        base = n - d;
        ilo  = (lo < 1) ? 1 : lo;
        ihi  = (hi > d - 1) ? (d - 1) : hi;

        {
            const int  * __restrict q2 = p2;
            const int  * __restrict q1 = p1;
            int        * __restrict cc = cur;
            const char * __restrict aa = a;
            const char * __restrict bb = brb;
            for (i = ilo; i <= ihi; i++) {
                int s   = (aa[i - 1] == bb[base + i]) ? MATCH : MISMATCH;
                int dg  = q2[i - 1] + s;
                int uu  = q1[i - 1] + GAP;
                int ll  = q1[i] + GAP;
                int bst = dg > uu ? dg : uu;
                if (ll > bst) bst = ll;
                cc[i] = bst;
            }
        }

        if (lo == 0) cur[0] = GAP * d;
        if (hi == d) cur[d] = GAP * d;
        cur[lo - 1] = NEGV;
        cur[hi + 1] = NEGV;

        if (d == twon) { result = cur[n]; break; }
        { int *tmp = p2; p2 = p1; p1 = cur; cur = tmp; }
    }
    free(buf); free(brb);
    return result;
}

/* ------------------------------------------------------------------------ *
 * kernel: first the straight-grain pass (the height the worm earns walking
 * the main furrow), which both fences the tray and tells it which regime it
 * is in; then the worm.
 * ------------------------------------------------------------------------ */
int kernel(int n, const char *a, const char *b)
{
    int H = 0, W, i;
    if (n <= 0) return 0;

    /* straight grain: Hamming distance, vectorized, O(n) */
    for (i = 0; i < n; i++) H += (a[i] != b[i]);

    if (H == 0) return n;                 /* cords identical: nothing to do */

    /* Exact band: any optimal path has max offset m <= floor(2H/5).
       Proof: score = n - 2X - 5k with k = Ia = Ib >= m, so a path through
       offset m scores <= n - 5m; the straight grain already guarantees
       S* >= n - 2H; n - 5m < n - 2H whenever m > 2H/5.                     */
    W = (2 * H) / 5;
    if (W < 2) W = 2;                     /* keeps every diagonal non-empty */
    if (W > n) W = n;

    if (n < 96)     return nw_rowdp(n, a, b);   /* tray too small for a worm */
    if (n <= 15000) return nw_wave16(n, a, b, W);
    return nw_wave32(n, a, b, W);
}
```

## PREDICTION

PREDICTION: speedup_vs_dp = 12.0

Reasoning behind the number, stated before any measurement so it is falsifiable: the reference spends ≈ 2.0–2.5 cycles/cell (its `dp[i][j-1]` chain is a serial add→max, ~2 cycles latency, and it touches an `(n+1)²` array that leaves cache above n≈2500). The worm does 16 cells (AVX2) or 32 (AVX-512) per ~9 vector ops ≈ 0.20 cycles/cell, from a working set of three diagonals (≈ 6n bytes, L1/L2 resident). That is ~10–11×. Random DNA gives H ≈ 0.75n ⇒ W ≈ 0.3n ⇒ ≈ 0.51n² cells instead of n², another ~1.9×. Against that, ~2n diagonals of bookkeeping cost ~20 cycles each, which at n = 1000 eats ~30% of the win, and short diagonals near the corners never reach vector width. Net ≈ 12× for n in the low thousands; I expect **6–8× at n ≈ 128–256** (bookkeeping-dominated) and **20–40× at n ≥ 4000** (where the reference's 100 MB+ table becomes memory-bound while the worm does not). If the benchmark feeds *similar* sequences (say 5% divergence), W collapses to ≈ 0.02n and the speedup should exceed 100× — that would be the band, not the SIMD, and I would want to report it as such rather than claim the worm was fast.

## MEASUREMENT

**Not performed — honest report.** `alignment_bench` and `alignment_contract` were declared available but no tool interface exists in this session, so I could not compile, verify against the reference, or time anything. Everything above is prediction, and the "improve it at most four times" loop did not run: what is shipped is iteration zero.

What I *did* verify, by hand rather than by machine, and what a reader should re-check first:

- **Band bound is exact, not heuristic.** `score = n − 2X − 5k`, `k = Ia = Ib ≥ m`, so `m > ⌊2H/5⌋ ⇒ score < n − 2H ≤ S*`. Checked on `ACGT`/`CGTA` (H=4, W=1, optimum −1 via a path of max offset 1 — fits exactly) and on `AAAAAAAAAA`/`TTAAAAAAAA` (H=2, W=0→clamped 2, optimum 6 on the main diagonal, the tempting 2-shift scores 0).
- **Index algebra.** `cur[i] = max(p2[i−1]+s, p1[i−1]+GAP, p1[i]+GAP)` walked through by hand for d = 0, 1, 2 and d = 2n−1, 2n; `lo == 0 ⟺ hi == d`, so every in-band position is written exactly once; reads from diagonal d land inside `[lo−1, hi+1]` of diagonals d−1 and d−2 because `lo` is non-decreasing and `hi` rises by at most 1 per step.
- **No 16-bit overflow.** Every in-band cell satisfies `D ≥ −min(i,j) − 2|i−j| ≥ −1.8n` (since `W ≤ 2n/5`), so at n ≤ 15000 real values stay above −27000 while guards sit at −32000; a guard can be read (−32002) but can never win a max, and every visited cell's diagonal predecessor is provably in-band, so no computed value is ever guard-derived.
- **Untested at runtime:** the AVX-512 block (I have no machine here to confirm `-march=native` even selects it), and the actual constant-factor split between SIMD and band.

## VERDICT

The metaphor was taken at face value and it paid: "crosswise cords" is not imagery, it is the `brev` layout that turns a diagonal of character comparisons into one instruction; "a single ink-worm, not a company of hooded memories" is an explicit instruction to use SIMD and *not* threads, which I followed (no OpenMP appears anywhere — the wavefront has 2n serial steps and barriers would swamp it); "spending no sand" is the letter-blind gap constant; and "the height it has earned" is the Ukkonen band. Every object landed on something concrete.

Two things worth saying plainly rather than dressing up. First, the mechanism converges on two techniques that already exist and are validated — Wozniak's anti-diagonal SIMD and Ukkonen-style adaptive banding — which is the correct outcome under the rules, but it means the novelty here is the *combination and its runtime regime switch*, not a new algorithm. Second, I did not measure, so the 12× is a claim, not a result; if it comes back at 4× the most likely culprits are per-diagonal bookkeeping at small n and the corner diagonals never filling a vector.

The two risks my own reasoning named are both guarded in the shipped code rather than waved at: tiny trays fall back to plain rolling-row DP below n = 96, and trays too large for 16-bit sand fall back to a 32-bit worm above n = 15000, with a rolling-row fallback on allocation failure. Both regimes the known_way describes — the bounded-score/banded case and the full-grid case — are handled by one runtime probe rather than by picking a favourite.