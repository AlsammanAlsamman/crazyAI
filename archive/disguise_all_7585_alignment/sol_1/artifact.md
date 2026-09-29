# APPROACH

**Mapping the disguised solution onto the real problem.** The garden‑path scoreboard is the Needleman–Wunsch DP table; every box must be chalked (I fill *every* cell — no banding, no heuristic early exit, no score-range truncation), and each box still needs exactly the three neighbours: diagonal (match/mismatch, ±1), above and left (gap, −2). The mechanism is unchanged: same recurrence, same full table, same final corner cell.

What I change is the **order of chalking**, which the transformed text explicitly flags as the unexamined house rule: *"both kids start reading from the very same end and walk in the same direction… but there's nothing sacred about that. Mia and Leo are standing at opposite ends."* That is the anti‑diagonal (wavefront) traversal. Cells on anti‑diagonal `d = i+j` depend only on diagonals `d−1` and `d−2`, never on each other — so the serial `dp[i][j-1]` chain that forces the reference to ~2–3 cycles/cell disappears and a whole diagonal is computed in parallel SIMD lanes.

Concretely:
- **"One of them reads from the far end"** is literal here: on diagonal `d`, cell `i` needs `a[i-1]` and `b[d-i-1]`. Walking `i` forward walks `a` forward and `b` *backward*. So I pre-reverse `b` once (`br[k]=b[n-1-k]`); then both character streams are contiguous forward loads and a single `vpcmpeqb` yields 16 (or 32) match/mismatch verdicts at once.
- **Three rolling 1‑D buffers** (`d−2`, `d−1`, current) replace the (n+1)² table: O(n) memory, fully streaming, L1/L2 resident. The reference's 16 MB table at n=2000 is a second, independent source of its slowness.
- **Per lane:** `cur = max(prev2[i-1] + s, max(prev1[i-1], prev1[i]) + GAP)` — the two gap terms share one `+GAP` after a `max`, saving an add.
- **`s` without a blend:** the compare mask is `0xFFFF`/`0x0000`, so `s = (mask & 2) − 1` gives exactly `+1`/`−1`.
- **Boundaries:** the only out-of-range reads are at `i=0` and `i=d` (the `dp[0][d]` / `dp[d][0]` edge cells), and those cells are unconditionally overwritten with `−2d` right after the vector loop — so no sentinels, no masking, no scalar peel. Buffers are padded 64 elements each side so the tail vector may overshoot freely.
- **Width:** int16 lanes (16-wide AVX2, 32-wide AVX‑512BW). Scores live in `[−2n, n]`, so int16 is exact for n ≤ 16000; above that, and for tiny n where per-diagonal overhead dominates, a two-row scalar DP (same recurrence) is used. Same answer bit-for-bit in all paths.

No OpenMP: the wavefront would need 2n barriers, which at these sizes costs more than the whole computation.

# ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(__AVX2__) || defined(__AVX512BW__)
#include <immintrin.h>
#endif

#define NW_MATCH      1
#define NW_MISMATCH  -1
#define NW_GAP       -2

/* ---- exact reference recurrence, two rolling rows (fallback path) ---- */
static int nw_rows(int n, const char *a, const char *b)
{
    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int i, j, r;
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (j = 0; j <= n; j++) prev[j] = NW_GAP * j;
    for (i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = NW_GAP * i;
        for (j = 1; j <= n; j++) {
            int dgl  = prev[j - 1] + (ai == b[j - 1] ? NW_MATCH : NW_MISMATCH);
            int up   = prev[j] + NW_GAP;
            int left = cur[j - 1] + NW_GAP;
            int best = dgl > up ? dgl : up;
            if (left > best) best = left;
            cur[j] = best;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

#if defined(__AVX2__)
/* ---- anti-diagonal wavefront, int16 lanes ---- */
static int nw_wave(int n, const char *a, const char *b)
{
    const int PAD = 64;
    const size_t clen = (size_t)n + 2 * (size_t)PAD;   /* bytes per char buffer  */
    const size_t blen = (size_t)n + 2 * (size_t)PAD;   /* int16 per score buffer */
    unsigned char *cbuf;
    int16_t *ibuf, *p2, *p1, *p0;
    unsigned char *ap, *bp;
    int d, k, result;

    cbuf = (unsigned char *)malloc(2 * clen);
    ibuf = (int16_t *)calloc(3 * blen, sizeof(int16_t));
    if (!cbuf || !ibuf) { free(cbuf); free(ibuf); return nw_rows(n, a, b); }

    memset(cbuf, 0, 2 * clen);
    ap = cbuf + PAD;                 /* a forward                 */
    bp = cbuf + clen + PAD;          /* b reversed: bp[k]=b[n-1-k] */
    memcpy(ap, a, (size_t)n);
    for (k = 0; k < n; k++) bp[k] = (unsigned char)b[n - 1 - k];

    p2 = ibuf + PAD;                 /* diagonal d-2 */
    p1 = ibuf + blen + PAD;          /* diagonal d-1 */
    p0 = ibuf + 2 * blen + PAD;      /* diagonal d   */

    p2[0] = 0;                                   /* d = 0 */
    p1[0] = (int16_t)NW_GAP; p1[1] = (int16_t)NW_GAP;   /* d = 1 */

#if defined(__AVX512BW__) && defined(__AVX512F__)
    {
        const __m512i vgap = _mm512_set1_epi16(NW_GAP);
        const __m512i vone = _mm512_set1_epi16(1);
        const __m512i vtwo = _mm512_set1_epi16(2);
        for (d = 2; d <= 2 * n; d++) {
            int lo = d - n; int hi = (d < n) ? d : n; int bb = n - d; int i;
            if (lo < 0) lo = 0;
            for (i = lo; i <= hi; i += 32) {
                __m256i ca = _mm256_loadu_si256((const __m256i *)(ap + (i - 1)));
                __m256i cb = _mm256_loadu_si256((const __m256i *)(bp + (i + bb)));
                __m512i eq = _mm512_cvtepi8_epi16(_mm256_cmpeq_epi8(ca, cb));
                __m512i s  = _mm512_sub_epi16(_mm512_and_si512(eq, vtwo), vone);
                __m512i dg = _mm512_add_epi16(
                                 _mm512_loadu_si512((const void *)(p2 + (i - 1))), s);
                __m512i u  = _mm512_loadu_si512((const void *)(p1 + (i - 1)));
                __m512i l  = _mm512_loadu_si512((const void *)(p1 + i));
                __m512i g  = _mm512_add_epi16(_mm512_max_epi16(u, l), vgap);
                _mm512_storeu_si512((void *)(p0 + i), _mm512_max_epi16(dg, g));
            }
            if (d <= n) {
                int16_t e = (int16_t)(NW_GAP * d);
                p0[0] = e; p0[d] = e;
            }
            { int16_t *t = p2; p2 = p1; p1 = p0; p0 = t; }
        }
    }
#else
    {
        const __m256i vgap = _mm256_set1_epi16(NW_GAP);
        const __m256i vone = _mm256_set1_epi16(1);
        const __m256i vtwo = _mm256_set1_epi16(2);
        for (d = 2; d <= 2 * n; d++) {
            int lo = d - n; int hi = (d < n) ? d : n; int bb = n - d; int i;
            if (lo < 0) lo = 0;
            for (i = lo; i <= hi; i += 16) {
                __m128i ca = _mm_loadu_si128((const __m128i *)(ap + (i - 1)));
                __m128i cb = _mm_loadu_si128((const __m128i *)(bp + (i + bb)));
                __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
                __m256i s  = _mm256_sub_epi16(_mm256_and_si256(eq, vtwo), vone);
                __m256i dg = _mm256_add_epi16(
                                 _mm256_loadu_si256((const __m256i *)(p2 + (i - 1))), s);
                __m256i u  = _mm256_loadu_si256((const __m256i *)(p1 + (i - 1)));
                __m256i l  = _mm256_loadu_si256((const __m256i *)(p1 + i));
                __m256i g  = _mm256_add_epi16(_mm256_max_epi16(u, l), vgap);
                _mm256_storeu_si256((__m256i *)(p0 + i), _mm256_max_epi16(dg, g));
            }
            if (d <= n) {
                int16_t e = (int16_t)(NW_GAP * d);
                p0[0] = e; p0[d] = e;
            }
            { int16_t *t = p2; p2 = p1; p1 = p0; p0 = t; }
        }
    }
#endif

    result = (int)p1[n];             /* after the last rotation p1 holds diagonal 2n */
    free(cbuf); free(ibuf);
    return result;
}
#endif /* __AVX2__ */

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
#if defined(__AVX2__)
    /* int16 is exact while |score| <= 2n stays inside the type */
    if (n >= 16 && n <= 16000) return nw_wave(n, a, b);
#endif
    return nw_rows(n, a, b);
}
```

# PREDICTION

PREDICTION: 14

(14× speedup over the reference `kernel`, geometric mean across benchmarked `n`. Reasoning: the reference is pinned at ~2–3 cycles/cell by the `dp[i][j-1]→add→max` loop-carried chain plus (n+1)²·4 B of table traffic; the wavefront issues 16 cells per ~3 cycles ≈ 0.19 cycles/cell with an O(n) working set. Raw ratio ~12–16×, expected to exceed 20× at n where the reference table spills L2 and to fall to ~6–8× at small n where per-diagonal overhead bites; 32× AVX‑512 lanes, if present, would push it higher but are load/store-port limited, not lane limited.)

# MEASUREMENT

Not performed. No tools were available in this session — `alignment_bench` and `alignment_contract` could not be invoked, so this artifact is **unmeasured and unverified against the reference**. I am reporting that plainly rather than inventing numbers, and none of the four permitted improvement rounds were spent, because there was no measurement to drive them.

What I did verify, by hand rather than by machine:
- Hand-traced the full wavefront on `a="AC"`, `b="AG"` (n=2), cell by cell against the reference table: diagonals 0–4 reproduce `dp[1][1]=1`, `dp[1][2]=−1`, `dp[2][1]=−1`, `dp[2][2]=0`. Final read `p1[n]` returns 0, matching.
- Audited every index range: `ap+(i−1) ∈ [−1, n+30]`, `bp+(i+bb) ∈ [0, n+30]`, `p2+(i−1) ∈ [−1, n+30]`, `p0+i ∈ [lo, n+31]` — all inside the 64-element pads on both sides of every buffer. Both pointer offsets are written as `ptr + (i + bb)` so the negative `bb` never forms an out-of-range intermediate pointer.
- Audited the garbage-read argument: the only cells reading unwritten slots are `i=0` and `i=d` for `d ≤ n`, both unconditionally overwritten with `−2d`; and tail-overshoot writes past `hi` are only ever re-read at positions that are themselves overwritten boundary cells (checked separately for `d<n−1`, `d=n−1`, `d=n`, `d>n`). Buffers are `calloc`'d so nothing uninitialized is ever loaded.
- Confirmed the int16 range guard: values stay in `[−2n−2, n+1]`, safe for `n ≤ 16000`; outside that the scalar path runs.

Untested risks I would want the harness to settle first: the AVX‑512 branch (never compiled here), and whether `n < 16` / `n > 16000` ever appear in the benchmark at all.

# VERDICT

The mechanism is preserved exactly — full table, same three-neighbour recurrence, same scores — and the single assumption the transformed problem invited me to drop (both strings marched start-to-end in lockstep, one cell at a time) is the one I dropped, by chalking the board along anti-diagonals with `b` read from the far end. That is what makes 16–32 cells fall out per iteration instead of one.

But the honest verdict is **unproven**. I predicted 14× and then had no way to measure it, so the claim stands as an estimate with a hand-verified correctness argument behind it, not a result. If the harness compiles and runs it, the two things most likely to go wrong are the AVX‑512 block and a size regime I guessed at; if the measured number comes back well under 14×, the first thing to check is per-diagonal overhead at small `n`, which would argue for raising the scalar threshold rather than for changing the approach.