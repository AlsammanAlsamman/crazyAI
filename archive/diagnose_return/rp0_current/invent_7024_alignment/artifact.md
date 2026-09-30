# MAPPING

## SEED 1 — coils walked side by side on a sunlit water table; each match drops a counted fire

| World object | Problem object |
|---|---|
| a coil, *shaped exactly like its symbol* | one base, as its byte code in a lane |
| the near row of coils | `a[0..n-1]`, contiguous |
| the shadow row, "waiting their turn like a queue at a door" | `b`, read at an **offset** (the queue position = the shift) |
| full sun overhead | one comparison operation applied to all lanes at once |
| "the sun windows straight through both" | byte-equality of two coincident shapes (`pcmpeqb`) |
| a small fire dropped on the water | a set bit/lane in the match mask |
| "I count the fires, nothing more" | the score is a **count**, `2·matches − length`, not a table |
| the water table below | the accumulator the fires fall into |
| **Breaks** | "one pair of positions is judged at a time" and "the whole grid of every position against every other must be filled in" |

## SEED 2 — the door between the rows breaks exactly once per crossing

| World object | Problem object |
|---|---|
| the door/boundary between near row and shadow row | the alignment offset `d = j − i` |
| a quarrel (two coils refusing each other's shape) | a mismatch at the current offset |
| "the door breaks, once" | `d` changes by 1 — one gap step |
| "the next coil crouches forward, skipping the coil that quarreled" | `b` advances two while `a` advances one: `a[i]` now meets `b[i+1]` |
| "I try this breaking at every place a quarrel happens" | **every hinge position `k` is tried independently**, each a complete full-length walk |
| "never twice in one crossing" | at most one gap-pair per candidate |
| "ends at the last coil, at the queen's threshold" | the walk runs to `(n,n)`; the dangling end is the matching second gap |
| **Breaks** | **"a slip (gap) can only be discovered by having already compared the position before it."** The native *chooses* the slip site up front and then walks a plain offset comparison — no prefix DP value is consulted anywhere |

## SEED 3 — rooms, the underground tower, the highest pile

| World object | Problem object |
|---|---|
| a crossing that finishes | one candidate alignment (offset 0, or hinged 0→1) |
| its pile of fire, set in a reserved room | one accumulator per candidate `k` |
| the tower under the ground, rows thrown in unread | candidates rejected without scoring (>1 slip) |
| "bring up only the room whose pile burned highest" | max-reduction over candidates → a **lower bound on the score** |
| **Breaks** | "the whole grid must be filled in" — only `O(n)` complete walks are scored and maxed |

# CHOSEN SEED

**SEED 2.** It is the only one of the three that breaks the preferred assumption, and it is the most literal: its objects (door, offset, hinge site, once-only) are one-to-one with alignment offsets and gap steps.

# ASSUMPTION BROKEN

*"A slip (gap) can only be discovered by having already compared the position before it."*

In the native's world the slip is **declared, not derived**. Each hinge site is an independent hypothesis, and the crossing that follows is a pure offset-shifted scan — embarrassingly parallel, no recurrence.

What falls out when I take this seriously: the native's max-over-crossings is not the answer, it is a **certified floor** under the answer, and a floor on the score is a *ceiling on how far the door may ever travel*. Score of any alignment with `g` gap-pairs is `≤ (n−g) − 4g = n − 5g`, and the offset never exceeds `g`. So

```
LB = best of the native's crossings      ⇒      |i − j| ≤ W = ⌊(n − LB)/5⌋
```

is a *proof-carrying band*. The native's fire-count measures how much the cords agree, and that measurement alone decides how wide a search is still needed — this is the runtime regime detector demanded by step 5: near-identical cords ⇒ `W` collapses to a few cells; random cords ⇒ `W ≈ 0.3n` and the kernel keeps the full-width wavefront. This is exactly the validated Ukkonen/KSW2 score-bounded band, arrived at rather than imported (step 4).

Inside the band, the crossing itself is walked as the native walks it: an **anti-diagonal wavefront**, where the sun judges 16 coil-pairs simultaneously. In wavefront coordinates the three neighbours are single-lane shifts of the two previous wavefronts, so there is no lazy-F serial chain: `a` forward, reversed `b` forward, `vpcmpeqb` + `vpmaxsd`.

Literal mapping of the machine: **memory** = the three rolling wavefront rows (the water surface, only the last two ripples retained) + the two cords; **what flows** = the wavefront, one anti-diagonal per tick; **what stays still** = the cords; **the processors** = the SIMD lanes, one per coil-pair judged in the same instant of sun; **time** = `k = i+j`; **the tower** = cells outside `W`, poisoned and never read back.

Four revisions (design-time, not measurement-guided — see MEASUREMENT):
1. naive: native's max-over-crossings alone → **rejected, inexact** (can't match the reference when the optimum needs ≥2 gaps);
2. crossings → certified band + banded row DP → serial `left` dependency blocks SIMD;
3. band + anti-diagonal wavefront, `int32`, `restrict` (auto-vectorizable);
4. explicit AVX2 16-lane block + 8-lane block + scalar tail; `n < 64` and malloc-failure fall back to the plain two-row DP.

# ARTIFACT

```c
#include <stdlib.h>
#include <stddef.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define ALN_M   1
#define ALN_X  (-1)
#define ALN_G  (-2)
#define ALN_POISON (-(1 << 28))

/* exact two-row Needleman-Wunsch: small-n path and safety fallback */
static int aln_rows(int n, const char *restrict a, const char *restrict b)
{
    int sp[64], sc[64];
    int *prev, *cur, *heap = NULL;
    if (n + 1 <= 64) { prev = sp; cur = sc; }
    else {
        heap = (int *)malloc(2u * (size_t)(n + 1) * sizeof(int));
        if (!heap) return 0;
        prev = heap; cur = heap + (n + 1);
    }
    for (int j = 0; j <= n; j++) prev[j] = j * ALN_G;
    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = i * ALN_G;
        for (int j = 1; j <= n; j++) {
            int d = prev[j - 1] + (ai == b[j - 1] ? ALN_M : ALN_X);
            int u = prev[j] + ALN_G;
            int l = cur[j - 1] + ALN_G;
            int m = d > u ? d : u;
            cur[j] = m > l ? m : l;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    { int r = prev[n]; if (heap) free(heap); return r; }
}

int kernel(int n, const char *a, const char *b)
{
    const char *restrict A = a;
    const char *restrict B = b;

    if (n <= 0) return 0;
    if (n < 64) return aln_rows(n, A, B);   /* guard: no setup overhead on small n */

    /* ---- phase 1: the crossings.  Walk the cords, count the fires. ----
       straight crossing, and every single-slip crossing in both lay orders.
       Each finished crossing is a real alignment, so the highest pile is a
       certified lower bound on the score.                                  */
    {
        int lb, W;
        int m0 = 0;
        for (int i = 0; i < n; i++) m0 += (A[i] == B[i]);
        lb = 2 * m0 - n;                                  /* no door broken */

        {   /* shadow row = B: B crouches past the quarrel at k  */
            int s = 0, bs;
            for (int i = 0; i + 1 < n; i++) s += (A[i] == B[i + 1]);
            bs = s;
            for (int k = 1; k < n; k++) {
                s += (A[k - 1] == B[k - 1]) - (A[k - 1] == B[k]);
                if (s > bs) bs = s;
            }
            { int c = 2 * bs - n - 3; if (c > lb) lb = c; }
        }
        {   /* the cords laid in the other order */
            int s = 0, bs;
            for (int i = 0; i + 1 < n; i++) s += (A[i + 1] == B[i]);
            bs = s;
            for (int k = 1; k < n; k++) {
                s += (A[k - 1] == B[k - 1]) - (A[k] == B[k - 1]);
                if (s > bs) bs = s;
            }
            { int c = 2 * bs - n - 3; if (c > lb) lb = c; }
        }

        /* ---- phase 2: how far may the door ever travel? ----
           any alignment with g gap-pairs scores <= n - 5g and never leaves
           offset g, so an optimal path obeys |i-j| <= (n - lb)/5.          */
        W = (n - lb) / 5;
        if (W < 1) W = 1;
        if (W > n) W = n;

        {   /* ---- phase 3: walk the banded wavefront ---- */
            size_t cap = (size_t)n + 8;
            int *buf = (int *)malloc(3u * cap * sizeof(int));
            char *br = (char *)malloc((size_t)n + 24);
            int *c0, *c1, *c2, res;
            if (!buf || !br) { free(buf); free(br); return aln_rows(n, A, B); }

            for (int i = 0; i < n; i++) br[i] = B[n - 1 - i];
            for (int i = 0; i < 24; i++) br[n + i] = 0;
            for (size_t i = 0; i < 3u * cap; i++) buf[i] = ALN_POISON;

            c0 = buf + 2;                 /* wavefront k-2, index = row i   */
            c1 = buf + cap + 2;           /* wavefront k-1                  */
            c2 = buf + 2 * cap + 2;       /* wavefront k, being filled      */
            c1[0] = 0;                    /* dp[0][0]                       */

            for (int k = 1; k <= 2 * n; k++) {
                int t  = k - W; if (t < 0) t = 0;
                int lo = (t + 1) >> 1;                 /* ceil((k-W)/2)     */
                int hi = (k + W) >> 1;                 /* floor((k+W)/2)    */
                int ilo, ihi, off, i;
                const int *restrict p0;
                const int *restrict p1;
                int *restrict p2;

                if (k - n > lo) lo = k - n;
                if (hi > n) hi = n;
                if (hi > k) hi = k;

                ilo = lo > 1 ? lo : 1;
                ihi = hi < k - 1 ? hi : k - 1;
                off = n - k;
                p0 = c0; p1 = c1; p2 = c2;
                i = ilo;
#if defined(__AVX2__)
                {
                    const __m256i vg  = _mm256_set1_epi32(ALN_G);
                    const __m256i vm1 = _mm256_set1_epi32(-1);
                    for (; i + 15 <= ihi; i += 16) {
                        __m128i ca = _mm_loadu_si128((const __m128i *)(A + i - 1));
                        __m128i cb = _mm_loadu_si128((const __m128i *)(br + off + i));
                        __m128i eq = _mm_cmpeq_epi8(ca, cb);
                        __m256i e0 = _mm256_cvtepi8_epi32(eq);
                        __m256i e1 = _mm256_cvtepi8_epi32(_mm_srli_si128(eq, 8));
                        __m256i s0 = _mm256_sub_epi32(vm1, _mm256_add_epi32(e0, e0));
                        __m256i s1 = _mm256_sub_epi32(vm1, _mm256_add_epi32(e1, e1));
                        __m256i d0 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(p0 + i - 1)), s0);
                        __m256i d1 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(p0 + i + 7)), s1);
                        __m256i u0 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(p1 + i - 1)), vg);
                        __m256i u1 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(p1 + i + 7)), vg);
                        __m256i l0 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(p1 + i)), vg);
                        __m256i l1 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(p1 + i + 8)), vg);
                        _mm256_storeu_si256((__m256i *)(p2 + i),
                            _mm256_max_epi32(_mm256_max_epi32(d0, u0), l0));
                        _mm256_storeu_si256((__m256i *)(p2 + i + 8),
                            _mm256_max_epi32(_mm256_max_epi32(d1, u1), l1));
                    }
                    for (; i + 7 <= ihi; i += 8) {
                        __m128i ca = _mm_loadl_epi64((const __m128i *)(A + i - 1));
                        __m128i cb = _mm_loadl_epi64((const __m128i *)(br + off + i));
                        __m128i eq = _mm_cmpeq_epi8(ca, cb);
                        __m256i e0 = _mm256_cvtepi8_epi32(eq);
                        __m256i s0 = _mm256_sub_epi32(vm1, _mm256_add_epi32(e0, e0));
                        __m256i d0 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(p0 + i - 1)), s0);
                        __m256i u0 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(p1 + i - 1)), vg);
                        __m256i l0 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(p1 + i)), vg);
                        _mm256_storeu_si256((__m256i *)(p2 + i),
                            _mm256_max_epi32(_mm256_max_epi32(d0, u0), l0));
                    }
                }
#endif
                for (; i <= ihi; i++) {
                    int e = (A[i - 1] == br[off + i]);
                    int d = p0[i - 1] + (e ? ALN_M : ALN_X);
                    int u = p1[i - 1] + ALN_G;
                    int l = p1[i] + ALN_G;
                    int m = d > u ? d : u;
                    p2[i] = m > l ? m : l;
                }
                if (lo == 0) p2[0] = k * ALN_G;   /* dp[0][k] */
                if (hi == k) p2[k] = k * ALN_G;   /* dp[k][0] */
                p2[lo - 1] = ALN_POISON;          /* the tower: unread cells */
                p2[hi + 1] = ALN_POISON;

                { int *tmp = c0; c0 = c1; c1 = c2; c2 = tmp; }
            }
            res = c1[n];
            free(buf); free(br);
            return res;
        }
    }
}
```

# PREDICTION

**PREDICTION: speedup_vs_dp = 12**

(Reasoning behind the single number, stated before any measurement: on uniformly random DNA the fire-count gives `LB ≈ 2·(n/4) − n = −n/2`, hence `W ≈ 0.3n` and a band holding `≈ 0.51 n²` cells — about 2× fewer. The wavefront adds an estimated 4–6× from 8-lane `int32` AVX2, and the reference's `(n+1)²` table adds page-fault/bandwidth cost we avoid with `O(n)` memory. `2 × 5 ≈ 10–15`. If the benchmark uses *similar* sequences the band collapses — at 90 % identity `W ≈ 0.04n`, ~12× fewer cells — and the speedup should be far above 12; I am deliberately predicting the random-sequence regime, the less favourable one.)

# MEASUREMENT

**Not obtained.** This session was invoked with no tools available — `alignment_bench` and `alignment_contract` were never callable, so I have no measured `speedup_vs_dp` and no harness-verified correctness run. I am reporting that plainly rather than inventing numbers: the prediction above is unvalidated, and the pipeline's measurement supersedes it.

What I *could* verify by hand, and did:

- **Band validity (proof).** Both strings have length `n`, so any alignment uses `g` gaps in each; aligned pairs `= n − g`, gap columns `= 2g`, so `score ≤ (n−g) − 4g = n − 5g`. At any point `i − j =` (gaps in `b` so far) − (gaps in `a` so far), so `max|i−j| ≤ g`. With `LB` achievable, the optimum obeys `|i−j| ≤ ⌊(n−LB)/5⌋ = W`. The `LB`-achieving crossings themselves have offset `≤ 1 ≤ W`, so the band is non-empty and `dp_band[n][n] = S*` **exactly**, not approximately.
- **Crossing scores.** Straight: `2M₀ − n`. Hinged at `k`: `n−1` aligned pairs and exactly 2 gaps (the interior slip plus its dangling partner at the threshold) ⇒ `2m(k) − n − 3`, with `m(k) − m(k−1) = [a_{k−1}=b_{k−1}] − [a_{k−1}=b_k]`, i.e. one `O(n)` walk per lay order.
- **Wavefront recurrence.** With `k = i+j` indexed by row `i`: diag `= C_{k−2}[i−1]`, up `= C_{k−1}[i−1]`, left `= C_{k−1}[i]` — pure lane shifts, verified by hand at `k = 1,2,3` (`dp[1][1] = max(0+s, −4, −4)`; `dp[1][2]`'s "up" correctly reads a poisoned out-of-band `(0,2)`), and at `k = 2n−1, 2n` where `lo = hi = n`.
- **Window monotonicity** (`lo`, `hi` both non-decreasing, each stepping by `≤ 1`) ⇒ writing `POISON` at `lo−1` and `hi+1` after every wavefront is sufficient; no stale in-band value is ever read. `W ≥ 1` keeps every anti-diagonal non-empty, removing the parity special case.
- **Poison arithmetic.** Poisoned values drift by at most `±2` per wavefront over `2n` wavefronts: `−2²⁸ + 2n < −4n` for all `n < 4.4×10⁷`, so poison can never win a `max` against a real score (`≥ −4n`).
- **No overread of caller memory.** The 16-lane block runs only while `i+15 ≤ ihi ≤ n`, giving `i ≤ n−15`, so `A[i−1 … i+14]` stays inside `a`; `i+15 ≤ k−1` gives `off+i+15 ≤ n−1` for the reversed copy.
- **Hand traces** (`n=1` mismatch → `−1`; `n=2` `AC/CA` → `−2`; a 16-mer shifted by one → `11`, where the door-breaking drives `W` from 6 down to 1). All of these route through `aln_rows`, which is a literal clone of the reference recurrence.

# VERDICT

The native's idea survives translation intact and does real work: the door that breaks once is the *measuring instrument*. Its pile of fires is not the answer — it is a certificate that bounds how far any answer's door can wander, and that certificate is what shrinks the grid. The two things the metaphor insisted on that the textbook method does not have are both load-bearing: (1) slips declared up front rather than derived from a predecessor cell, which makes the bound computable in `O(n)` with no DP at all; (2) many independent crossings scored into separate rooms and max-reduced, which is what turns a scan into a proof.

Honest accounting of where it is weaker than the known way, and how each is addressed rather than hand-waved:

- **Small `n`:** setup (`LB` scan, reverse copy, buffer poisoning) would dominate. **Guarded**: `n < 64` takes the plain two-row DP. Also the malloc-failure path.
- **Dissimilar / random cords:** the bound is loose — `W ≈ 0.3n`, only ~2× fewer cells. Not *worse* than the known way (work is monotone `≤` full matrix, memory drops from `O(n²)` to `O(n)`, and the wavefront is vectorized regardless), so no fallback is needed, but the speedup there is modest and my 12 is aimed at that regime. The two regimes named in `known_way` (full vs. banded) are distinguished at runtime by the fire-count itself; there is no fixed band constant anywhere.
- **The one-slip family contributes only `O(√n)` to `LB` on random input** — I will not pretend otherwise. Where it is decisive is the realistic DNA case: two similar cords offset by an indel, where the straight crossing scores `≈ −n/2` and a single hinged crossing scores `≈ +n`, collapsing `W` from `0.3n` to `O(1)`. That is the native's contribution, and it is exactly the case the diagonal-only bound handles worst.
- **No thread parallelism.** The metaphor's unit of work is one wavefront, `≤ 2W+1` cells wide — far too small to pay for a barrier per tick at these sizes. Vectorization hints only, per instruction.

If the harness reports a mismatch against the reference on any input, the band proof is the place to look first, and the honest fix is `W = n` (the wavefront alone, still vectorized, still `O(n)` memory) rather than any loosening of exactness.