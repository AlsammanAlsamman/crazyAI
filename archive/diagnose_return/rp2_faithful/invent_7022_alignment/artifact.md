## MAPPING

**SEED 1 — "Two knot-cords laid crosswise over a sand-tray mark every place their symbols meet."**

| world object | problem object |
|---|---|
| knot-cord running north, horse's length | string `a`, index `i = 0..n` |
| knot-cord running east, mountain's width | string `b`, index `j = 0..n` |
| sand-tray | the score lattice, *as a surface*, not as a stored table |
| furrow-point / junction | the pair `(i,j)` and its score `H[i][j]` |
| mound height at a junction | the DP value |
| crossing "marks" the lattice | the lattice is *implied* by the two cords, so it need never be materialized |

*Breaks:* "the whole grid of every position against every other must be filled in" — the tray is a surface the worm re-smooths, not a ledger; only O(n) of it exists at once.

**SEED 2 — "A single ink-worm walks the eight desert winds and curls back across its own trail to test a junction twice."**

| world object | problem object |
|---|---|
| ink-worm, one cell wide, "a stitch cut loose from the verses" | one SIMD register: a 16-lane int16 vector, a single one-cell-thick slice cut out of the grid |
| the worm's **body**, which occupies many junctions at once | one anti-diagonal `i + j = d`: every junction on it is mutually independent, so a whole body-length is judged in one settling |
| the worm's crawl from near corner to far corner | time = `d = 0 … 2n`; the wavefront index is the clock |
| "curls its own body across ground already crossed" | the body's two overlapping unaligned reads of the *same* previous diagonal at offsets `i-1` and `i` — the same ground read twice, from two different faces |
| "tries the junction from another face" | the three-way `max` over the three predecessor faces, evaluated in-lane |
| "keeps no ledger apart from the sand itself" | three rolling O(n) buffers, no `n²` table, no separate traceback store |
| "a beaten mound is thrown away entirely" | `max` is destructive; nothing is retained |
| "done when no junction holds a mound it could improve" | the wavefront order is a topological order, so the fixpoint is reached in exactly one crawl |

*Breaks:* **"one pair of positions is judged at a time"** (the worm's body settles 16 junctions in one drop), *and* "every cell depends on above/left/diagonal computed in that order" (the traversal order is by anti-diagonal, not row-major), *and* "both strings are read start to end in the same direction" (to make the body's two symbol-reads both run forward in memory, `b` must be laid down reversed).

**SEED 3 — "The worm crosses a furrow sideways, spending no sand, to let one cord slip a step ahead of the other."**

| world object | problem object |
|---|---|
| sideways crossing, spending nothing | the gap move, a pure shift with no substitution lookup |
| "one permitted stumble" | `GAP` applied once, uniformly, to both prev-diagonal faces |
| "spending no sand" | no character comparison needed on gap edges — hence both gap candidates are the *same* constant added to two neighbouring slots of one buffer |

*Breaks:* "a slip (gap) can only be discovered by having already compared the position before it" — the slip is a face of the junction, available with no comparison at all.

## CHOSEN SEED

**SEED 2.** It is the only one of the three that breaks *"one pair of positions is judged at a time"*, which the instructions tell me to prefer, and it is the most literal: a worm one cell wide whose **body** lies across the tray is not a metaphor for a vector, it *is* a vector, and the set of junctions a curled body can simultaneously occupy in a lattice — each independent of the others — is forced by geometry to be an anti-diagonal. It is also maximally distant from the known way, which fills a row-major `n²` table one cell at a time.

## ASSUMPTION BROKEN

Primary: **one pair of positions is judged at a time.** Sixteen pairs are judged per settling, because the worm's body is sixteen junctions long.
Secondary: **the fixed above/left/diagonal order** (replaced by wavefront order), **both strings read in the same direction** (`b` is laid down as a reversed cord so the body's two symbol reads are both forward-contiguous), and **the whole grid must be filled in** (three O(n) sand-troughs only).

## ARTIFACT

Literal correspondences in the code:

- `nw_worm16` / `nw_worm32` — the worm. `d0/d1/d2` are the sand itself: the current diagonal and the two behind it. There is **no other ledger**.
- the `for (d = 2; d <= 2*n; d++)` loop — the crawl; `d` is the clock.
- `ilo/ihi` — how much of the body is actually on the tray at this moment (the worm's body shortens near the two corners).
- `_mm256_loadu_si256(q1m + k)` and `_mm256_loadu_si256(q1 + k)` — **the curl**: the same diagonal `B` read twice at offsets `i-1` and `i`, the body crossing ground it has already crossed, to test each junction from a second face.
- `_mm256_loadu_si256(q2m + k)` + `sc` — the diagonal slant, "spending a grain of sand on the difference" (SEED 3's furrow read as the substitution face).
- `GV` added to both curl-reads — the sideways crossing that spends nothing on comparison (SEED 3).
- `_mm256_max_epi16(dg, max(up, lf))` — "if its own tally beats the standing mound it kicks the old one flat"; destructive, nothing kept.
- `br[k] = b[n-1-k]` — the east cord laid down against its own grain, so that both symbol-reads under the body run forward.
- `C[0] = C[d] = -2d` — the two tray edges the body's ends rest on.
- termination: no re-sweep loop exists, because every predecessor of a junction on diagonal `d` lies on `d-1` or `d-2`, both already final — the native's "no junction anywhere still holds a mound it could improve" is discharged analytically after one crawl, not searched for.
- **regime recognition (step 5):** the native's own unit is *how deep a cup a mound needs*. Score range is `[-2n, n]`, so a short tray takes shallow cups (16 int16 lanes) and a long tray takes deep cups (8 int32 lanes); `n < 64` is a tray too small to be worth waking a worm at all and falls back to a single hooded memory (`nw_scalar`, rolling-row, still O(n) memory). No AVX2 → same scalar fallback. This is the guard demanded by step 4 for my own stated risk. No thread parallelism: the worm is one worm, and at these sizes its body is already the right grain of work.

```c
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH -1
#define GAP      -2

/* The single hooded memory: rolling-row DP, O(n) scratch.
   Used for trays too small to be worth waking a worm, and where no winds blow. */
static int nw_scalar(int n, const char *restrict a, const char *restrict b)
{
    int *row = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!row) return 0;
    for (int j = 0; j <= n; j++) row[j] = j * GAP;
    for (int i = 1; i <= n; i++) {
        int diag = row[0];
        row[0] = i * GAP;
        const char ai = a[i - 1];
        for (int j = 1; j <= n; j++) {
            int up = row[j];
            int v  = diag + (ai == b[j - 1] ? MATCH : MISMATCH);
            int t  = up + GAP;        if (t > v) v = t;
            t      = row[j - 1] + GAP; if (t > v) v = t;
            diag = up;
            row[j] = v;
        }
    }
    int r = row[n];
    free(row);
    return r;
}

#if defined(__AVX2__)
/* ---- the worm, shallow cups: 16 int16 mounds settled per drop ---- */
static int nw_worm16(int n, const char *restrict a, const char *restrict b)
{
    const int PAD = 40;
    char  *ap = (char *)malloc((size_t)n + 80);
    char  *br = (char *)malloc((size_t)n + 80);
    size_t dl = (size_t)n + 1 + PAD;
    short *d2 = (short *)calloc(dl, sizeof(short));
    short *d1 = (short *)calloc(dl, sizeof(short));
    short *d0 = (short *)calloc(dl, sizeof(short));
    if (!ap || !br || !d2 || !d1 || !d0) {
        free(ap); free(br); free(d2); free(d1); free(d0);
        return nw_scalar(n, a, b);
    }
    memcpy(ap, a, (size_t)n);
    memset(ap + n, 'X', 80);                       /* north cord, grain forward  */
    for (int k = 0; k < n; k++) br[k] = b[n - 1 - k]; /* east cord, laid reversed */
    memset(br + n, 'Z', 80);

    const __m256i TWO = _mm256_set1_epi16(2);
    const __m256i ONE = _mm256_set1_epi16(1);
    const __m256i GV  = _mm256_set1_epi16(GAP);

    short *A = d2, *B = d1, *C = d0;
    A[0] = 0;                                      /* diagonal 0 */
    B[0] = (short)GAP; B[1] = (short)GAP;          /* diagonal 1 */

    for (int d = 2; d <= 2 * n; d++) {
        int ilo = d - n; if (ilo < 1) ilo = 1;
        int ihi = d - 1; if (ihi > n) ihi = n;
        int cnt = ihi - ilo + 1;
        if (cnt > 0) {
            const char  *pa  = ap + (ilo - 1);
            const char  *pb  = br + (n - d + ilo);
            const short *q2m = A  + (ilo - 1);     /* diagonal d-2, face i-1 */
            const short *q1m = B  + (ilo - 1);     /* diagonal d-1, face i-1 */
            const short *q1  = B  + ilo;           /* diagonal d-1, face i   */
            short       *op  = C  + ilo;
            for (int k = 0; k < cnt; k += 16) {
                __m128i ca = _mm_loadu_si128((const __m128i *)(pa + k));
                __m128i cb = _mm_loadu_si128((const __m128i *)(pb + k));
                __m256i m  = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
                __m256i sc = _mm256_sub_epi16(_mm256_and_si256(m, TWO), ONE);
                __m256i dg = _mm256_add_epi16(
                                 _mm256_loadu_si256((const __m256i *)(q2m + k)), sc);
                __m256i up = _mm256_add_epi16(
                                 _mm256_loadu_si256((const __m256i *)(q1m + k)), GV);
                __m256i lf = _mm256_add_epi16(
                                 _mm256_loadu_si256((const __m256i *)(q1  + k)), GV);
                _mm256_storeu_si256((__m256i *)(op + k),
                    _mm256_max_epi16(dg, _mm256_max_epi16(up, lf)));
            }
        }
        if (d <= n) {                              /* the two tray edges */
            short e = (short)(-2 * d);
            C[0] = e; C[d] = e;
        }
        short *t = A; A = B; B = C; C = t;
    }
    int res = (int)B[n];
    free(ap); free(br); free(d2); free(d1); free(d0);
    return res;
}

/* ---- the worm, deep cups: 8 int32 mounds, for trays long enough to overflow ---- */
static int nw_worm32(int n, const char *restrict a, const char *restrict b)
{
    const int PAD = 24;
    char *ap = (char *)malloc((size_t)n + 80);
    char *br = (char *)malloc((size_t)n + 80);
    size_t dl = (size_t)n + 1 + PAD;
    int *d2 = (int *)calloc(dl, sizeof(int));
    int *d1 = (int *)calloc(dl, sizeof(int));
    int *d0 = (int *)calloc(dl, sizeof(int));
    if (!ap || !br || !d2 || !d1 || !d0) {
        free(ap); free(br); free(d2); free(d1); free(d0);
        return nw_scalar(n, a, b);
    }
    memcpy(ap, a, (size_t)n);
    memset(ap + n, 'X', 80);
    for (int k = 0; k < n; k++) br[k] = b[n - 1 - k];
    memset(br + n, 'Z', 80);

    const __m256i TWO = _mm256_set1_epi32(2);
    const __m256i ONE = _mm256_set1_epi32(1);
    const __m256i GV  = _mm256_set1_epi32(GAP);

    int *A = d2, *B = d1, *C = d0;
    A[0] = 0;
    B[0] = GAP; B[1] = GAP;

    for (int d = 2; d <= 2 * n; d++) {
        int ilo = d - n; if (ilo < 1) ilo = 1;
        int ihi = d - 1; if (ihi > n) ihi = n;
        int cnt = ihi - ilo + 1;
        if (cnt > 0) {
            const char *pa  = ap + (ilo - 1);
            const char *pb  = br + (n - d + ilo);
            const int  *q2m = A  + (ilo - 1);
            const int  *q1m = B  + (ilo - 1);
            const int  *q1  = B  + ilo;
            int        *op  = C  + ilo;
            for (int k = 0; k < cnt; k += 8) {
                __m128i ca = _mm_loadl_epi64((const __m128i *)(pa + k));
                __m128i cb = _mm_loadl_epi64((const __m128i *)(pb + k));
                __m256i m  = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(ca, cb));
                __m256i sc = _mm256_sub_epi32(_mm256_and_si256(m, TWO), ONE);
                __m256i dg = _mm256_add_epi32(
                                 _mm256_loadu_si256((const __m256i *)(q2m + k)), sc);
                __m256i up = _mm256_add_epi32(
                                 _mm256_loadu_si256((const __m256i *)(q1m + k)), GV);
                __m256i lf = _mm256_add_epi32(
                                 _mm256_loadu_si256((const __m256i *)(q1  + k)), GV);
                _mm256_storeu_si256((__m256i *)(op + k),
                    _mm256_max_epi32(dg, _mm256_max_epi32(up, lf)));
            }
        }
        if (d <= n) { int e = -2 * d; C[0] = e; C[d] = e; }
        int *t = A; A = B; B = C; C = t;
    }
    int res = B[n];
    free(ap); free(br); free(d2); free(d1); free(d0);
    return res;
}
#endif /* __AVX2__ */

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
#if defined(__AVX2__)
    if (n < 64)      return nw_scalar(n, a, b);  /* tray too small to wake a worm */
    if (n <= 16000)  return nw_worm16(n, a, b);  /* shallow cups: |score| < 2^15  */
    return nw_worm32(n, a, b);                   /* deep cups                     */
#else
    return nw_scalar(n, a, b);
#endif
}
```

## PREDICTION

PREDICTION: speedup_vs_dp = 12.0

Reasoning stated before any measurement: the reference has a loop-carried dependency on `dp[i][j-1]`, so it is latency-bound at roughly 3 cycles per cell and cannot be auto-vectorized; it also writes a full `4(n+1)²`-byte table, leaving L2 above `n ≈ 350`. The worm has no loop-carried dependency at all — every junction under the body is independent — and settles 16 cells in about 13 instructions, with a working set of `6n + 2n` bytes that stays in L1 up to `n ≈ 5000`. That is ~0.3 cycles/cell against ~3, minus edge-diagonal inefficiency at the two corners and per-call setup, so I expect 8–15× and predict 12.

## MEASUREMENT

**Not measured.** No tools are available in this session — `alignment_bench` and `alignment_contract` could not be called, so I have no number to report and I will not invent one. What I *can* verify by hand, and did, is correctness of the wavefront indexing:

- Cell `(i, j)` sits on diagonal `d = i+j` at slot `i`; its three predecessors are `prev1[i-1]` (up), `prev1[i]` (left), `prev2[i-1]` (diag). Checked against the recurrence directly.
- Read window on `B` is exactly `[max(0,d-1-n), min(n,d-1)]`, which is exactly the valid extent of diagonal `d-1`; read window on `A` is inside the valid extent of `d-2`. No stale or unwritten real cell is ever read.
- Tail lanes past `ihi` write only to slots `> min(n,d)`, which are provably never read again before being overwritten (as `prev1` the next step reads at most slot `min(n,d)`; as `prev2` two steps later, at most `min(n,d+1)-1 ≤ min(n,d)`). Buffers are `calloc`ed and padded by 40/24 elements so every rounded-up vector load and store is in-bounds; `ap`/`br` are padded copies by 80 bytes, so no over-read of the caller's strings.
- `sc = (mask & 2) - 1` gives `+1` on `mask = 0xFFFF` and `-1` on `mask = 0`. Checked both cases.
- Hand-traced `n = 1` (`d = 2`, `ilo = ihi = 1`, result `B[1]`) and the `d ≤ n` edge writes `C[0] = C[d] = -2d`.

This is an argument for correctness, not evidence of speed. The speed claim is unverified.

## VERDICT

The literal reading paid off, and it paid off precisely at the assumption I was told to prefer breaking. Taking "a worm whose *body* curls back across the tray" at face value forces the body to be an anti-diagonal — because that is the only set of lattice junctions a body can occupy simultaneously without any part of it depending on another part — and that single geometric consequence delivers three things the textbook route does not: sixteen pairs judged per drop instead of one, zero loop-carried dependency where the reference has a serial one, and O(n) sand instead of an O(n²) table. The "curl back to test the junction from another face" turned out to be literally the two overlapping unaligned loads of the same previous diagonal — the mechanism's strangest-sounding clause is the one that makes the whole thing cheap.

Two honest limitations, both guarded rather than hand-waved:

1. **Small trays.** Waking the worm costs five allocations and two cord copies; below `n = 64` that dominates. Guarded: `n < 64` routes to `nw_scalar`, a rolling-row DP that is itself still O(n) memory and never builds the `n²` table. I did not ship the risk unaddressed.
2. **Cup depth.** int16 mounds overflow once `2n > 32767`. Guarded: `n > 16000` routes to the int32 worm — the same mechanism, half the lanes, not a different algorithm. This is the two-regime requirement answered in the native's own terms (how deep a cup a mound needs), with runtime recognition and a real path for each regime.

No thread parallelism was added, per instruction and per the metaphor: the native wakes *one* worm, and at benchmark sizes one body of sixteen junctions per drop is already the right grain — OpenMP across diagonals would fight the wavefront's own sequential clock for no gain.

The thing I cannot claim: I do not know the measured speedup. My prediction of 12× is a cycle-count argument, not a result. If the harness reports well under ~5×, the first place I would look is the per-call `malloc`/`calloc`/`memcpy` setup at small-to-mid `n` — the fix stays inside this mechanism (one reusable static scratch arena, or fusing the reversed-cord build into the first few diagonals), not outside it.