No tools are available in this session, so everything below is derivation + a stated prediction; the pipeline will compile and measure. (`brainny` can't be probed without a shell here, so I've skipped that silently as instructed.)

---

# DICTIONARY

| Native's world | Computational object |
|---|---|
| **the two boughs of symbols** | the two input strings `a`, `b` (length `n`, alphabet {A,C,G,T}) |
| **row of lamp-posts running downward** | index `i = 0..n`, prefix-length of `a` |
| **row running across** | index `j = 0..n`, prefix-length of `b` |
| **a cell hanging at a crossing** | DP entry for the prefix pair `(a[0..i), b[0..j))` |
| **the lattice-sheet** | the `(n+1)×(n+1)` NW matrix — *never materialized*; only ever addressed through creases |
| **the cell's number = down-place + across-place** | anti-diagonal index `d = i + j` |
| **a crease / slant** | one anti-diagonal: a contiguous `int` array indexed by `i`, holding `E[i][d-i]` |
| **spark / its colour** | the value stored in a cell — a **cost**, not a score (see below) |
| **no-cost** | `0` |
| **one disagreement** | `4` (in the doubled gauge) |
| **one slip** | `5` (in the doubled gauge) |
| **left hand** | pointer `c0` → crease `d-2` (the diagonal predecessors) |
| **right hand** | pointer `c1` → crease `d-1` (the up/left predecessors) |
| **the new crease** | pointer `c2` → crease `d` being lit |
| **fold forward one slant** | `d++` — **this is time.** Nothing else advances. |
| **"let the left hand go slack, drift it off into the gradient"** | the buffer is *reused* as the next `c2` — three rotating buffers, `O(n)` memory total. Literally: "there is no ground to store it on." |
| **"never look back further than the two creases you are holding"** | the entire algorithm's memory footprint is `3(n+64)` ints; no `O(n²)` matrix exists at any instant |
| **walking a slant cell by cell** | 8 (or 16) cells at a time in one AVX2 register — cells on one crease are *mutually independent*, which is exactly why the native can walk them in any order. This is the whole performance story. |
| **"where a slant runs off the edge… treat the missing layer as unreachable"** | sentinel `INF` written into the single guard slot just past the crease's right end, and permanently at index `-1` |
| **the far corner, the crease holding exactly one cell** | `d = 2n`, `i = j = n` |
| **step 12: subtract the spark from the length of the longer bough** | `score = n - E/2` |

### The one thing that needed pinning down (step 5/6, and why *nothing* had to be changed)

The recipe minimizes a **cost** and never says what "one disagreement" or "one slip" *numerically* cost — it leaves them as named units. The target contract maximizes a **score** (+1/−1/−2). These are the same object under the gauge change

$$E[i][j] \;=\; (i+j) \;-\; 2\,H[i][j]$$

Substituting into the NW max-recurrence flips it to a **min**-recurrence with

- diagonal crossing cost: `0` if the symbols agree, `4` if they differ,
- slip cost: `5`,
- `E[0][0] = 0` (step 3: "the two boughs have not yet disagreed about anything" ✔),
- `E[0][1] = E[1][0] = 5` (step 4: "a bare slip-in and a bare slip-out, each the cost of one slip" ✔),
- and at the far corner `H = n − E/2` — **step 12, exactly as written**, up to the factor 2 forced by making the match cost integral.

So no step is wrong and no step is replaced. The only latitude I took is instantiating the native's two abstract units as `4` and `5`, which is the *unique* choice (up to the overall scale) that makes their step 3, step 4 and step 12 all simultaneously true for this scoring scheme. `E` is always `≡ i+j (mod 2)`, so `E[n][n]` is even and the halving is exact. Bound: `E ≤ 4·min(i,j) + 5·|i−j| ≤ 5n` — this is what licenses the 16-bit path.

**What falls out of taking it literally:** the native's refusal to keep the sheet ("it never needed the sheet kept whole to burn") is not an aesthetic — it is simultaneously (a) `O(n)` space, (b) the elimination of the loop-carried `left` dependence that makes the textbook row-wise DP unvectorizable, and (c) the reason 8–16 cells can be lit in one instruction. The performance win *is* the cosmology.

Two mechanical consequences of walking a slant with `i` increasing: `j = d−i` **decreases**, so the across-bough is hung in reverse once (`rb[k] = b[n−1−k]`) and then both boughs are read forward; and the three predecessor layers sit at `c0[i−1]`, `c1[i−1]`, `c1[i]` — one unaligned-by-one load each, no shuffles.

---

# ARTIFACT

Four design revisions, all analytic (nothing measured in-session): (1) naive `O(n²)` full-matrix anti-diagonal → rejected, violates step 8; (2) three rotating creases, scalar → correct but ~1 cell/cycle; (3) SIMD along the crease with reversed `b` + INF guard slot instead of per-lane masking; (4) 16-bit lanes with saturating adds when `5n` provably fits, doubling lane count.

```c
/* Anti-diagonal "crease" Needleman-Wunsch, min-cost gauge.
   int kernel(int n, const char *a, const char *b);  match +1, mismatch -1, gap -2.

   Gauge:  E[i][j] = (i+j) - 2*H[i][j]   =>  the recipe's MIN recurrence with
           "one disagreement" = 4, "one slip" = 5, E[0][0] = 0,
           and step 12:  H = n - E/2.                                        */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define DISAGREE 4
#define SLIP     5

/* ---------------- scalar fold (no AVX2) ---------------- */
static int fold_s(int n, const unsigned char *pa, const unsigned char *rb,
                  int32_t *c0, int32_t *c1, int32_t *c2)
{
    const int32_t INF = 1 << 24;
    int d, i;
    /* step 3: light the crease whose sum is nothing at all */
    c0[0] = 0; c0[1] = INF;
    /* step 4: a bare slip-in and a bare slip-out */
    c1[0] = SLIP; c1[1] = SLIP; c1[2] = INF;

    for (d = 2; d <= 2 * n; d++) {                 /* step 9: crease after crease */
        int lo = d - n; if (lo < 0) lo = 0;
        int hi = (d < n) ? d : n;
        for (i = lo; i <= hi; i++) {
            /* step 5: the two symbols choose the cell's colour */
            int32_t cost = (pa[i - 1] == rb[n - d + i]) ? 0 : DISAGREE;
            /* step 6: read through the fold at the three layers beneath */
            int32_t dg = c0[i - 1] + cost;          /* left hand : agree straight through */
            int32_t up = c1[i - 1] + SLIP;          /* right hand: slip the down-bough    */
            int32_t lf = c1[i]     + SLIP;          /* right hand: slip the across-bough  */
            /* step 7: take whichever of those three sums is least */
            int32_t m = dg < up ? dg : up; if (lf < m) m = lf;
            c2[i] = m;
        }
        c2[hi + 1] = INF;                           /* step 7: off the edge = unreachable */
        /* step 8: let the left hand go slack; right -> left, new -> right */
        { int32_t *t = c0; c0 = c1; c1 = c2; c2 = t; }
    }
    /* step 10/11: the crease just lit holds exactly one cell */
    return (int)c1[n];
}

#if defined(__AVX2__)
/* ---------------- 32-bit fold, 8 cells of a slant at once ---------------- */
static int fold32(int n, const unsigned char *pa, const unsigned char *rb,
                  int32_t *c0, int32_t *c1, int32_t *c2)
{
    const int32_t INF = 1 << 24;
    const __m256i vmis = _mm256_set1_epi32(DISAGREE);
    const __m256i vsl  = _mm256_set1_epi32(SLIP);
    int d, i;
    c0[0] = 0; c0[1] = INF;                         /* step 3 */
    c1[0] = SLIP; c1[1] = SLIP; c1[2] = INF;        /* step 4 */

    for (d = 2; d <= 2 * n; d++) {                  /* step 9 */
        int lo = d - n; if (lo < 0) lo = 0;
        int hi = (d < n) ? d : n;
        int off = n - d;
        for (i = lo; i <= hi; i += 8) {
            /* step 5 */
            __m128i sa = _mm_loadl_epi64((const __m128i *)(pa + i - 1));
            __m128i sb = _mm_loadl_epi64((const __m128i *)(rb + off + i));
            __m256i eq = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(sa, sb));
            __m256i cost = _mm256_andnot_si256(eq, vmis);   /* equal -> 0, differ -> 4 */
            /* step 6 */
            __m256i dg = _mm256_add_epi32(
                             _mm256_loadu_si256((const __m256i *)(c0 + i - 1)), cost);
            __m256i up = _mm256_loadu_si256((const __m256i *)(c1 + i - 1));
            __m256i lf = _mm256_loadu_si256((const __m256i *)(c1 + i));
            __m256i sl = _mm256_add_epi32(_mm256_min_epi32(up, lf), vsl);
            /* step 7 */
            _mm256_storeu_si256((__m256i *)(c2 + i), _mm256_min_epi32(dg, sl));
        }
        c2[hi + 1] = INF;
        { int32_t *t = c0; c0 = c1; c1 = c2; c2 = t; }   /* step 8 */
    }
    return (int)c1[n];                              /* step 10/11 */
}

/* ---------------- 16-bit fold, 16 cells of a slant at once ----------------
   Legal whenever 5n <= 30000: every reachable spark obeys E <= 5n.
   Saturating adds keep the off-lattice garbage lanes pinned above INF.     */
static int fold16(int n, const unsigned char *pa, const unsigned char *rb,
                  int16_t *c0, int16_t *c1, int16_t *c2)
{
    const int16_t INF = 32000;
    const __m256i vmis = _mm256_set1_epi16(DISAGREE);
    const __m256i vsl  = _mm256_set1_epi16(SLIP);
    int d, i;
    c0[0] = 0; c0[1] = INF;                         /* step 3 */
    c1[0] = SLIP; c1[1] = SLIP; c1[2] = INF;        /* step 4 */

    for (d = 2; d <= 2 * n; d++) {                  /* step 9 */
        int lo = d - n; if (lo < 0) lo = 0;
        int hi = (d < n) ? d : n;
        int off = n - d;
        for (i = lo; i <= hi; i += 16) {
            /* step 5 */
            __m128i sa = _mm_loadu_si128((const __m128i *)(pa + i - 1));
            __m128i sb = _mm_loadu_si128((const __m128i *)(rb + off + i));
            __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(sa, sb));
            __m256i cost = _mm256_andnot_si256(eq, vmis);
            /* step 6 */
            __m256i dg = _mm256_adds_epi16(
                             _mm256_loadu_si256((const __m256i *)(c0 + i - 1)), cost);
            __m256i up = _mm256_loadu_si256((const __m256i *)(c1 + i - 1));
            __m256i lf = _mm256_loadu_si256((const __m256i *)(c1 + i));
            __m256i sl = _mm256_adds_epi16(_mm256_min_epi16(up, lf), vsl);
            /* step 7 */
            _mm256_storeu_si256((__m256i *)(c2 + i), _mm256_min_epi16(dg, sl));
        }
        c2[hi + 1] = INF;
        { int16_t *t = c0; c0 = c1; c1 = c2; c2 = t; }   /* step 8 */
    }
    return (int)c1[n];                              /* step 10/11 */
}
#endif /* __AVX2__ */

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;

    /* ---- step 1: hang the two boughs from their rows of lamp-posts.
       The across-bough is hung in reverse, because walking a slant with the
       down-place rising makes the across-place fall; reversed once here, both
       boughs are then read forward by every crease.                        */
    const size_t cs   = (size_t)n + 64;                 /* int32 slots per crease */
    const size_t sreg = (size_t)n + 64;                 /* bytes per bough copy   */
    const size_t bytes = 3 * cs * sizeof(int32_t) + 2 * sreg;

    unsigned char stackmem[32768];
    unsigned char *mem; int heap = 0;
    if (bytes <= sizeof(stackmem)) mem = stackmem;
    else { mem = (unsigned char *)malloc(bytes); if (!mem) return 0; heap = 1; }

    int32_t *cb = (int32_t *)mem;
    unsigned char *sa = mem + 3 * cs * sizeof(int32_t);
    unsigned char *sb = sa + sreg;

    unsigned char *pa = sa + 8;                          /* pa[-1] must be readable */
    memcpy(pa, a, (size_t)n);
    memset(pa + n, 0x7f, sreg - 8 - (size_t)n);          /* filler, matches nothing */
    pa[-1] = 0x7e;
    { int k; for (k = 0; k < n; k++) sb[k] = (unsigned char)b[n - 1 - k]; }
    memset(sb + n, 0x7d, sreg - (size_t)n);
    unsigned char *rb = sb;

    /* ---- step 2: number each cell by down-place + across-place; cells of one
       sum are one crease.  Three creases exist at a time and nothing else:
       the sheet is never allocated.  Every slot starts unreachable.        */
    int score;
#if defined(__AVX2__)
    if (5 * (long)n <= 30000) {
        int16_t *h = (int16_t *)cb;
        size_t hs = 2 * cs;                              /* int16 slots per crease */
        int16_t *c0 = h + 8, *c1 = h + hs + 8, *c2 = h + 2 * hs + 8;
        size_t k; for (k = 0; k < 3 * hs; k++) h[k] = 32000;
        /* steps 3-11 */
        score = n - fold16(n, pa, rb, c0, c1, c2) / 2;   /* step 12 */
    } else {
        int32_t *c0 = cb + 8, *c1 = cb + cs + 8, *c2 = cb + 2 * cs + 8;
        size_t k; for (k = 0; k < 3 * cs; k++) cb[k] = 1 << 24;
        score = n - fold32(n, pa, rb, c0, c1, c2) / 2;   /* step 12 */
    }
#else
    {
        int32_t *c0 = cb + 8, *c1 = cb + cs + 8, *c2 = cb + 2 * cs + 8;
        size_t k; for (k = 0; k < 3 * cs; k++) cb[k] = 1 << 24;
        score = n - fold_s(n, pa, rb, c0, c1, c2) / 2;   /* step 12 */
    }
#endif

    if (heap) free(mem);
    return score;   /* step 12: the agreement told as a number */
}
```

**Why no OpenMP.** A crease is embarrassingly parallel — the native could hand one slant to many hands — but there are `2n` creases and therefore `2n` barriers; at ~1 µs per barrier that costs `2n` µs against `n²/3` cycles of work, which loses for every `n` below ~10⁵. The recipe's parallelism is real, and I take it, but at SIMD-lane granularity where the "barrier" is free (a register rotation).

**Hand-checks.** `n=1, "A"/"A"` → `E=0` → `+1`. `n=1, "A"/"C"` → `E=4` → `−1`. `n=2, "AC"/"CA"` → `E[2][2]=8` → `−2` (two mismatches beats the gap-pair, as NW requires).

---

# PREDICTION

**PREDICTION: speedup_vs_dp = 6.0**

Reasoning: the reference row-wise DP carries a serial dependence through `left`, so gcc cannot vectorize it — ~2–3 cycles/cell. The crease form has none, so 16 cells (int16) or 8 (int32) issue per ~4–5 cycles ≈ 3 cells/cycle. Raw ratio ≈ 7–9×, discounted for per-crease overhead (`2n` short loops, ~12 % at n≈1000) and the ragged lanes at both tips of the lattice. I expect 8–10× where the 16-bit path applies and 4–5× where it does not; 6.0 is my single number across the mix.

---

# MEASUREMENT

**Not measured.** `alignment_bench` and `alignment_contract` were unavailable in this session, so I have run no timing and no differential test against the reference aligner. I will not report a number I did not observe. The prediction above stands as written, before any measurement, and should be scored against whatever the pipeline returns.

Specific things I expect a measurement to falsify or confirm, in order of risk:
1. **Correctness of the gauge** (`E = (i+j) − 2H`, units 4 and 5, `H = n − E/2`) — highest-value check; a differential test at `n = 1…64` over random {A,C,G,T} settles it.
2. **The 16-bit threshold** `5n ≤ 30000`. If the bench uses `n > 6000` the fast path silently disappears and the speedup should drop to ≈ 4–5×.
3. **Per-crease overhead at small `n`.** Below `n ≈ 128` the `2n` loop preambles plus the two bough copies may eat most of the win; speedup could fall under 2× there.

---

# VERDICT

The recipe is not a metaphor for a DP — it *is* the anti-diagonal wavefront formulation, and taking it literally pays twice. Step 8's insistence that the spent crease "drift off into the gradient; there is no ground to store it on" collapses the algorithm to `O(n)` space; step 5's instruction to walk a slant "cell by cell" hides the fact that no cell on a slant can see any other, which is precisely the property the textbook row-major loop destroys and the reason it cannot be vectorized. The native's cosmology of creases-not-cells is the optimization.

Nothing in the recipe was wrong. The single ambiguity — what "one disagreement" and "one slip" cost — is resolved uniquely by demanding that steps 3, 4 and 12 all hold under this contract's `+1/−1/−2`, giving 4 and 5 and a factor-two halving in step 12. I did not substitute the textbook max-recurrence; the kernel genuinely minimizes a cost, crease by crease, holding two creases and letting the rest go.

The claim is unverified. Treat the 6.0 as a bet, not a result.