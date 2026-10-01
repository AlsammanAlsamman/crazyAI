## MAPPING

**SEED 1** — *fixed row + one-slip row on a shared road*

| World object | Problem object |
|---|---|
| road of lines and dots | the shared index axis `0..n-1` (the DP main diagonal) |
| house | one index position / one alignment column |
| first row of pawns, flush to the marks, never moves | `a`, read at its natural index, never re-indexed |
| second row of pawns | `b`, whose read-index may be displaced |
| carved beast on a crown (lion/bird/elephant/lotus) | the base `A/C/G/T` — a nominal symbol, only `==` is defined on it |
| "no two beasts mistaken even in poor lamplight" | exact byte equality; no substitution matrix, no partial credit |
| horse's-head piece | the *single* gap event; the only move that changes the correspondence |
| "every pawn behind it shuffles one house forward, opening a gap" | inserting one gap at column *k*: suffix of `b` shifts by one — offset `±1` |
| "the horse never rests until it has stood in each doorway once" | the trial set is **all** `k = 0..n-1` |
| "and once more standing nowhere at all" | the `k = ∅` trial: the ungapped diagonal |
| rows set "nose to nose, house against house" | the two rows lie in **opposite senses** along the one road → `b` stored reversed |

**SEED 2** — *lotus petal / red knot, one per flute note*

| World object | Problem object |
|---|---|
| lotus petal in a house | match, contributes `+1` |
| red knot in a house | mismatch, contributes `−1` |
| "cool and easy" petal vs. tied knot | asymmetric cost of the two outcomes (here `+1` vs `−1`) |
| the dark figure's flute, one low note per house | the clock: **exactly one column retired per tick**, no skipping, no doubling |
| "never two houses on one breath" | *this is the assumption I get to violate* — SIMD retires 16 houses per tick |
| red thread from the laughing woman's ball | a shared pool of identical tokens ⇒ mismatches are fungible ⇒ only the *count* matters |

**SEED 3** — *fists weighed, losers flung to the fish*

| World object | Problem object |
|---|---|
| gathering a trial's knots into a fist | reducing one trial's column outcomes to a scalar |
| weighing the fist | `reduce(max)` / accumulate — an O(1) summary |
| "the count is the whole judgment of that trial, nothing else matters" | **no traceback, no table retained**; state per trial is one register |
| knot-counts set beside the road | an array of `n+1` independent candidate scores |
| keeping only the lightest fist | `max` over trials |
| flinging the rest to the flying fish | the discarded trials are never revisited ⇒ trials are **mutually independent** ⇒ no recurrence |

## CHOSEN SEED

**SEED 1.** It is the most literal of the three: every noun in it is already a computational object with no interpretive slack — the road is the index axis, a house is a column, a beast is a base, the fixed row is `a`, and the horse's-head is *the gap*, named as "the one piece permitted to make a string slip." Seeds 2 and 3 describe the *scoring* and the *reduction*, which are generic; Seed 1 describes the **search space**, which is where the algorithm actually lives.

## ASSUMPTION BROKEN

Honest first: **none of the three seeds breaks "both strings are read start to end in the same direction" in its index arithmetic.** All three walk the road forward. I will not manufacture a reversal that isn't there.

But SEED 1 contains a *geometric* fact that does break it, and it turned out to be load-bearing: the two rows are set **"nose to nose, house against house."** Opposing rows in this game face each other, so they are laid out in *opposite senses* along the one shared road. Taken literally, that says: keep `a` forward and keep `b` **reversed**. That is exactly the layout that makes the anti-diagonal sweep work — along an anti-diagonal `i+j=d`, the pair needed at step `i` is `(a[i-1], b[d-i-1])`, whose `b` index *decreases*; with `b` stored reversed, both operands become **two forward contiguous loads**, which is the whole reason 16 houses can be retired per tick. So the direction break is real, it comes from the native's board geometry, and it is the thing that pays.

The bigger assumption SEED 1 + SEED 3 break together: **"the score must be computed by a recurrence over a 2-D table, each cell depending on its neighbours."** The native never builds a table. He runs `n+1` *independent* trials and takes a max. That is not the textbook method and it is not exact — so I did not pretend it was. Instead I used it for what it actually is:

> Every one of the native's trials is a genuine Needleman–Wunsch alignment. So his kept fist is a **certified lower bound** `L ≤ OPT`, obtained in O(n) with no dependencies.

Then the gap accounting closes it. For two length-`n` strings, an alignment with `g` gaps in each string has `n−g` aligned pairs and `2g` gap columns, so

    score ≤ (n − g) − 4g = n − 5g,   and   |i − j| ≤ g everywhere on its path.

With `W = ⌊(n−L)/5⌋` we get `5(W+1) > n−L`, hence any alignment with `g > W` scores `< L` and cannot be optimal. **The optimum therefore lies entirely inside a band of half-width `W`, and the banded DP is exact.** Because `L ≥ n − 2·Hamming ≥ −n`, we always have `W ≤ ⌊2n/5⌋` — the native's mechanism *provably* deletes at least ~36 % of the table on every input, and 98–99 % of it on similar sequences. That is the native's idea doing the work, not being replaced by it.

Full object map: **memory** = the mud-brick floor (three rolling anti-diagonals, `O(n+W)`); **what stays still** = `a` and the road (the fixed row, never re-indexed); **what flows** = the anti-diagonal wavefront; **the processor** = the horse's-head — a 16-lane vector that occupies sixteen doorways at once; **time** = the flute, one note per anti-diagonal (`2n+1` notes total, instead of `n²` cell-visits); **the flying fish** = every cell outside the band, flung out before it is ever computed.

## ARTIFACT

```c
/* ================================================================
 * Needleman-Wunsch global alignment score of two equal-length
 * sequences.  match = +1, mismatch = -1, gap = -2.  EXACT.
 *
 *   int kernel(int n, const char *a, const char *b);
 *
 * Engine (the native's mechanism, literally):
 *   Stage A -- "the horse stands in every doorway once, and once
 *   nowhere at all": evaluate the ungapped diagonal and every
 *   single-slip alignment, in O(n), with no loop-carried dependence.
 *   Every trial is a real NW alignment, so the kept fist is a
 *   certified lower bound L <= OPT.
 *
 *   Stage B -- gap accounting: any alignment with g gaps per string
 *   scores <= n - 5g and never leaves |i-j| <= g.  With
 *   W = floor((n-L)/5) we have 5(W+1) > n-L, so every alignment with
 *   g > W scores < L and is flung to the fish.  The band of
 *   half-width W is therefore exact.  L >= -n always, so W <= 2n/5:
 *   the band is never wider than ~64% of the table, for ANY input.
 *
 *   Stage C -- the flute: fill the band along anti-diagonals, where
 *   all cells are independent, 16 houses per note (int16 AVX2).
 *   'b' is stored reversed ("nose to nose") so both character
 *   operands are forward contiguous loads.
 *
 * Guards for every risk this mechanism's own verdict names:
 *   - int16 range: real in-band cells are >= -(n + 2W + 8) >= -1.8n-8;
 *     sentinel is -30000.  int16 path only for n <= 12000; an
 *     otherwise identical int32 path covers larger n.
 *   - no AVX2 (-march=native on an older host): scalar banded DP.
 *   - tiny n (vector lanes wasted, 2W+1 < 16): scalar banded DP.
 *   - W == 0: return the native's diagonal answer directly, O(n),
 *     no DP at all.
 *   - allocation failure: degrade to the certified bound, never crash.
 * ================================================================ */

#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---------------- scratch (per thread, grow-only) ---------------- */
static __thread unsigned char *nb_raw = 0;
static __thread size_t         nb_cap = 0;

static unsigned char *nb_scratch(size_t need)
{
    if (need > nb_cap) {
        unsigned char *p = (unsigned char *)realloc(nb_raw, need);
        if (!p) return 0;
        nb_raw = p;
        nb_cap = need;
    }
    return nb_raw;
}

#define NB_AL(p) ((void *)(((size_t)(p) + 63u) & ~(size_t)63u))

/* ---------------- scalar banded DP (fallback / tiny n) ----------- */
static int nb_scalar(int n, const char *a, const char *b, int W,
                     int *prev, int *cur)
{
    const int NEG = -(1 << 28);
    int i, j, jhi;

    for (j = 0; j <= n + 1; ++j) { prev[j] = NEG; cur[j] = NEG; }

    jhi = (W < n) ? W : n;                 /* row 0 inside the band   */
    for (j = 0; j <= jhi; ++j) prev[j] = -2 * j;
    if (jhi + 1 <= n + 1) prev[jhi + 1] = NEG;

    for (i = 1; i <= n; ++i) {
        int jlo = i - W, jh = i + W;
        int *t;
        if (jlo < 0) jlo = 0;
        if (jh > n)  jh  = n;
        if (jlo > 0) cur[jlo - 1] = NEG;
        for (j = jlo; j <= jh; ++j) {
            int best;
            if (j == 0) {
                best = -2 * i;
            } else {
                int s = (a[i - 1] == b[j - 1]) ? 1 : -1;
                int v = prev[j - 1] + s;
                int u = prev[j] - 2;
                int l = cur[j - 1] - 2;
                best = v;
                if (u > best) best = u;
                if (l > best) best = l;
            }
            cur[j] = best;
        }
        if (jh + 1 <= n + 1) cur[jh + 1] = NEG;
        t = prev; prev = cur; cur = t;
    }
    return prev[n];
}

/* ---------------- anti-diagonal SIMD band sweep ------------------ */
#if defined(__AVX2__)
/* One body, instantiated at two widths, so the two paths cannot drift. */
#define NB_AVX_DP(FN, TY, SFX, CVT, CHLD, LANES, NEGV)                        \
static int FN(int n, const unsigned char *ca, const unsigned char *cbr,       \
              int W, TY *b0, TY *b1, TY *b2)                                  \
{                                                                             \
    TY *db[3];                                                                \
    const __m256i vone = _mm256_set1_##SFX(1);                                \
    const __m256i vtwo = _mm256_set1_##SFX(2);                                \
    int d, i, span = n + LANES + 4;                                           \
    db[0] = b0; db[1] = b1; db[2] = b2;                                       \
    for (i = -1; i < span; ++i) {                                             \
        db[0][i] = NEGV; db[1][i] = NEGV; db[2][i] = NEGV;                    \
    }                                                                         \
    db[0][0] = 0;                              /* H(0,0) = 0, d = 0 */        \
    for (d = 1; d <= 2 * n; ++d) {                                            \
        TY       *cc = db[d % 3];                                             \
        const TY *p1 = db[(d + 2) % 3];        /* anti-diagonal d-1 */        \
        const TY *p2 = db[(d + 1) % 3];        /* anti-diagonal d-2 */        \
        int lo = d - n, hi = (d < n) ? d : n, t;                              \
        if (lo < 0) lo = 0;                                                   \
        t = (d - W + 1) >> 1; if (t > lo) lo = t;   /* ceil((d-W)/2) */       \
        t = (d + W) >> 1;     if (t < hi) hi = t;   /* floor((d+W)/2) */      \
        {                                                                     \
            const unsigned char *pa = ca  + lo;                               \
            const unsigned char *pb = cbr + (lo + n - d);                     \
            for (i = lo; i <= hi; i += LANES, pa += LANES, pb += LANES) {     \
                __m256i xa = CVT(CHLD((const __m128i *)pa));                  \
                __m256i xb = CVT(CHLD((const __m128i *)pb));                  \
                __m256i eq = _mm256_cmpeq_##SFX(xa, xb);                      \
                __m256i s  = _mm256_sub_##SFX(                                \
                                 _mm256_slli_##SFX(                           \
                                     _mm256_and_si256(eq, vone), 1), vone);   \
                __m256i dg = _mm256_loadu_si256((const __m256i *)(p2 + i-1)); \
                __m256i up = _mm256_loadu_si256((const __m256i *)(p1 + i-1)); \
                __m256i lf = _mm256_loadu_si256((const __m256i *)(p1 + i  )); \
                __m256i v  = _mm256_max_##SFX(                                \
                                 _mm256_add_##SFX(dg, s),                     \
                                 _mm256_sub_##SFX(                            \
                                     _mm256_max_##SFX(up, lf), vtwo));        \
                _mm256_storeu_si256((__m256i *)(cc + i), v);                  \
            }                                                                 \
        }                                                                     \
        cc[lo - 1] = NEGV;                    /* the flying fish */           \
        cc[hi + 1] = NEGV;                                                    \
    }                                                                         \
    return (int)db[(2 * n) % 3][n];                                           \
}

NB_AVX_DP(nb_dp16, short, epi16, _mm256_cvtepi8_epi16, _mm_loadu_si128, 16,
          (short)-30000)
NB_AVX_DP(nb_dp32, int,   epi32, _mm256_cvtepi8_epi32, _mm_loadl_epi64,  8,
          -(1 << 28))
#endif /* __AVX2__ */

/* ============================== kernel ========================== */
int kernel(int n, const char *a, const char *b)
{
    size_t nz, sz_ch, sz_r, sz_dp, need;
    unsigned char *raw, *cz, *ca, *cbr;
    int *R1, *R2;
    void *D0, *D1, *D2;
    int pre, bestpm, S0, L, W, i, k, r;

    if (n <= 0) return 0;

    nz    = (size_t)n;
    sz_ch = nz + 96;                            /* ca, cbr           */
    sz_r  = (nz + 8) * sizeof(int);             /* R1, R2            */
    sz_dp = (nz + 40) * sizeof(int);            /* each DP buffer    */
    need  = 8u * 64u + 2u * sz_ch + 2u * sz_r + 3u * sz_dp;

    raw = nb_scratch(need);

    /* ---------- Stage A: the native's trials, O(n), no recurrence
       across trials.  R1[t] and R2[t] are the two slip directions.  */
    if (!raw) {                                 /* degrade, don't die */
        int m = 0;
        for (i = 0; i < n; ++i) m += (a[i] == b[i]);
        return 2 * m - n;                       /* diagonal only      */
    }

    cz  = (unsigned char *)NB_AL(raw);
    ca  = cz;  cz = (unsigned char *)NB_AL(cz + sz_ch);
    cbr = cz;  cz = (unsigned char *)NB_AL(cz + sz_ch);
    R1  = (int *)cz; cz = (unsigned char *)NB_AL(cz + sz_r);
    R2  = (int *)cz; cz = (unsigned char *)NB_AL(cz + sz_r);
    D0  = cz;  cz = (unsigned char *)NB_AL(cz + sz_dp);
    D1  = cz;  cz = (unsigned char *)NB_AL(cz + sz_dp);
    D2  = cz;

    R1[n] = 0;
    R2[n] = 0;
    for (i = n - 1; i >= 1; --i) {
        R1[i] = R1[i + 1] + (a[i]     == b[i - 1]);  /* slip b forward */
        R2[i] = R2[i + 1] + (a[i - 1] == b[i]    );  /* slip a forward */
    }

    pre    = 0;
    bestpm = -(1 << 28);
    for (k = 0; k < n; ++k) {                   /* horse in house k   */
        int m1 = R1[k + 1], m2 = R2[k + 1];
        int mm = (m1 > m2) ? m1 : m2;
        int v  = pre + mm;
        if (v > bestpm) bestpm = v;
        pre += (a[k] == b[k]);
    }
    S0 = 2 * pre - n;                           /* horse nowhere      */
    L  = S0;
    {   int slip = 2 * bestpm - (n - 1) - 4;    /* two gap columns    */
        if (slip > L) L = slip;
    }

    /* ---------- Stage B: the band the certificate buys ------------ */
    W = (n - L) / 5;                            /* n - L >= 0 always  */
    if (W < 0) W = 0;
    if (W > n) W = n;
    if (W == 0) return S0;                      /* proven: no gaps    */

    /* ---------- Stage C: fill the band ---------------------------- */
#if defined(__AVX2__)
    if (n >= 32) {
        ca[0] = 0x01;                           /* non-ACGT sentinel  */
        memcpy(ca + 1, a, nz);
        memset(ca + 1 + nz, 0x01, sz_ch - 1 - nz);
        for (i = 0; i < n; ++i) cbr[i] = (unsigned char)b[n - 1 - i];
        memset(cbr + nz, 0x02, sz_ch - nz);     /* differs from 0x01  */

        if (n <= 12000)
            r = nb_dp16(n, ca, cbr, W,
                        (short *)D0 + 1, (short *)D1 + 1, (short *)D2 + 1);
        else
            r = nb_dp32(n, ca, cbr, W,
                        (int *)D0 + 1, (int *)D1 + 1, (int *)D2 + 1);
        return (r > L) ? r : L;
    }
#endif
    r = nb_scalar(n, a, b, W, (int *)D0, (int *)D1);
    return (r > L) ? r : L;
}
```

PREDICTION: speedup_vs_dp = 12

Reasoning behind the number, stated before any measurement: two multiplicative factors. (i) **Cells deleted by the native's certificate.** On near-random DNA, `Hamming ≈ 0.75n`, the best of the `~2n` trials lands near `L ≈ −0.4n`, so `W ≈ 0.28n` and the band holds `≈0.48n²` cells — a 2.1× cut. On sequences with 1–10 % divergence, `W` falls to `0.004n–0.04n` and the cut is 12–120×. (ii) **Constant factor from the flute break.** The reference inner loop is serially dependent through `H[i][j-1]` (≈2–3 cycles/cell); the anti-diagonal sweep has no loop-carried dependence and retires 16 cells per ~5 cycles (≈0.3 cycles/cell), ≈7×. Product on the pessimistic (random) input ≈ 15; I shade to **12** for per-anti-diagonal loop overhead (`2n+1` bookkeeping steps) and for lane waste when `2W+1` is not a multiple of 16.

## MEASUREMENT

**Not measured. I must report this as a failure of the run, not soften it.** No tools were available in this session — `alignment_bench` and `alignment_contract` were both listed but neither is callable here, so the artifact above is **unverified against the reference aligner and untimed**, and the four allowed improvement rounds never happened. Everything in the PREDICTION section is analysis, not data. Treat the speedup figure as a hypothesis with a stated derivation, and treat exactness as *argued* (the `score ≤ n − 5g` bound plus `|i−j| ≤ g`) rather than *checked*.

What I did instead of measuring, since it was all that was available: hand-executed the kernel on cases chosen to break it.

| case | native `L` | `W` | expected NW | path taken |
|---|---|---|---|---|
| `A` vs `A` | 1 | 0 | +1 | Stage A only |
| `A` vs `T` | −1 | 0 | −1 | Stage A only |
| `AC` vs `CA` | −2 | 0 | −2 (slip costs −3) | Stage A only |
| `ACGTACGTAC` vs `TACGTACGTA` | **+5** (slip at house 0) | 1 | +5 | band DP, W=1 |
| identical, any `n` | `n` | 0 | `n` | Stage A only, O(n) |

The fourth row is the one that matters: the ungapped diagonal scores −10 there, the native's single slip finds +5, and `n − 5g = 0 < 5` for `g = 2` proves no wider band can beat it. Sentinel behaviour at `i = 0` and `j = 0` was traced by hand and the generic recurrence reproduces `H(0,d) = H(d,0) = −2d` without special-casing, provided the `NEG` writes at `lo−1` and `hi+1` are in place — they are.

The residual risk this leaves, honestly: the band-index arithmetic (`lo`, `hi`, the `p2 + i − 1` offset, the rolling `db[d%3]` rotation) is exactly the kind of code that needs a differential test against the reference DP over thousands of random pairs, and it has not had one.

## VERDICT

The native's mechanism is **not** a correct alignment scorer, and I will not claim it is. It allows one slip, ignores the gap penalty, and drops a dangling character; it is an approximation that undershoots. Two structural facts rescue it and make it load-bearing rather than decorative:

1. Every trial he runs is a **real** Needleman–Wunsch alignment, so his kept fist is a *certificate*, not an estimate — `L ≤ OPT`, computable in O(n) with zero dependencies.
2. Under this particular scoring (`gap = −2`, so a gap pair costs 4 and must repair ≥3 mismatches to pay), the certificate converts directly into a **proof of band width**: `W = ⌊(n−L)/5⌋ ≤ ⌊2n/5⌋`. The native's cheap, dependency-free, embarrassingly-parallel search buys the expensive DP a provably safe pruning. That is genuinely not the textbook route to a banded aligner (textbook banding either guesses a width or iterates Ukkonen-style), and it fell out of taking "the one permitted slip" literally.

The thing that surprised me was the **direction break** arriving from the board's *furniture* rather than its arithmetic — "nose to nose, house against house" is the reversed-`b` buffer, and reversed-`b` is precisely what turns the anti-diagonal character fetch into two forward contiguous loads. I would not have reached for it from the DP recurrence.

**Named risk, and how it is guarded (nothing shipped unaddressed):**

- **Near-random inputs are the weak case.** `W` only falls to `≈0.28n`, so the cell cut is ~2.1× and nearly all the speedup has to come from the SIMD constant factor. *Guard:* the bound `L ≥ −n ⇒ W ≤ 2n/5` is unconditional, so the band never exceeds ~64 % of the table and the kernel can never do *more* work than the full DP — the mechanism degrades to "vectorised full-ish DP," never to a loss. If a measurement shows it still loses on random inputs, the honest fix is to drop Stage B's band (set `W = n`) and keep only the anti-diagonal sweep; the code path is already the same.
- **int16 overflow.** Range is `[−(n+2W+8), n]`; guarded at `n ≤ 12000` with sentinel `−30000` (≈8× margin), with a structurally identical int32 path above it.
- **No AVX2 under `-march=native`.** Scalar banded DP fallback.
- **Lane waste at small `n` / narrow bands.** Scalar path below `n = 32`; above it a narrow band means `W` is tiny, where the 12–120× cell cut swamps any lane waste.
- **`W = 0`.** Returned directly from Stage A in O(n) with no DP — this is the case where the native's answer *is* the exact answer, and it is provably so (`L ≥ n−4` forces `g = 0`).
- **Allocation failure.** Returns the diagonal score rather than dereferencing null. This is a *wrong* answer, not a crash — flagged here deliberately; if the harness prefers a hard failure, that branch should abort instead.

Confidence: high on exactness of the argument, **unverified** on the implementation, and the speedup number is an unmeasured prediction. Run `alignment_contract` and `alignment_bench` against this before trusting either.