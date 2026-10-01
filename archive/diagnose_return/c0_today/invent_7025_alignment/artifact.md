# MAPPING

## SEED 1 — "A fixed row of beast-pawns and a second row that the horse's-head piece may slip forward at exactly one house along the shared road of lines and dots."

| World object | Problem object |
|---|---|
| The road of lines and dots along the wall | The shared index axis of both sequences — concretely the **lane index `i` of an anti-diagonal**, `j = k − i` |
| Furrow 1, "flush against the marks, never moves again" | `a`, read at a fixed offset (`a[i−1]`), base pointer never re-based |
| Furrow 2, the row that can be shuffled | `b`, but stored **reversed** so that a fixed anti-diagonal index reads it contiguously (`brev[n−k+i]`) |
| Cone-pawn / spool-pawn with a carved beast on its crown | One base character; the beast is the whole meaning ⇒ a single-byte equality test, nothing ordinal |
| "No two beasts mistaken even in poor lamplight" | `_mm_cmpeq_epi8` — exact byte equality, 16 crowns inspected in one glance |
| The horse's head, *the only piece permitted to make a string slip* | The **one-lane shift** of the previous wavefront register: `D1[i−1]` (slipped) vs `D1[i]` (unslipped). Exactly one house of slip, no more |
| "Every pawn behind it shuffles one house forward, opening a gap" | An indel: the alignment path leaves diagonal `d = i−j` for `d±1` |
| "Tried both unslipped and slipped at every possible house in turn" | Both shifted reads are taken at **every** lane, **every** tick |
| "Once more standing nowhere at all" | The no-gap option: `D0[i−1] + s(a,b)` |

**Silent assumption broken:** *"a slip (gap) can only be discovered by having already compared the position before it."* In the native's picture the slip is placed **before** the walk begins — it is a property of the whole row, not something learned cell-by-cell. Translated, the gap becomes a *register lane offset chosen up front*, and all offsets are evaluated concurrently. Secondarily it breaks *"one pair of positions is judged at a time"*: one flute note judges 16 houses.

## SEED 2 — "Matching beasts are marked with a lotus petal and disagreeing beasts with a red knot, house by house, paced one count per flute note."

| World object | Problem object |
|---|---|
| Lotus petal (cool, easy) | `+1` |
| Red knot from the laughing woman's ball | `−1` |
| The flute's one low note per house | The **wavefront clock**: one tick per anti-diagonal `k = i+j`, `k = 0 … 2n`. This is the only serial axis |
| "Never count two houses on one breath or skip one" | The tick is the sole dependency carrier; within a tick every house is independent |
| Walking the two furrows *together* | Both operand loads advance by +1 per lane ⇒ contiguous `loadu`, no gather |

**Silent assumption broken:** *"every cell depends on the ones above, to the left, and diagonally above-left, computed in that order."* The flute imposes a different total order — by `i+j`, not by `i` then `j`. The three dependencies survive, but they are re-indexed as *(this tick − 1, slipped)*, *(this tick − 1, unslipped)*, *(this tick − 2, slipped)*, which makes an entire tick data-parallel.

## SEED 3 — "Every trial's knots are gathered into a fist and weighed, and all fists but the lightest are flung out to the passing fish."

| World object | Problem object |
|---|---|
| A fist | The running score of one diagonal / one offset |
| Weighing the fist; "the count is the whole judgment, nothing else matters" | Only a scalar survives per lane — no traceback, no stored table |
| Keeping the lightest fist | `max` reduction over the three candidates |
| **Flinging the other fists to the fish, never seen again** | **Pruning whole diagonals**: any offset proved incapable of holding the optimum is never allocated, never touched |
| The very first fist (the "standing nowhere" trial) | The Hamming distance `H₀` — and it is this first weighing that *certifies how far the horse can ever need to roam* |

**Silent assumption broken:** *"the whole grid of every position against every other must be filled in."* Most fists are flung before the walk.

## Does any seed break "both strings are read start to end in the same direction"?

**No — plainly, none of the three does.** The native walks both furrows "together house by house" in one direction, from the wall's near end to its far end. I say this rather than pretend: the reversal of `b` in my artifact is a *storage* trick for contiguity, not a reversal of the reading order, and I do not claim it as a break of that assumption. So I fall back, as instructed, to the most literal seed.

---

# CHOSEN SEED

**SEED 1** — the fixed row, the slipping row, and the horse that may slip by exactly one house.

It is the most literal (every noun has a register or a pointer), and it is the furthest from the textbook: Needleman–Wunsch discovers gaps by row-major induction; the native *places* the gap first and then walks. The other two seeds are not discarded — SEED 2 supplies the clock and SEED 3 supplies the pruning — but SEED 1 is the load-bearing one.

### The honest problem with the literal reading, and the literal repair

Taken at absolute face value, SEED 1 says **exactly one slip per trial, best trial wins**. I checked that against the contract before building anything, and it is *wrong*, demonstrably:

An alignment of two length-`n` strings with `g` indel pairs has `n−g` aligned columns and score
$$S \;=\; (n-g) - 2\,mm \; -\; 4g \;=\; n - 5g - 2\,mm .$$
A single-slip kernel is exact only when `g_opt ≤ 1`. For `AT…`-style divergent DNA the optimum routinely takes `g ≥ 2`, and the single-slip answer is then strictly below the reference. Shipping it would fail the contract.

The repair is not to abandon the native — it is to **take the flute seriously**. The native's horse makes one slip *per trial*, and the fist is weighed *once at the end of the furrow*. If instead the fist is weighed at **every flute note** (SEED 3 applied per tick rather than per trial), then a single-house slip at tick `k₁`, another at tick `k₂`, another at `k₃` compose freely — and the composition of "at most one one-house slip per note, best-so-far kept at each note" is *exactly* the Needleman–Wunsch recurrence. The native's three seeds, run under the native's own clock, are exact. Nothing is replaced; the weighing is simply moved from the end of the furrow to each note of it.

This lands on a **validated real-world technique**: the anti-diagonal SIMD wavefront (Wozniak 1997), plus an Ukkonen-style provably-sufficient band. That is deliberate, per step 4 — I let the mechanism arrive at the known good thing rather than invent a rival.

# ASSUMPTION BROKEN

Primary: **"a slip (gap) can only be discovered by having already compared the position before it."**
The slip is a *lane offset chosen before the walk*; all offsets are alive simultaneously in the SIMD register, and the previous wavefront is read both slipped-by-one and unslipped.

Carried along: **"one pair of positions is judged at a time"** (16 houses per flute note), **"cells must be computed above→left→diagonal in that order"** (order is by `i+j`), and **"the whole grid must be filled in"** (flung fists).

## The certificate that lets us fling fists — derived from the native's own first weighing

The native's *first* trial — the horse "standing nowhere at all" — yields knot count `H₀`, worth `L₀ = n − 2H₀`. Since the optimum is at least any achievable alignment, `S_opt ≥ L`, and from `S = n − 5g − 2mm ≤ n − 5g`:

$$g_{opt} \;\le\; \frac{n - S_{opt}}{5} \;\le\; \Big\lfloor \frac{n-L}{5} \Big\rfloor \;=\; W,$$

and the path's maximum excursion `max|i−j| ≤ g_opt ≤ W`. **Every diagonal beyond `W` can be flung to the fish with proof, not hope.** I strengthen `L` cheaply by walking the horse to the first 8 doorways on each side (`O(8n)`), exactly the native's trial loop.

- Identical strings: `W = 2` → *O(n)* work.
- 5 % divergence: `W ≈ 0.02n` → ~25× fewer cells.
- Uniform random DNA: `H₀ ≈ 0.75n` → `W ≈ 0.3n` → ~1.7× fewer cells (still exact).

**Two regimes, recognized at runtime by weighing the first fists** (step 5): a *near* regime (narrow band → scalar banded rows, which beat SIMD when the diagonal is shorter than two vectors) and a *far* regime (wide band → AVX2 wavefront). Both exact; the wide path degenerates gracefully to full DP when `W ≥ n`.

**Guards for my own stated risks** (step 4): the wavefront is skipped for `n ≤ 128` (loop overhead), for `2W+1 < 48` (poor lane fill), for `n > 7000` (int16 range: values live in `[−4n, n]`), and whenever AVX2 is absent — each with a correct fallback. No OpenMP: the flute is a strictly serial clock, threads would only fight it.

# ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH -1
#define GAP      -2

/* ---------- tiny strings: plain rolling-row DP (no wavefront overhead) ---------- */
static int nw_tiny(int n, const char *restrict a, const char *restrict b)
{
    int A[130], B[130];
    int *prev = A, *cur = B;
    for (int j = 0; j <= n; j++) prev[j] = j * GAP;
    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = i * GAP;
        for (int j = 1; j <= n; j++) {
            int bst = prev[j - 1] + ((ai == b[j - 1]) ? MATCH : MISMATCH);
            int u = prev[j] + GAP;      if (u > bst) bst = u;
            int l = cur[j - 1] + GAP;   if (l > bst) bst = l;
            cur[j] = bst;
        }
        int *t = prev; prev = cur; cur = t;
    }
    return prev[n];
}

/* ---------- narrow-band regime: scalar banded rows (exact for the proved W) ---- */
static int nw_banded_scalar(int n, const char *restrict a, const char *restrict b, int W)
{
    const int NEG = -(1 << 28);
    int blk = n + 4;
    int *base = (int *)malloc((size_t)(2 * blk) * sizeof(int));
    if (!base) return 0;
    int *prev = base + 1;          /* legal indices -1 .. n+2 */
    int *cur  = base + blk + 1;

    int jhi0 = (W < n) ? W : n;
    for (int j = 0; j <= jhi0; j++) prev[j] = j * GAP;
    prev[-1] = NEG;
    prev[jhi0 + 1] = NEG;

    for (int i = 1; i <= n; i++) {
        int jlo = i - W; if (jlo < 0) jlo = 0;
        int jhi = i + W; if (jhi > n) jhi = n;
        char ai = a[i - 1];
        int left = NEG;
        int j = jlo;
        if (j == 0) { cur[0] = i * GAP; left = cur[0]; j = 1; }
        for (; j <= jhi; j++) {
            int bst = prev[j - 1] + ((ai == b[j - 1]) ? MATCH : MISMATCH);
            int u = prev[j] + GAP;  if (u > bst) bst = u;
            int l = left + GAP;     if (l > bst) bst = l;
            cur[j] = bst;
            left = bst;
        }
        cur[jlo - 1] = NEG;
        cur[jhi + 1] = NEG;
        int *t = prev; prev = cur; cur = t;
    }
    int r = prev[n];
    free(base);
    return r;
}

#if defined(__AVX2__)
/* ---------- wide-band regime: the flute-clocked wavefront -----------------------
   One tick k = i+j = one flute note.  Within a tick every house is independent.
   The horse's single permitted one-house slip is the one-lane offset:
       slipped   -> D1[i-1], D0[i-1]
       unslipped -> D1[i]
   b is stored reversed so that both crowns stream contiguously along a tick.     */
static int nw_wave_avx2(int n, const char *restrict a, const char *restrict b, int W)
{
    const int PAD = 40;
    const short NEG = -30000;                 /* below -4n for n <= 7000 */
    int stride = n + 1 + 2 * PAD;

    short *buf = (short *)malloc((size_t)(3 * stride) * sizeof(short));
    char  *ap  = (char  *)malloc((size_t)n + 64);
    char  *bp  = (char  *)malloc((size_t)n + 64);
    if (!buf || !ap || !bp) {
        free(buf); free(ap); free(bp);
        return nw_banded_scalar(n, a, b, W);
    }
    memcpy(ap, a, (size_t)n);
    memset(ap + n, 0x7f, 64);                 /* padding crowns that never match */
    for (int t = 0; t < n; t++) bp[t] = b[n - 1 - t];
    memset(bp + n, 0x01, 64);

    for (int t = 0; t < 3 * stride; t++) buf[t] = NEG;
    short *D0 = buf + PAD;                    /* tick k-2 */
    short *D1 = buf + stride + PAD;           /* tick k-1 */
    short *D2 = buf + 2 * stride + PAD;       /* tick k   */

    const __m256i vgap = _mm256_set1_epi16(GAP);
    const __m256i vm1  = _mm256_set1_epi16(-1);

    int result = 0;
    const int K = 2 * n;
    for (int k = 0; k <= K; k++) {
        int ilo = k - n; if (ilo < 0) ilo = 0;
        int t1 = k - W;  t1 = (t1 <= 0) ? 0 : (t1 + 1) / 2;   /* ceil((k-W)/2) */
        if (t1 > ilo) ilo = t1;
        int ihi = (k < n) ? k : n;
        int t2 = (k + W) / 2;                                  /* floor((k+W)/2) */
        if (t2 < ihi) ihi = t2;

        int vlo = (ilo < 1) ? 1 : ilo;
        int vhi = (ihi < k - 1) ? ihi : k - 1;
        int base = n - k;

        for (int i = vlo; i <= vhi; i += 16) {
            __m128i ca = _mm_loadu_si128((const __m128i *)(ap + (i - 1)));
            __m128i cb = _mm_loadu_si128((const __m128i *)(bp + base + i));
            __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb)); /* 0 / -1 */
            /* lotus petal (+1) where beasts agree, red knot (-1) where they differ */
            __m256i sc = _mm256_sub_epi16(vm1, _mm256_add_epi16(eq, eq));
            __m256i dg = _mm256_adds_epi16(
                             _mm256_loadu_si256((const __m256i *)(D0 + i - 1)), sc);
            __m256i up = _mm256_adds_epi16(
                             _mm256_loadu_si256((const __m256i *)(D1 + i - 1)), vgap);
            __m256i lf = _mm256_adds_epi16(
                             _mm256_loadu_si256((const __m256i *)(D1 + i)), vgap);
            _mm256_storeu_si256((__m256i *)(D2 + i),
                _mm256_max_epi16(dg, _mm256_max_epi16(up, lf)));
        }
        if (ilo == 0) D2[0] = (short)(k * GAP);   /* (0,k) */
        if (ihi == k) D2[k] = (short)(k * GAP);   /* (k,0) */

        D2[ilo - 1] = NEG; D2[ilo - 2] = NEG;     /* flung fists: out of band */
        D2[ihi + 1] = NEG; D2[ihi + 2] = NEG;

        if (k == K) { result = D2[n]; break; }
        short *tmp = D0; D0 = D1; D1 = D2; D2 = tmp;
    }
    free(buf); free(ap); free(bp);
    return result;
}
#endif

static int hamming_seg(const char *restrict x, const char *restrict y, int len)
{
    int c = 0;
    for (int t = 0; t < len; t++) c += (x[t] != y[t]);
    return c;
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0)   return 0;
    if (n <= 128) return nw_tiny(n, a, b);

    /* ---- weigh the first fists: the horse in the first few doorways, and in none.
       Each trial is an achievable alignment, so its weight is a valid lower bound. */
    int S = 8; if (S > n - 1) S = n - 1;
    int L = -4 * n;
    for (int s = 0; s <= S; s++) {
        int len = n - s;
        int v1 = (len - 2 * hamming_seg(a, b + s, len)) - 4 * s;
        if (v1 > L) L = v1;
        if (s) {
            int v2 = (len - 2 * hamming_seg(a + s, b, len)) - 4 * s;
            if (v2 > L) L = v2;
        }
    }

    /* score = n - 5g - 2*mm  =>  g_opt <= (n - S_opt)/5 <= (n - L)/5,
       and max|i-j| <= g_opt.  Every diagonal past W is provably discardable. */
    int W = (n - L) / 5;
    if (W < 2) W = 2;
    if (W > n) W = n;

#if defined(__AVX2__)
    if (n <= 7000 && (2 * W + 1) >= 48)        /* wide regime: lanes are worth it  */
        return nw_wave_avx2(n, a, b, W);
#endif
    return nw_banded_scalar(n, a, b, W);       /* near regime / no AVX2 / large n  */
}
```

# PREDICTION

**PREDICTION: speedup_vs_dp = 18**

Reasoning, stated before any measurement so it can be scored honestly. Against the reference at `n ≈ 1024`: the baseline allocates and streams a full `(n+1)²` int table (~4.2 MB), scalar, ~1.05 M cells. The artifact on uniform-random DNA holds `W ≈ 0.3n`, so ~0.6 M cells (**1.7×**), processes 16 per instruction group (**~10× realised of a 16× ceiling**, eroded by ~2n ticks of loop prologue at ~20 cycles each), and keeps three `n+80` int16 rows — ~6 KB, fully L1-resident, so the baseline's DRAM write traffic disappears entirely. Product ≈ 17–20×; I commit to **18**.

Regime-dependent expectations, also stated up front:
- near-identical inputs: `W = 2`, scalar banded, **O(n)** → 100×+;
- ~5 % divergence: `W ≈ 0.02n` → 30–60×;
- uniform random: **~18×**;
- `n ≤ 128`: ~1–2× (deliberately conservative; the guard exists to avoid a *loss*, not to win).

# MEASUREMENT

**Not performed by me — `alignment_bench` and `alignment_contract` were unavailable in this session (no tools).** I will not report numbers I did not take. What I can state is what I verified by hand, and what remains at risk:

*Hand-verified:*
- `n=1, a="A", b="A"` → wavefront ticks k=0,1,2 give `D[1]=+1`. ✓ (routed to `nw_tiny` in practice, checked the wavefront arithmetic anyway)
- `n=2, a="AC", b="CA"` → −2; `g_opt=0`, bound `W=⌊4/5⌋=0`. ✓ consistent
- `n=4, a="ACGT", b="CGTA"` → −1 via one slip (3 matches, 2 gap columns); `H₀=4 ⇒ W=1`, band contains it. ✓
- Buffer bounds: max read `ap[n+14]`, `bp[n+14]` (pads of 64); max wavefront index `n+15` against `PAD=40`; sentinels at `ilo−2 … ihi+2` cover every read, since `ilo` and `ihi` each advance by at most 1 per tick. ✓
- int16 range: values lie in `[−4n, n]`; at the `n=7000` gate that is `[−28000, 7000]`, sentinel `−30000`, saturating `adds` floor `−32768`. ✓

*Open risk I am not hiding:* correctness of the band rests entirely on `max|i−j| ≤ g_opt ≤ (n−L)/5`. I believe the derivation is airtight (`h_steps = v_steps = g` because both strings have length `n`; each of the `2g` gap columns costs `−2`; deviation is bounded by the count of one-sided steps). But it is a *proof*, not a *measurement*, and the pipeline's exact-match check against the reference DP is what will actually settle it. If any case mismatches, the first thing to try is forcing `W = n` — that reduces the artifact to a plain exact SIMD wavefront and isolates whether the fault is in the fling-to-the-fish bound or in the lanes.

# VERDICT

The native's picture survives translation almost unchanged, and one thing genuinely fell out of taking it literally.

What did **not** survive: the literal single-slip reading. A kernel that places exactly one gap and weighs the fist only at the end of the furrow is *measurably wrong* against the contract whenever the optimum needs two or more indel pairs — which divergent DNA routinely does. I say that plainly rather than shipping it. But the repair came from inside the metaphor, not from the textbook: **move the weighing from the end of the furrow to each note of the flute**, and single one-house slips compose into arbitrary gap patterns. The result is exact Needleman–Wunsch, and it is the validated Wozniak anti-diagonal wavefront — which is the right outcome per step 4: the mechanism *arrived at* a known-good technique instead of inventing a rival to it.

What fell out that I did not expect: **the native's first fist is a certificate.** "The horse standing nowhere at all" is the Hamming distance, and because `score = n − 5g − 2·mm`, that one cheap weighing *proves* the horse can never need to wander more than `⌊(n−L)/5⌋` houses from the road. The fish get their fists with a receipt. That is a real Ukkonen-style bound, it costs `O(n)`, it is exact, and it is the runtime regime detector step 5 asked for — narrow band for similar strings, wide band for random ones, full table when the proof demands it.

Where this can lose, and what I did about it, since I refuse to ship a risk I named:
- *short sequences* → wavefront prologue dominates ⇒ guarded, `n ≤ 128` falls to plain rolling DP;
- *very narrow bands* → SIMD lanes go unfilled ⇒ guarded, `2W+1 < 48` falls to scalar banded rows;
- *large `n`* → int16 overflow ⇒ guarded at 7000, falls to the int32 scalar band;
- *no AVX2* → compile-time guard, same fallback.
Every path is exact; none of them is slower than the baseline.

Honest accounting of novelty: the wavefront is known, the band is known. What the metaphor contributed was **the specific constant 5** — the exact exchange rate between a knot and a slip in this scoring scheme, which is what makes the band provable rather than heuristic, and which I would not have gone looking for if the native had not insisted that the horse's slip be placed "where it does the least harm." The claim of an 18× speedup is a prediction, not a result. It is the pipeline's to confirm or to refute, and I would rather it be refuted cleanly than defended.