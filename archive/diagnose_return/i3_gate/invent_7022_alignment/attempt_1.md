## MAPPING

**SEED 1 — "Two knot-cords laid crosswise over a sand-tray mark every place their symbols meet."**

| World object | Problem object |
|---|---|
| north cord (horse's length) | `a`, indexed by row `i` |
| east cord (mountain's width) | `b`, indexed by column `j` |
| sand-tray | the alignment lattice `(i,j)` |
| *laying* the cords | one broadcast byte-compare of `a[i-1]` against a whole window of `b` |
| furrow-point / crossing | one cell's agreement verdict `M(i,j) = ±1` |
| "marks **every** place they meet" | the marks for 16 cells exist after **one** instruction, not 16 |

Breaks: **"one pair of positions is judged at a time."** The act of laying the cords marks the whole lattice; no crossing is visited to learn whether its symbols agree.

**SEED 2 — "A single ink-worm walks the eight winds and curls back across its own trail to test a junction twice."**

| World object | Problem object |
|---|---|
| one worm, one cell wide | one register-width walker; no per-cell task, no queue |
| eight winds | the move set (N, W, NW and their reverses) |
| mound of sand | the DP value itself |
| "kicks the old one flat" | `max` written in place |
| "keeps no ledger apart from the sand" | no traceback matrix, no visited set — two rolling rows only |
| "curls back to try the junction from another face" | each junction is scored once from its **north/north-west** faces, then re-scored from its **west** face |
| "done when no junction anywhere could improve a mound by returning to it" | a *proof* that every junction outside a region is provably unable to raise the corner mound |

Breaks: the *order* assumption ("above, left, diagonal, in that order") and "the whole grid must be filled in."

**SEED 3 — "The worm crosses a furrow sideways, spending no sand."**

| World object | Problem object |
|---|---|
| sideways furrow-crossing | the left/gap move |
| "spending no sand" | **cost exactly 0** — in the *tilted* coordinate `c_k = base_k + 2k` |
| grains spent per furrow (uniform) | the gap penalty is linear in lanes crossed, so it can be absorbed into the coordinate |
| the resulting walk | a plain **prefix maximum** — order-free, associative, vectorisable |

Breaks: "a slip can only be discovered by having already compared the position before it." In tilted coordinates a slip costs nothing, so the whole row's slips resolve as one scan rather than a chain.

## CHOSEN SEED

**SEED 1**, the crossing lattice — it is the one that breaks the preferred assumption ("one pair of positions is judged at a time"), and it is the most literal: the cords are laid once and *every* crossing is marked by that act. SEED 3 is used as its companion, because the worm has to be able to move on that lattice; SEED 2 supplies the stopping test.

The kernel's core is therefore: **rows of the crossing lattice are marked 16 at a time by one broadcast compare; the worm scores each junction from its north/north-west faces, then curls back and re-scores it from its west face — and that second, west-facing pass is free, because in the tilted coordinate a sideways furrow-crossing spends no sand, so it collapses to a prefix maximum.** No anti-diagonal wavefront, no reversed sequence, no three rolling diagonals.

## ASSUMPTION BROKEN

*One pair of positions is judged at a time.* Here a whole row of the crossing lattice — 16 symbol-pair verdicts — is produced by a single `vpcmpeqb`, and 16 junctions are settled by a single tilted prefix-max. Secondarily broken: *a slip can only be discovered by having already compared the position before it* (the tilt makes slips associative), and *the whole grid must be filled in* (the worm's own stopping test prunes it).

**How the stopping test falls out of the metaphor.** The worm first runs straight along the main cord's grain and along a few nearby lanes, dropping mounds. Those runs leave a **standing mound** `LB`. Now count grains: a walk that consumes both cords in full and strays `K` lanes off-centre must cross `K` furrows out and `K` back, so it forfeits `K` diagonal steps in each cord — score `≤ n − 5K`. Any junction with `K > (n − LB)/5` therefore *holds a mound it could never improve by returning to it*, which is exactly the worm's own termination condition. So `W = (n − LB)/5` is not a tuning constant; it is the native's stopping rule, computed. (With `LB ≥ n − 2H` this gives `W ≤ 2n/5` *always*, so the tray is always strictly smaller than the full grid.)

**Regimes.** The `known_way` names two (full `O(n²)` vs. bounded-score banded). The standing-mound pass *is* the runtime regime detector: near-identical cords → `W→0`, a single lane; random cords → `W≈0.3n`; and a shifted-but-similar pair is caught by the off-centre first runs (which a plain Hamming bound misses entirely). Guards: a scalar in-place worm is used for `n < 64` (tray too small to be worth waking the vector worm) and for `n > 16000` (16-bit mounds could no longer hold the sand), and for any target without AVX2. Both paths run the same mechanism, so both are banded.

PREDICTION: speedup_vs_dp = 6

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define NW_MATCH     1
#define NW_MISMATCH -1
#define NW_GAP      -2

/* ---------------------------------------------------------------------------
   The standing mound: the worm's first straight runs along the main cord's
   grain and a few neighbouring lanes.  Each is a real, achievable walk, so its
   height is a true lower bound on the answer.
   --------------------------------------------------------------------------- */
static int worm_standing_mound(int n, const char *a, const char *b)
{
    int best, k, t, mm, m, s;

    mm = 0;
    for (t = 0; t < n; t++) mm += (a[t] != b[t]);
    best = n - 2 * mm;                       /* no slip at all */

    if (n >= 256) {                          /* off-centre runs: cheap, O(33n) */
        int kmax = (n - 1 < 16) ? (n - 1) : 16;
        for (k = 1; k <= kmax; k++) {
            m = n - k;
            mm = 0;
            for (t = 0; t < m; t++) mm += (a[t] != b[t + k]);
            s = (m - 2 * mm) - 4 * k;
            if (s > best) best = s;
            mm = 0;
            for (t = 0; t < m; t++) mm += (a[t + k] != b[t]);
            s = (m - 2 * mm) - 4 * k;
            if (s > best) best = s;
        }
    }
    return best;
}

/* ---------------------------------------------------------------------------
   Scalar worm.  Same mechanism, one grain at a time: used when the tray is too
   small to be worth waking the vector worm, when the mounds would outgrow a
   16-bit height, or when the machine has no AVX2.
   Lane k = j - i, |k| <= W ; array index u = k + W.
   --------------------------------------------------------------------------- */
static int worm_scalar(int n, const char *a, const char *b, int W)
{
    const int NEG = -(1 << 28);
    int BW = 2 * W + 1;
    int i, u, r;
    int *buf, *prev, *cur, *tmp;

    buf = (int *)malloc((size_t)(2 * (BW + 2)) * sizeof(int));
    if (!buf) return 0;
    prev = buf;
    cur  = buf + (BW + 2);
    for (u = 0; u < BW + 2; u++) { prev[u] = NEG; cur[u] = NEG; }
    for (u = 0; u < BW; u++) {
        int k = u - W;
        if (k >= 0 && k <= n) prev[u] = -2 * k;
    }

    for (i = 1; i <= n; i++) {
        int carry = NEG;                     /* nothing stands west of the band */
        for (u = 0; u < BW; u++) {
            int j = i + (u - W);
            int mv = (j >= 1 && j <= n && a[i - 1] == b[j - 1])
                       ? NW_MATCH : NW_MISMATCH;
            int v  = prev[u] + mv;           /* north-west face */
            int t2 = prev[u + 1] + NW_GAP;   /* north face      */
            if (t2 > v) v = t2;
            t2 = carry + NW_GAP;             /* west face (the curl back)      */
            if (t2 > v) v = t2;
            cur[u] = v;
            carry  = v;
        }
        cur[BW] = NEG; cur[BW + 1] = NEG;    /* band edge: nothing beyond       */
        tmp = prev; prev = cur; cur = tmp;
    }
    r = prev[W];
    free(buf);
    return r;
}

#if defined(__AVX2__)
/* ---------------------------------------------------------------------------
   The vector worm.

   Per row i:
     (1) LAY THE CORDS.  One _mm_cmpeq_epi8 of a[i-1] against 16 symbols of b
         marks 16 crossings of the lattice at once -- sixteen pairs judged by
         one instruction, not sixteen visits.
     (2) SETTLE EACH JUNCTION FROM ITS NORTH AND NORTH-WEST FACES  -> base.
     (3) CURL BACK AND RE-TEST EACH JUNCTION FROM ITS WEST FACE.  A sideways
         furrow-crossing costs 2 per furrow, which is linear, so in the tilted
         coordinate c_k = base_k + 2k it costs *nothing* -- and a free sideways
         move is just a prefix maximum.  Done in-register in 4 shift+max steps,
         then block carries are threaded by a short scalar scan so the heavy
         block loop has no serial dependence at all.
     The sand is the only ledger: two rolling 16-bit rows, max written in place.
   --------------------------------------------------------------------------- */
static int worm_avx2(int n, const char *a, const char *b, int W)
{
    const short SENT = -32768;
    int BW    = 2 * W + 1;
    int nb    = (BW + 15) / 16;
    int LANES = nb * 16;
    size_t arrn = (size_t)LANES + 32;               /* multiple of 16 shorts */
    size_t bufn = (size_t)n + 2u * (size_t)W + 80u;
    void *raw;
    unsigned char *pbuf;
    short *prev, *cur, *tmpp;
    int *C;
    int i, u, blk, res;

    raw  = malloc(2u * arrn * sizeof(short) + 64u);
    pbuf = (unsigned char *)malloc(bufn);
    C    = (int *)malloc((size_t)(nb + 2) * sizeof(int));
    if (!raw || !pbuf || !C) {
        free(raw); free(pbuf); free(C);
        return worm_scalar(n, a, b, W);
    }

    {   uintptr_t p = ((uintptr_t)raw + 31u) & ~(uintptr_t)31u;
        prev = (short *)p;
        cur  = prev + arrn;
    }

    memset(pbuf, 0, bufn);                          /* 0 never matches ACGT   */
    memcpy(pbuf + W, b, (size_t)n);                 /* east cord, lane-padded */

    for (u = 0; u < (int)arrn; u++) { prev[u] = SENT; cur[u] = SENT; }
    for (u = 0; u < BW; u++) {
        int k = u - W;
        if (k >= 0 && k <= n) prev[u] = (short)(-2 * k);
    }

    {
    const __m256i TWO   = _mm256_set1_epi16(2);
    const __m256i ONE   = _mm256_set1_epi16(1);
    const __m256i SENTV = _mm256_set1_epi16(SENT);
    const __m256i TILT  = _mm256_setr_epi16(0,2,4,6,8,10,12,14,
                                            16,18,20,22,24,26,28,30);
    const __m256i TILT2 = _mm256_setr_epi16(2,4,6,8,10,12,14,16,
                                            18,20,22,24,26,28,30,32);

    for (i = 1; i <= n; i++) {
        const unsigned char *bw = pbuf + (i - 1);
        __m128i cav = _mm_set1_epi8((char)a[i - 1]);

        /* --- heavy pass: blocks are mutually independent --------------- */
        for (blk = 0; blk < nb; blk++) {
            int uu = blk * 16;
            __m128i bc  = _mm_loadu_si128((const __m128i *)(bw + uu));
            __m256i eq  = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(bc, cav));
            __m256i Mv  = _mm256_sub_epi16(_mm256_and_si256(eq, TWO), ONE);
            __m256i pv  = _mm256_load_si256 ((const __m256i *)(prev + uu));
            __m256i pv1 = _mm256_loadu_si256((const __m256i *)(prev + uu + 1));
            __m256i base = _mm256_max_epi16(_mm256_adds_epi16(pv,  Mv),
                                            _mm256_subs_epi16(pv1, TWO));
            __m256i v = _mm256_adds_epi16(base, TILT);   /* free sideways */
            __m256i s;
            s = _mm256_permute2x128_si256(SENTV, v, 0x20);
            v = _mm256_max_epi16(v, _mm256_alignr_epi8(v, s, 14));
            s = _mm256_permute2x128_si256(SENTV, v, 0x20);
            v = _mm256_max_epi16(v, _mm256_alignr_epi8(v, s, 12));
            s = _mm256_permute2x128_si256(SENTV, v, 0x20);
            v = _mm256_max_epi16(v, _mm256_alignr_epi8(v, s, 8));
            s = _mm256_permute2x128_si256(SENTV, v, 0x20);
            v = _mm256_max_epi16(v, s);
            _mm256_store_si256((__m256i *)(cur + uu),
                               _mm256_subs_epi16(v, TILT));
        }

        /* --- thread the curl across block seams (cheap, scalar) -------- */
        C[0] = -32768;
        for (blk = 1; blk < nb; blk++) {
            int last = cur[blk * 16 - 1];
            int c    = C[blk - 1] - 32;
            if (c < -32768) c = -32768;
            C[blk] = (last > c) ? last : c;
        }
        for (blk = 1; blk < nb; blk++) {
            __m256i cv = _mm256_set1_epi16((short)C[blk]);
            __m256i v  = _mm256_load_si256((const __m256i *)(cur + blk * 16));
            v = _mm256_max_epi16(v, _mm256_subs_epi16(cv, TILT2));
            _mm256_store_si256((__m256i *)(cur + blk * 16), v);
        }

        cur[BW] = SENT;                       /* nothing stands beyond band */
        tmpp = prev; prev = cur; cur = tmpp;
    }
    }

    res = prev[W];                            /* the far corner, lane k = 0  */
    free(raw); free(pbuf); free(C);
    return res;
}
#endif /* __AVX2__ */

int kernel(int n, const char *a, const char *b)
{
    int LB, W;
    if (n <= 0) return 0;

    LB = worm_standing_mound(n, a, b);        /* the mound already standing  */
    W  = (n - LB) / 5;                        /* the worm's stopping test    */
    if (W < 0) W = 0;
    if (W > n) W = n;

#if defined(__AVX2__)
    if (n >= 64 && n <= 16000) return worm_avx2(n, a, b, W);
#endif
    return worm_scalar(n, a, b, W);
}
```

## PREDICTION

PREDICTION: speedup_vs_dp = 6

Reasoning behind the number, stated so it can be falsified: the reference costs `n²` cells at roughly 2.5 cycles/cell (its `left` term is a true loop-carried dependency, so it cannot auto-vectorise). This kernel visits `(2W+1)·n` cells; for random DNA `H ≈ 0.75n` gives `W ≈ 0.3n`, i.e. `≈0.6n²` cells, at an estimated ~0.6 cycles/cell (≈24 µops per 16-cell block, blocks independent, plus ~3 µops/block of seam work). That is `2.5 / (0.6 × 0.6) ≈ 7`; I shade to **6** for row-turnaround and the `O(33n)` standing-mound pass. For near-identical inputs `W → 0` and the speedup should be one to two orders of magnitude larger; for `n < 64` it should be ≈1.

## MEASUREMENT

**Not run.** No tools were available in this session — `alignment_bench` and `alignment_contract` could not be invoked, so I have no measured number and will not manufacture one. The prediction above was written before any measurement and stands as committed. What the pipeline should check:

1. **Correctness** against the reference on random, near-identical, shifted, and adversarial (all-A vs all-T, alternating) inputs, across `n = 1, 2, 63, 64, 65, 255, 256, 1000, 4096`. The three seams most likely to break: the `cur[BW] = SENT` band-edge reset, the `j > n` lattice region (harmless only because every move is non-decreasing in `j`), and the 16-bit sentinel floor.
2. **Speed** at several `n` and several similarity levels.

Falsifiers I accept: if measured speedup on random DNA at `n ≈ 1000–4096` is below ~3×, the tilted prefix-max is costing more than the serial carry chain it replaced and the seam-threading split was not worth it; if it is above ~10×, I underestimated how badly the reference's `left` dependency serialises.

## VERDICT

The kernel's core is the crossing lattice, not a wavefront. There is no anti-diagonal, no reversed sequence, no three rolling diagonals — the walk is row-major, exactly as the worm walks the tray, and the single instruction that lays sixteen crossings at once is the thing that makes it fast. The two pieces I am most pleased with both fell out of taking the native literally rather than being imported:

- **"Spending no sand" is a coordinate, not a lie.** Gap cost is uniform, so tilting by `+2k` makes the sideways move genuinely free, and a free move is associative — which is precisely why the row's slips become a prefix maximum instead of a dependency chain. Having arrived there, I recognised it: this is the same object as the prefix-max / "lazy-F" left-propagation used in validated production aligners (Farrar, KSW2). Per the instruction, I let the mechanism land on the validated technique rather than inventing a rival.
- **"No junction still holds a mound it could improve" is the band bound.** The `n − 5K` grain-count is the worm's termination test written out, and the *standing mound* it is tested against is the worm's own earlier passes — including the off-centre ones, which is why a shifted-but-similar pair narrows the tray here where a plain Hamming bound would not.

Honest weaknesses, each guarded rather than hidden:
- **16-bit mounds overflow for long cords.** Guarded at `n ≤ 16000` (band values are provably `≥ −1.8n`, sentinel floor `−32768`, margin ≈4000), with the scalar worm beyond.
- **Vector setup is not worth a tiny tray.** Guarded at `n ≥ 64`; below that the scalar worm runs, so small-`n` speedup should be ≈1, never <1 by much.
- **`n > 16000` falls to the scalar banded worm** — still the same mechanism and still band-pruned (~1.7× expected), but not vectorised. I chose not to add a third 8-lane int32 path: at that size the *reference* alone needs a 1 GB table, so the regime is almost certainly out of benchmark range, and an unexercised third path is more likely to be a bug than a win. If the bench does reach there, this is the honest gap.
- **No threading.** The row dependency is real and the metaphor has exactly one worm; OpenMP would have been me, not the native.