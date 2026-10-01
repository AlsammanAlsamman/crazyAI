# MAPPING

## SEED 1 — "Two knot-cords laid crosswise over a sand-tray mark every place their symbols meet."

| World object | Problem object |
|---|---|
| knot-cord running north, the horse's length | sequence `a`, indexed by row `i` |
| knot-cord running east, the mountain's width | sequence `b`, indexed by column `j` — **and, because the tray's frame turns it, read as `b` reversed** |
| the crossing of the two cords | the (n+1)×(n+1) score lattice |
| a furrow-point where "a symbol from one cord faces a symbol from the other" | the match/mismatch indicator `a[i-1]==b[j-1]`, existing **before** any walker arrives |
| "mark **every** place their symbols meet" | all symbol judgements of a whole strip are struck at once, not one at a time |
| laying the cords (an act done once, by hand, with no walker) | one `vpcmpeqb` over 16 characters → 8–16 substitution scores in one instruction |

**Silent assumption broken:** *"one pair of positions is judged at a time."* The lattice of agreements is a property of the crossed cords, established wholesale, prior to and independent of any traversal.

## SEED 2 — "A single ink-worm walks the eight desert winds and curls back across its own trail to test a junction twice."

| World object | Problem object |
|---|---|
| a single ink-worm, **one cell wide**, "a stitch cut loose from the verses" | one SIMD register: width 1 in the direction of travel, many junctions long across it |
| eight desert winds | the 8-neighbourhood; only 3 (N, W, NW) ever pay |
| mound of sand at a junction | `dp[i][j]` |
| "keeps no ledger apart from the sand itself" | no work-queue, no predecessor table — the rolling score arrays *are* the whole state |
| "curls back... to test the junction from another face"; "no junction still holds a mound it could improve" | Bellman–Ford-style re-relaxation to a fixed point |
| "a beaten mound is thrown away entirely" | max-reduce in place; traceback discarded |

**Silent assumption broken:** *"every cell depends on above, left and diagonal, computed in that order."* The worm arrives from an arbitrary face and re-tests. (It does **not** break "one pair at a time" — a one-junction-wide worm settling at "every junction it settles" is still serial; and the doubling-back fixed point is strictly *more* work than one ordered sweep, since the NW dependency graph is already a DAG.)

## SEED 3 — "The worm crosses a furrow sideways, spending no sand, to let one cord slip a step ahead of the other."

| World object | Problem object |
|---|---|
| crossing a furrow sideways | moving one lane sideways within a register |
| "spending no sand" | the move costs **zero instructions**: an unaligned load offset by one `int`, i.e. a free lane-slip |
| "one cord slips a step ahead of the other" | the relative offset `i-j` changes by ±1 — which is exactly what a gap *is* |
| "the one permitted stumble, where step and step fall out of match" | the gap step; its −2 is charged in the arithmetic, not in the slip |
| straight along a cord's grain | the main diagonal, `i-j` constant |
| the worm's body lying across a line of junctions | an anti-diagonal front `d = i+j`, all of whose cells are mutually independent |

**Silent assumption broken:** *"a slip (gap) can only be discovered by having already compared the position before it."* The slip is a structural re-indexing of the front, available to every lane simultaneously, with no serial left-to-right chain.

---

# CHOSEN SEED

**SEED 1.** It is the only one of the three that breaks the preferred assumption ("one pair of positions is judged at a time"), and its mapping is the most literal: the cords laid crosswise over the tray *are* two character vectors compared edge-to-edge, and the furrow-points they mark *are* the substitution scores, all struck at once.

SEED 1 alone, however, only gives me the free half of the problem (substitution scores vectorise trivially). The reason a textbook row-wise loop cannot use it is the serial `dp[i][j-1]` chain — so I take SEED 3's free sideways slip as the *mechanism* that lets the wholesale judgement actually be spent, and SEED 2's "no ledger apart from the sand itself" as the memory discipline. All three seeds are one worm; I am not substituting a different idea.

I discard exactly one thing from SEED 2, and say so plainly: the **doubling-back fixed point**. Taken literally it is Bellman–Ford on a graph that is already a DAG with a known topological order — strictly slower, never different in answer. The worm's own words give me the licence: "walks *straight along a cord's grain*" is the topological order, and "it is done only when it reaches the far corner" is satisfied on the first pass. So the curling survives as *one* curl: the worm's body lies along the anti-diagonal and its trail crosses its own previous two trails — "never lifted from the sand, never broken into separate glances."

---

# ASSUMPTION BROKEN

**"One pair of positions is judged at a time"** — broken directly: 8 furrow-points of one anti-diagonal are judged, scored and settled per instruction group.

Broken as a consequence, and all of them load-bearing:

- **"cells computed above/left/diagonal in that order"** → the front advances along `d = i+j`; within a front there is no order at all.
- **"a slip can only be discovered by comparing the position before it"** → the slip is a one-lane unaligned load, free, for all lanes at once.
- **"both strings are read start to end in the same direction"** → the east cord is stored **reversed** (`R[t] = b[n-1-t]`), because that is the only way both character loads along a furrow are contiguous ascending. This is literally "one running north, the other running east."
- **"the whole grid must be filled in"** → the worm first walks the straight grain and counts the grains of sand it spends there; that tally bounds how far off-grain it could ever profitably wander, so it crawls only that furrow-band.

**Where the mechanism lands on validated ground (step 4).** I did not invent either half. The anti-diagonal front with a one-lane slip *is* Woźniak's 1997 diagonal SIMD alignment (and the diagonal scheme in CUDASW++/GASAL). The straight-grain tally bounding the band *is* Ukkonen/Fickett banding, with the bound rederived here for this exact scoring. The worm arrived at both; I let it, rather than shipping something novel and untested.

**The band bound, derived, not guessed.** Both strings have length `n`, so any path uses `g` gaps in each, `m = n-g` diagonal steps. Its score is at most `(n-g) + 2g·(-2) = n - 5g`. The pure straight-grain path scores `n - 2D`, where `D` is the Hamming distance of the main diagonal. An optimal path therefore needs `n - 5g ≥ n - 2D`, i.e. `g ≤ 2D/5`; and since `|i-j|` changes by 1 only on a gap step, `|i-j| ≤ g`. So half-width `k = ⌊2D/5⌋` is provably sufficient — I ship `⌊2D/5⌋+2`.

**Both regimes, one kernel (step 5).** `known_way` names two: banded (similar sequences) and full (divergent). The worm recognises which tray it is in by walking the grain first — `D` is measured at runtime in O(n). Identical cords → `k=2`, a 5-wide furrow, O(n) work. Random DNA → `D ≈ 0.75n` → `k ≈ 0.3n`, a 0.6n²-cell tray. Adversarially divergent → `k` clamps to `n` and the band constraint goes inactive, degenerating *exactly* to the full anti-diagonal sweep. One code path, one parameter.

**Risks named in my own verdict, and guarded rather than hand-waved:**
- small trays → `n < 64` falls back to plain two-row scalar DP;
- no AVX2 on the host → `#if defined(__AVX2__)`, scalar tail loop carries the kernel, and if additionally the band is wide (`2k ≥ n`) it hands off to the cache-friendly row DP, which has better locality than a scalar diagonal sweep;
- int16 lanes would double throughput but overflow near `4n > 32767` — **dropped entirely**, not guarded-and-shipped; int32 is unconditionally safe.
- **No OpenMP.** The metaphor insists on *one* worm, *never lifted from the sand*, and it is right: anti-diagonal fronts at n ≈ 10²–10³ are far too short to amortise a barrier per diagonal. Vectorisation only, per instruction.

---

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
#define NEGINF   (-(1 << 28))   /* deeper than any real mound (>= -4n), safe under repeated +GAP */

/* ---- plain exact two-row DP: tiny trays, and the wide-band no-AVX2 regime ---- */
static int nw_rows(int n, const char *a, const char *b)
{
    int *prev, *cur, *tmp, i, j, r;
    prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (j = 0; j <= n; j++) prev[j] = j * GAP;
    for (i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = i * GAP;
        for (j = 1; j <= n; j++) {
            int s  = (ai == b[j - 1]) ? MATCH : MISMATCH;
            int t1 = prev[j - 1] + s;
            int t2 = (prev[j] > cur[j - 1] ? prev[j] : cur[j - 1]) + GAP;
            cur[j] = t1 > t2 ? t1 : t2;
        }
        tmp = prev; prev = cur; cur = tmp;
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

/* ---- the ink-worm: one body-wide front along d = i+j, confined to |i-j| <= k ----
 * cur[i]  = dp[i][d-i]
 *         = max( p2[i-1] + s(a[i-1], b[d-i-1]),          (straight along the grain)
 *                max(p1[i-1], p1[i]) + GAP )             (the free sideways slip)
 * p1 = diagonal d-1, p2 = diagonal d-2. Nothing is kept but the sand itself.
 */
static int nw_band(int n, const char *a, const char *b, int k)
{
    const int PAD = 40;
    size_t stride = (size_t)n + 2 + 2 * (size_t)PAD;
    int *mem   = (int *)malloc(3 * stride * sizeof(int));
    char *cbuf = (char *)malloc(2 * ((size_t)n + 2 * (size_t)PAD));
    int *X, *Y, *Z, *cur, *p1, *p2;
    char *A, *R;
    int d, i, res;
    size_t t;

    if (!mem || !cbuf) { free(mem); free(cbuf); return nw_rows(n, a, b); }

    for (t = 0; t < 3 * stride; t++) mem[t] = NEGINF;
    X = mem + PAD;                    /* usable indices -PAD .. n+1+PAD */
    Y = mem + stride + PAD;
    Z = mem + 2 * stride + PAD;

    /* the two cords: north cord forward, east cord reversed, both padded with
       characters that can never agree with anything real */
    A = cbuf;
    R = cbuf + (size_t)n + 2 * (size_t)PAD;
    memcpy(A, a, (size_t)n);
    memset(A + n, 0x01, 2 * (size_t)PAD);
    for (i = 0; i < n; i++) R[i] = b[n - 1 - i];
    memset(R + n, 0x02, 2 * (size_t)PAD);

    Y[0] = 0;                         /* diagonal d = 0, the near corner */
    cur = X; p1 = Y; p2 = Z;          /* Z stands for the empty diagonal d = -1 */

    for (d = 1; d <= 2 * n; d++) {
        int lo = d - n, hi = d, lb, ub, ilo, ihi, q;
        if (lo < 0) lo = 0;
        lb = (d - k + 1) >> 1;        /* ceil((d-k)/2) */
        if (lb > lo) lo = lb;
        if (hi > n) hi = n;
        ub = (d + k) >> 1;            /* floor((d+k)/2) */
        if (ub < hi) hi = ub;

        ilo = lo > 1 ? lo : 1;
        ihi = hi < d - 1 ? hi : d - 1;
        q   = n - d;                  /* b[d-i-1] == R[q+i], ascending in i */
        i   = ilo;

#if defined(__AVX2__)
        {
            const __m256i vgap = _mm256_set1_epi32(GAP);
            const __m256i vpos = _mm256_set1_epi32(MATCH);
            const __m256i vneg = _mm256_set1_epi32(MISMATCH);
            for (; i + 7 <= ihi; i += 8) {
                /* SEED 1: the cords crossing marks eight furrow-points at once */
                __m128i av = _mm_loadl_epi64((const __m128i *)(A + i - 1));
                __m128i bv = _mm_loadl_epi64((const __m128i *)(R + q + i));
                __m256i eq = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(av, bv));
                __m256i s  = _mm256_blendv_epi8(vneg, vpos, eq);
                /* SEED 3: the sideways slip is a one-lane offset, spending nothing */
                __m256i dg = _mm256_loadu_si256((const __m256i *)(p2 + i - 1));
                __m256i up = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
                __m256i lf = _mm256_loadu_si256((const __m256i *)(p1 + i));
                __m256i best = _mm256_max_epi32(
                        _mm256_add_epi32(dg, s),
                        _mm256_add_epi32(_mm256_max_epi32(up, lf), vgap));
                _mm256_storeu_si256((__m256i *)(cur + i), best);
            }
        }
#endif
        for (; i <= ihi; i++) {
            int s  = (A[i - 1] == R[q + i]) ? MATCH : MISMATCH;
            int t1 = p2[i - 1] + s;
            int t2 = (p1[i - 1] > p1[i] ? p1[i - 1] : p1[i]) + GAP;
            cur[i] = t1 > t2 ? t1 : t2;
        }

        /* the two tray edges, where a cord has run out entirely */
        if (lo == 0) cur[0] = GAP * d;
        if (hi == d) cur[d] = GAP * d;

        /* fence the furrow: one junction beyond each rim holds no mound at all */
        cur[lo - 1] = NEGINF;
        cur[hi + 1] = NEGINF;

        { int *sw = p2; p2 = p1; p1 = cur; cur = sw; }
    }

    res = p1[n];                      /* far corner: dp[n][n], diagonal d = 2n */
    free(mem); free(cbuf);
    return res;
}

int kernel(int n, const char *a, const char *b)
{
    int i, D, k;
    long long kk;

    if (n <= 0) return 0;
    if (n < 64) return nw_rows(n, a, b);          /* guard: tiny tray, no worm */

    /* walk the straight grain once and count the sand it costs: this is the
       runtime regime test.  D small -> narrow furrow; D large -> whole tray. */
    D = 0;
    for (i = 0; i < n; i++) D += (a[i] != b[i]);

    /* any optimal path uses g gaps per cord with n-5g >= n-2D, so g <= 2D/5,
       and |i-j| <= g.  Half-width 2D/5 is provably sufficient; +2 for slack. */
    kk = (2LL * (long long)D) / 5 + 2;
    k  = (kk > (long long)n) ? n : (int)kk;       /* k == n -> band inactive */
    if (k < 2) k = 2;

#if !defined(__AVX2__)
    /* no vector lanes and a wide furrow: the row DP has better locality */
    if (2 * k >= n) return nw_rows(n, a, b);
#endif
    return nw_band(n, a, b, k);
}
```

---

# PREDICTION

**PREDICTION: speedup_vs_dp = 10.0**

Written before any measurement. Reasoning: 8 int32 lanes per front × ~1.67× from the band at random-DNA divergence (`D ≈ 0.75n → k ≈ 0.3n`, so ≈0.6n² cells instead of n²) ≈ 13× of arithmetic, discounted for short fronts near the two corners, ~2n front-setup overheads, and a scalar tail of ≤7 cells per front. Against that, the reference also pays O(n²) write traffic to a full (n+1)² table (16 MB at n=2048) which my O(n) rolling sand does not, so the true figure could exceed 10× at large n and fall short of it near n≈64–128 where front lengths are short.

---

# MEASUREMENT

**Not performed. No tools were available in this session** — `alignment_bench`, `alignment_contract` and the `symbolic_*`/`unconventional_*` helpers were all named in the brief but none are callable here, so I could neither validate against the reference aligner nor time anything. I am reporting that as plainly as I would report a bad number, and the four permitted improvement rounds are therefore **unused**: what is above is the first artifact, not a tuned one.

What I verified by hand instead, since I could not verify it by machine:

- **Recurrence indices.** For `cur[i] = dp[i][d-i]`: `dp[i-1][j-1]` sits at index `i-1` of diagonal `d-2`; `dp[i-1][j]` at index `i-1` of `d-1`; `dp[i][j-1]` at index `i` of `d-1`. Both gap terms take the same `GAP`, so `max(up,left)` can be taken before adding it — 1 vector add saved per 8 cells.
- **Reversed-cord addressing.** `b[d-i-1] = R[n-d+i]`, ascending in `i`, so both character loads are contiguous.
- **Sentinel sufficiency.** Reads on front `d` span indices `lo(d)-1 … hi(d)` of `d-1` and `lo(d)-1 … hi(d)-1` of `d-2`. `lo` is non-decreasing and `hi` rises by at most 1 per front, so the two `NEGINF` fence cells written each front exactly cover every stale read. Nothing outside the furrow can leak a real value in.
- **No spurious `NEGINF` inside the furrow.** If `|i-j| ≤ k` then `|(i-1)-(j-1)| ≤ k`, so every in-band interior cell has an in-band diagonal predecessor and chains back to a tray edge with a finite mound.
- **Overflow.** Mounds lie in `[-4n, n]`; `NEGINF = -2^28` can absorb `2n` further `GAP`s without approaching int32 range. (This is precisely why the int16 variant was dropped rather than guarded.)
- **Bounds.** Vector stores are constrained by `i + 7 <= ihi`, so nothing is written outside `[ilo, ihi]`; loads reach at most index `n+6`, covered by `PAD = 40` on all four buffers.
- **Band correctness.** Restricting the domain can only lower the score, and the derivation above shows the optimal path never leaves `|i-j| ≤ ⌊2D/5⌋`. So banded = full = reference, exactly, for every input.

**The three numbers that would settle it,** in the order I would run them: (1) exact-match against the reference over ~10⁴ random pairs at `n = 64, 65, 127, 128, 1000` plus the adversarial edges `a == b`, `a`/`b` disjoint alphabets, and single-character-shifted pairs — the `d-1`/`d-2` fence and the `lo==0`/`hi==d` edge writes are where a bug would hide; (2) timing at `n = 128, 512, 2048, 4096` to locate the crossover where the front is long enough to pay for itself, and to move the `n < 64` guard to wherever it actually lies; (3) timing on near-identical pairs, where the band should show near-linear behaviour and the speedup should be enormous rather than ~10×.

---

# VERDICT

The worm is a real mechanism, not a re-description of the textbook loop. Three of its features do concrete work that a row-wise Needleman–Wunsch cannot do:

1. **Crossing the cords before walking** (SEED 1) turns substitution scoring into one `vpcmpeqb` over a whole strip — the broken assumption, cashed directly.
2. **The free sideways slip** (SEED 3) is what makes (1) spendable: it dissolves the serial `dp[i][j-1]` chain into a one-lane unaligned load, and it forces the east cord to be stored backwards, which is the metaphor's "north cord / east cord" read literally and turns out to be the only layout where both character streams are contiguous.
3. **No ledger but the sand** (SEED 2) collapses the reference's (n+1)² table to three rolling fronts of O(n), removing all O(n²) memory traffic.

And the straight-grain tally gives a genuine runtime regime test — the same kernel is a 5-wide furrow on identical cords and the full tray on divergent ones, with no branch between two implementations.

Where I broke faith with the native, explicitly: **the doubling-back fixed point is gone.** "It kicks the old one flat... if not, it doubles back... done only when no junction still holds a mound it could improve" is Bellman–Ford, and on this DAG it is strictly wasted passes for an identical answer. I kept the curling as a single pass whose trail crosses its own two previous trails, and kept "a beaten mound is thrown away entirely" as the in-place max with no traceback. If the native's fixed point were load-bearing — e.g. under a scoring scheme with positive cycles — my kernel would be wrong; under this contract it is not.

**Honest risks, all addressed in the artifact rather than merely noted:** small `n` falls back to the row DP (`n < 64`); absent AVX2 the scalar tail still runs the whole band, and a wide band additionally defers to the row DP for locality; int16 lanes were dropped outright rather than shipped behind a guard I could not test. **No thread parallelism**, because the metaphor's unit of work is one front — tens to hundreds of cells at these sizes — and a barrier per front would cost more than it buys.

**The one thing I cannot stand behind is the number.** `PREDICTION: speedup_vs_dp = 10.0` is an instruction-count argument, and an unmeasured instruction-count argument is a hypothesis, not a result. The artifact is complete and ready to compile against the contract; until `alignment_bench` runs it, treat the 10× as what it is — a stated prediction with nothing under it yet.