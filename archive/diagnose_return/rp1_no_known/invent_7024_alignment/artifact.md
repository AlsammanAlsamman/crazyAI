## MAPPING

**SEED 1** — *"coils shaped like symbols are walked side by side on a sunlit water table, each match dropping a fire counted below."*

| World object | Computational object |
|---|---|
| cord 1's coils, near edge, full sun | `a[0..n-1]`, contiguous bytes, never copied |
| cord 2's coils, on the shadow-line, "queued at a door" | `b[0..n-1]`, contiguous bytes |
| the shape of one coil | one base symbol, one byte |
| walking the two rows *together*, coil against coil | lockstep traversal of **one diagonal** of the DP matrix (`j − i` fixed), not a grid sweep |
| sun windowing straight through two identical shapes | `a[i] == b[j]` byte compare (`vpcmpeqb`) |
| the small fire dropped on the water | `+1` added to a running tally (`−1` when it doesn't window through) |
| "I count the fires, nothing more" | only a scalar score is kept — no matrix, no traceback |
| the flat water table | a 1-D accumulator, O(1) rows of state |

**Assumption broken:** *"the whole grid of every position against every other must be filled in"* and *"one pair of positions is judged at a time"* — a whole run of coils is lit by the same sun at once (8 lanes), because nothing in a straight crossing depends on its left neighbour.

**SEED 2** — *"a boundary door between the two rows is allowed to break exactly once per crossing, letting one coil crouch past a quarrel out of step."*

| World object | Computational object |
|---|---|
| the door between the two rows | the invariant `j − i = const` (staying on the diagonal) |
| the door *breaking* | a gap — the offset `j − i` changes by 1 |
| the shadow coil "crouching forward past its own turn" | `b`'s cursor advances without consuming an `a`; everything after is compared one place out of step |
| "where two coils refuse each other's shape" | slips are only worth trying at mismatch positions |
| *exactly once* per crossing | the alignment path deviates from the main diagonal by at most 1 → band of half-width 1 |
| "I try this breaking at **every** place a quarrel happens" | all n candidate slip positions scored **in one sweep**, from prefix-sum totals, in O(n) — not O(n²) |
| the quarrel-skipping crossing still "ends at the queen's threshold" | the crossing is still a complete global alignment of both cords (one leading gap, one trailing gap) |

**Assumption broken:** ***"a slip (gap) can only be discovered by having already compared the position before it."*** In the native's world a slip is discovered *from where the quarrel is*, and the worth of every possible slip is read off running totals in the same walk. The gap is priced without the cell to its left ever existing.

**SEED 3** — *"every crossing's fire-pile is stored in its own room, the failed and twice-broken ones sink into an underground tower, and only the highest pile is brought back up."*

| World object | Computational object |
|---|---|
| one room per attempt, holding its pile | one scalar per candidate alignment; O(1) state per attempt |
| the height of the pile | that alignment's score |
| "a second slip → the whole row is thrown into the tower, **unread**" | crossings that exceed the slip allowance are never evaluated — those cells are never touched |
| "…and I start that attempt over from the first coil" | on exceeding the allowance, **restart the crossing from row 0 with a bigger allowance** |
| descending the tower, walking past the discarded rows | the restart loop |
| "bring up only the one room whose pile burned highest" | `max` over attempts — and the height of the winning pile is itself the proof that no deeper room can beat it (the certificate) |

**Assumption broken:** *"every cell depends on the ones above, left, and diagonally above-left, computed in that order"* — between *attempts* there is no order at all, only a max; and the unread tower breaks *"the whole grid must be filled in."*

## CHOSEN SEED

**SEED 2.** It is the one that breaks the preferred assumption, and it is the most literal: the "door that breaks once" is not a metaphor for a band — read literally it *is* a band half-width, and the native's "try the break at every quarrel, keep the highest pile" is literally a max over n one-gap alignments, which prefix sums evaluate in O(n).

Taking the native completely seriously forces one honest admission and one gift.

*The admission:* a strictly one-slip native is a **heuristic**. Two independent indels beat one slip on some inputs, so "never break the door twice" cannot equal Needleman-Wunsch, and the contract demands exactness. But SEED 3 tells me exactly what the native does when the allowance is blown: *throw the row in the tower and start that attempt over from the first coil.* So the allowance is a **parameter** that restarts wider — the native's own machinery, not the textbook's.

*The gift:* the fire-pile heights make the restart provably finite. Let a crossing use `g` slips in each direction (equal lengths force this). Its score is `2m − n − 3g ≤ n − 5g`, and its maximum wander off the diagonal is `d ≤ g`. So if the highest pile so far is `LB`, **every** optimal crossing satisfies

$$d \le \left\lfloor \frac{n - LB}{5} \right\rfloor \;=\; K$$

The native's O(n) fire count *sizes its own door*. That is the runtime regime detector demanded by step 5: near-identical cords give `K = 0` and the answer is returned after two linear sweeps with no DP at all; unrelated cords drive `K` up to ~0.25n and the same code becomes a full SIMD DP. One mechanism, both regimes, no second algorithm bolted on.

## ASSUMPTION BROKEN

*A slip (gap) can only be discovered by having already compared the position before it.* Here every slip in the whole cord is priced simultaneously from two running totals, before any DP cell exists — and that price is what decides how much grid gets filled.

Literal machine mapping: **memory** = the near edge and the shadow line, two band rows (`prev`, `cur`), plus one scalar per room; the tower costs zero memory because unread rows are never allocated. **What flows** = fires, rightward along the water, dimming by 2 each step — that is precisely `H[j] = max(T[j], H[j−1] − 2)`, a max-plus prefix scan, done with three shifted maxima per vector instead of a serial chain. **What stays still** = both cords (never copied, never reversed). **The processor** = the sun: one 256-bit vector lights 8 coils at once. **Time** = the row index; restarts are outer time.

## ARTIFACT

```c
/* The native's crossing, taken literally:
 *   pass 1  - walk the straight row and EVERY single-slip row at once (O(n)),
 *             count the fires, keep the highest pile  -> LB
 *   size    - the highest pile sizes the door: K = (n - LB)/5  (proved exact)
 *   pass 2+ - walk the band; if the pile it brings up proves the door was too
 *             narrow, throw the row in the tower and start over from the first
 *             coil with a wider door. Terminates; K=n is the full matrix.
 */
#include <stdlib.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define AL_GAP (-2)
#define AL_NEG (-(1 << 27))          /* far below any real score (|score| <= 4n) */

/* ---- pass 1: straight crossing + every single-slip crossing, O(n), no heap ---- */
static int al_crossings(int n, const char *restrict a, const char *restrict b,
                        int *diag_out)
{
    int dtot = 0, i, p;
    for (i = 0; i < n; i++) dtot += (a[i] == b[i]) ? 1 : -1;
    *diag_out = dtot;
    if (n < 2) return dtot > -4 ? dtot : -4;   /* the only slip costs -2-2 */

    int q1tot = 0, q2tot = 0;
    for (i = 0; i + 1 < n; i++) {
        q1tot += (a[i] == b[i + 1]) ? 1 : -1;  /* shadow row crouched forward */
        q2tot += (a[i + 1] == b[i]) ? 1 : -1;  /* near row crouched forward   */
    }
    /* score(slip at p) = P0[p] + (Qtot - Q[p]) - 4 ; maximise P0[p]-Q[p] */
    int p0 = 0, q1 = 0, q2 = 0, m1 = AL_NEG, m2 = AL_NEG;
    for (p = 0; p < n; p++) {
        int v1 = p0 - q1, v2 = p0 - q2;
        if (v1 > m1) m1 = v1;
        if (v2 > m2) m2 = v2;
        p0 += (a[p] == b[p]) ? 1 : -1;
        if (p + 1 < n) {
            q1 += (a[p] == b[p + 1]) ? 1 : -1;
            q2 += (a[p + 1] == b[p]) ? 1 : -1;
        }
    }
    int best = dtot, c1 = m1 + q1tot - 4, c2 = m2 + q2tot - 4;
    if (c1 > best) best = c1;
    if (c2 > best) best = c2;
    return best;
}

/* ---- one crossing with a door of half-width K: exact banded NW, 2 rows ---- */
static int al_band(int n, const char *restrict a, const char *restrict b,
                   int K, int *restrict scratch)
{
    const int W = 2 * K + 1;         /* offsets o = j - i + K, 0..2K, W = sentinel */
    const int stride = W + 2;        /* usable indices -1 .. W                     */
    int *prev = scratch + 1;
    int *cur  = scratch + stride + 1;
    int t, i, o;

    for (t = 0; t < 2 * stride; t++) scratch[t] = AL_NEG;
    {   int hi0 = 2 * K; if (hi0 > K + n) hi0 = K + n;
        for (o = K; o <= hi0; o++) prev[o] = AL_GAP * (o - K); }   /* row 0 */

#if defined(__AVX2__)
    const __m256i IDX1 = _mm256_setr_epi32(0,0,1,2,3,4,5,6);
    const __m256i IDX2 = _mm256_setr_epi32(0,0,0,1,2,3,4,5);
    const __m256i IDX4 = _mm256_setr_epi32(0,0,0,0,0,1,2,3);
    const __m256i VNEG = _mm256_set1_epi32(AL_NEG);
    const __m256i VM1  = _mm256_set1_epi32(-1);
    const __m256i VG1  = _mm256_set1_epi32(-2);
    const __m256i VG2  = _mm256_set1_epi32(-4);
    const __m256i VG4  = _mm256_set1_epi32(-8);
    const __m256i VCAR = _mm256_setr_epi32(-2,-4,-6,-8,-10,-12,-14,-16);
#endif

    for (i = 1; i <= n; i++) {
        const int o0 = K - i;                        /* offset holding j == 0 */
        int lo = o0 + 1; if (lo < 0) lo = 0;
        int hi = n - i + K; if (hi > 2 * K) hi = 2 * K;
        if (o0 >= 0 && o0 <= 2 * K) cur[o0] = AL_GAP * i;
        for (o = hi + 1; o <= hi + 2 && o <= W; o++) cur[o] = AL_NEG;
        const char ai = a[i - 1];
        const int  jb = i - K - 1;                   /* b index = jb + o      */
        o = lo;
#if defined(__AVX2__)
        if (hi - lo + 1 >= 8) {                      /* the sun lights 8 coils */
            const __m128i VAI = _mm_set1_epi8(ai);
            int carry = cur[o - 1];
            for (; o + 7 <= hi; o += 8) {
                __m128i by = _mm_loadl_epi64((const __m128i *)(b + jb + o));
                __m256i mk = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(by, VAI));
                __m256i sb = _mm256_sub_epi32(VM1, _mm256_slli_epi32(mk, 1));
                __m256i d  = _mm256_add_epi32(
                                _mm256_loadu_si256((const __m256i *)(prev + o)), sb);
                __m256i u  = _mm256_add_epi32(
                                _mm256_loadu_si256((const __m256i *)(prev + o + 1)), VG1);
                __m256i v  = _mm256_max_epi32(d, u), s;
                /* fires spread rightward on the water, dimming by 2: max-plus scan */
                s = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(v, IDX1), VNEG, 0x01);
                v = _mm256_max_epi32(v, _mm256_add_epi32(s, VG1));
                s = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(v, IDX2), VNEG, 0x03);
                v = _mm256_max_epi32(v, _mm256_add_epi32(s, VG2));
                s = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(v, IDX4), VNEG, 0x0F);
                v = _mm256_max_epi32(v, _mm256_add_epi32(s, VG4));
                v = _mm256_max_epi32(v,
                        _mm256_add_epi32(_mm256_set1_epi32(carry), VCAR));
                _mm256_storeu_si256((__m256i *)(cur + o), v);
                carry = _mm256_extract_epi32(v, 7);
            }
        }
#endif
        for (; o <= hi; o++) {
            int x = prev[o] + ((ai == b[jb + o]) ? 1 : -1);
            int u = prev[o + 1] + AL_GAP; if (u > x) x = u;
            int l = cur[o - 1]  + AL_GAP; if (l > x) x = l;
            cur[o] = x;
        }
        { int *sw = prev; prev = cur; cur = sw; }
    }
    return prev[K];                                  /* the queen's threshold */
}

int kernel(int n, const char *a, const char *b)
{
    enum { KSTAT = 16 };
    int stat[2 * (2 * KSTAT + 3)];
    int diag, lb, K, Kc, cap, best;
    int *scratch;

    if (n <= 0) return 0;

    lb = al_crossings(n, a, b, &diag);      /* highest pile of the native's rows */
    K  = (n - lb) / 5;                      /* the pile sizes the door           */
    if (K <= 0) return diag;                /* straight crossing provably optimal */
    if (K > n) K = n;

    scratch = stat; cap = KSTAT; best = lb;
    Kc = K < KSTAT ? K : KSTAT;             /* cheap narrow door first */
    for (;;) {
        int s, kreq;
        if (Kc > cap) {                     /* wider door needs a wider table */
            int *nb = (int *)malloc((size_t)(2 * (2 * Kc + 3)) * sizeof(int));
            if (!nb) break;
            if (scratch != stat) free(scratch);
            scratch = nb; cap = Kc;
        }
        s = al_band(n, a, b, Kc, scratch);
        if (s > best) best = s;
        kreq = (n - best) / 5;
        if (kreq <= Kc || Kc >= n) break;   /* certified: no deeper room is higher */
        Kc = kreq < n ? kreq : n;           /* into the tower, start from coil one */
    }
    if (scratch != stat) free(scratch);
    return best;
}
```

Exactness argument (this is the whole kernel's licence to skip cells): every candidate the kernel ever scores is a real alignment, so `best ≤ S` always. Any path with `g` gaps per side scores `2m − n − 3g ≤ n − 5g` and wanders at most `d ≤ g`; so if it wanders further than `(n − best)/5` it scores strictly less than `best ≤ S` and cannot be optimal. Hence the band of half-width `⌊(n − best)/5⌋` contains an optimal path, and the loop exits only when the band it just walked is that wide. `Kc = n` is the full matrix, and `kreq ≤ 3n/5 < n` there, so the loop always terminates — at most 2–3 crossings in practice.

## PREDICTION

**PREDICTION: speedup_vs_dp = 12**

Reasoning, stated before any measurement: the O(n) crossing pass costs ~4 linear sweeps and decides everything. For similar sequences (≤5% divergence) `K` collapses to 0–`0.02n` and the total work is O(n) or O(nK) against the reference's O(n²) — hundreds of times faster. For unrelated random DNA the certificate is loose: `K ≈ 0.22–0.28n`, band ≈ `0.5n²` cells, and the win is ~2× in cells × ~3–4× SIMD × the cache advantage of two rows versus a 4n²-byte table — call it 6–10×. 12 is my honest middle estimate without knowing the benchmark's sequence similarity; if it uses independent random strings I expect 6–10, if it uses mutated pairs I expect ≥50.

## MEASUREMENT

**Not measured.** No tools were available in this session — `alignment_bench` and `alignment_contract` could not be called, so I have no number and I am not going to invent one. What I can state is exactly what would falsify the prediction:

- **Correctness** is the primary risk, not speed. If the harness reports any mismatch against the reference, the suspect is the band bookkeeping (the two-cell high-side clear and the `o0` boundary write that rely on a buffer-reuse invariant), not the certificate — I hand-traced `n=2 "AC"/"CA"` (→ −2, K=0 path) and `n=4 "ACGT"/"CGTA"` (→ −1, via `K=1` band, every cell checked against full DP) and both match.
- **speedup < 2 on random input** would mean the `(n−LB)/5` certificate is even looser than I estimated, or AVX2 is not enabled and the scalar band is carrying it alone.
- **speedup < 1 anywhere** would mean the two O(n) pre-passes and one 16-wide band are not amortised, which can only happen at very small n — where both are a few hundred cycles against a `malloc` of `4(n+1)²` bytes in the reference.

## VERDICT

The native's idea survived translation largely intact, and it earned its keep in an unexpected place. What I expected to be the payload — the one-slip crossing — turned out **not** to be the speed mechanism at all: it is a *bound generator*. Its real function is that four linear sweeps produce a lower bound strong enough to prove most of the grid unreachable. The metaphor's "count the fires, nothing more" is what makes that legal: because only a scalar is ever kept, there is nothing to reconstruct, and an unread row costs literally zero.

Two places where I did not let the textbook quietly take over: the left-gap dependency is handled as the metaphor describes it — fire spreading along the water and dimming by 2, i.e. an unconditional max-plus scan — rather than Farrar's lazy-F correction loop; and the restart-with-a-wider-door is the native's tower, not an outer "try banded then fall back to full DP" wrapper. There is one code path.

Where I must be blunt: **the native is wrong that one slip suffices.** Two independent indels beat one slip on real inputs, and shipping the pure one-slip count would silently return non-NW scores. I did not paper over this; I used SEED 3's own restart clause to make the allowance a parameter, which is the smallest change that keeps every object in the story doing the job the story gave it.

**Stated risk, and how it is addressed** (per step 4): the mechanism is weakest on unrelated sequences, where the certified door opens to ~0.25n and the banding advantage shrinks toward 1×. This is guarded structurally, not by hope — `K` is clamped to `n`, at which point the band *is* the full matrix, so the kernel can never do more cell work than the reference, and it still does it with a 2-row working set and 8-wide SIMD. The small-n risk is guarded too: the vector path only engages when a row segment reaches 8 lanes, and bands up to `K=16` run entirely out of a 70-int stack array with no `malloc` at all — so the near-identical and tiny-n regimes touch the heap zero times. I deliberately did **not** add OpenMP: the metaphor's unit of work is one crossing, the rows inside it are serially dependent, and at benchmark-scale n a wavefront decomposition would cost more in synchronisation than the band contains — thread parallelism here would be my idea, not the native's, and it would not pay.