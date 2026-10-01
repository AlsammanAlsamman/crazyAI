## MAPPING

**SEED 1 — "Two knot-cords laid crosswise over a sand-tray mark every place their symbols meet."**

| World object | Computational object |
|---|---|
| knot-cord running north (horse's length) | sequence `a`, laid along the **lane** axis of a 256-bit register |
| knot-cord running east (mountain's width) | sequence `b`, the outer (column) axis |
| the *single act of laying them crosswise* | one pass that materialises the crossing **all at once**: a striped profile `P[c][p]`, each slot a vector of 16 symbol-pair verdicts |
| a furrow-point where a symbol faces a symbol | one `(i,j)` cell; 16 of them live in one register slot |
| "marks **every** place they meet" (not visited, *marked*) | 16 symbol pairs judged per instruction (`add_epi16` of a profile slot) |

**Silent assumption broken: "one pair of positions is judged at a time."** The crossing is a single act, so the verdicts arrive 16 at a time.

**SEED 2 — "A single ink-worm walks the eight winds and curls back to test a junction twice."**

| World object | Computational object |
|---|---|
| a single worm, one cell wide, a stitch cut loose from the verses | one strip `H[0..L-1]` of mounds — **O(n) memory, no tray** — and exactly one thread |
| the mound it drops, "to the height its path has earned" | the candidate score for that junction |
| "reads what mound already sits there… kicks the old one flat" | in-place `max` relaxation; the beaten value is discarded, never stored |
| "keeps no ledger apart from the sand itself" | no queue, no priority structure, no n² table, no traceback |
| "doubles back — curling across ground already crossed" | the re-pass over the same strip carrying the north-face mound across a lane boundary |
| "done only when no junction anywhere still holds a mound it could improve" | fixpoint test: `testz(cmpgt(new,old) & real_mask)` |
| the eight winds (it may face backwards) | relaxation order is *not* required to be topological |

**Assumptions broken: "cells depend on above/left/diagonal computed in that order"** (order is replaced by convergence) and **"the whole grid must be filled in"** (one strip, beaten mounds thrown away).

**SEED 3 — "Crosses a furrow sideways, spending no sand, to let one cord slip ahead."**

| World object | Computational object |
|---|---|
| sideways crossing that spends nothing | the horizontal move made **literally cost 0** |
| "letting one cord slip a step ahead of the other" | re-origin the mounds by the east-edge drift: `H' = H + 2j` |
| the toll the slip used to cost | absorbed into the crossing marks (`match→+3`, `mismatch→+1`) |

Derivation: with `H = H' − 2j`, the recurrence `H=max(H_diag+s, H_up−2, H_left−2)` becomes
`H'[i][j] = max(H'[i−1][j−1] + s + 2, H'[i−1][j] − 2, H'[i][j−1] + 0)` — the sideways step is free, exactly as the native says. **Breaks "a slip can only be discovered by having already compared the position before it"**: the left term needs no chain of its own, so the only serial chain left is the vertical one.

## CHOSEN SEED

**SEED 1.** It is the only one of the three that breaks the preferred assumption *"one pair of positions is judged at a time"* — the crossing of the cords marks the whole lattice in one act rather than one kneeling visit per junction. SEED 2 supplies the traversal whose freedom from a fixed order is what *legalises* judging 16 pairs simultaneously (the worm curls back to fix what the simultaneous judgement got wrong), and SEED 3's free sideways step is folded into the mound heights.

Where this mechanism *lands*: a lattice marked 16-wide + a one-strip in-place relaxation + a curl-back that stops when no mound improves **is** Farrar's striped SIMD with the lazy-F corrective loop (SSW, parasail). Per step 4 I let the metaphor arrive there rather than inventing something new: the native's "doubles back until no junction improves" is precisely lazy-F, and lazy-F is validated production code. The offset form of SEED 3 is what lets linear gaps collapse the E-array away entirely.

## ASSUMPTION BROKEN

*One pair of positions is judged at a time* → 16 symbol pairs per instruction, from a crossing marked once.
*In that order* → the vertical cross-lane dependency is deliberately **left wrong** in the first pass and repaired by curling back to a fixpoint.
*The whole grid must be filled* → one strip, overwritten in place; a beaten mound is thrown away entirely.

## ARTIFACT

```c
#include <stdlib.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define SC_MATCH     1
#define SC_MISMATCH  (-1)
#define SC_GAP       (-2)

/* ============ the worm crawling with the grain ============
   One strip of sand, mounds 32 bits tall, no tray kept.
   Used when the tray is narrower than the lanes, or so wide
   that a mound would not fit a 16-bit measure.            */
static int worm_one_strip(int n, const char *restrict a, const char *restrict b)
{
    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int r;
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (int j = 0; j <= n; ++j) prev[j] = j * SC_GAP;
    for (int i = 1; i <= n; ++i) {
        const char ai = a[i - 1];
        int left = i * SC_GAP;
        cur[0] = left;
        for (int j = 1; j <= n; ++j) {
            int d = prev[j - 1] + (ai == b[j - 1] ? SC_MATCH : SC_MISMATCH);
            int u = prev[j] + SC_GAP;
            int l = left + SC_GAP;
            int best = d > u ? d : u;
            if (l > best) best = l;
            cur[j] = best;
            left = best;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

#if defined(__AVX2__)
/* One lane-step against the grain: the mound standing in lane w-1 becomes the
   mound standing north of lane w; lane 0 inherits the tray's blank edge (0). */
static inline __m256i wind_north(__m256i v)
{
    __m256i t = _mm256_permute2x128_si256(v, v, 0x08); /* [ 0 | v.lo ] */
    return _mm256_alignr_epi8(v, t, 14);               /* shift up 1 int16 */
}

/* ============ the cords laid crosswise, and one worm ============
   lane w, slot p  <->  a-index q = p + w*L        (W*L >= n)
   mounds are kept in the re-origined measure  H' = H + 2j, so that
   the sideways furrow-crossing costs nothing (SEED 3):
        H'[i][j] = max( H'[i-1][j-1] + mark ,   mark = 3 match / 1 mismatch
                        H'[i-1][j]   - 2   ,    the one permitted stumble
                        H'[i][j-1]         )    free sideways slip
   H'[0][j] = 0 for every j ; H'[i][0] = -2i ; answer = H'[n][n] - 2n.     */
static int worm_lattice(int n, const char *restrict a, const char *restrict b)
{
    enum { W = 16 };
    const int L = (n + W - 1) / W;
    const __m256i vGAP = _mm256_set1_epi16((short)SC_GAP);
    const __m256i vNEG = _mm256_set1_epi16((short)-30000);
    static const char sym[4] = { 'A', 'C', 'G', 'T' };
    int16_t code[256];
    int16_t out[W];
    __m256i *P, *H, *M;
    void *slab;
    int res, k;

    slab = malloc((size_t)(6 * L) * 32u + 64u);
    if (!slab) return worm_one_strip(n, a, b);
    P = (__m256i *)(((uintptr_t)slab + 31u) & ~(uintptr_t)31u);
    H = P + 4 * L;   /* the single strip of sand  */
    M = H + L;       /* real ground vs blank margin */

    for (k = 0; k < 256; ++k) code[k] = 0;
    code[(unsigned char)'C'] = 1;
    code[(unsigned char)'G'] = 2;
    code[(unsigned char)'T'] = 3;
    (void)sym;

    /* --- SEED 1: lay the cords crosswise once; every crossing is marked --- */
    {
        int16_t *p16 = (int16_t *)P;
        int16_t *m16 = (int16_t *)M;
        int16_t *h16 = (int16_t *)H;
        for (int w = 0; w < W; ++w) {
            for (int p = 0; p < L; ++p) {
                int q  = p + w * L;
                int cc = (q < n) ? (int)code[(unsigned char)a[q]] : -1;
                for (int c = 0; c < 4; ++c)
                    p16[(size_t)(c * L + p) * W + w] = (int16_t)((c == cc) ? 3 : 1);
                m16[(size_t)p * W + w] = (int16_t)((q < n) ? -1 : 0);
                h16[(size_t)p * W + w] = (int16_t)(-2 * (q + 1)); /* west edge */
            }
        }
    }

    for (int j = 1; j <= n; ++j) {
        const __m256i *vP = P + (size_t)code[(unsigned char)b[j - 1]] * L;
        /* mounds standing north-west, read before the strip is overwritten */
        __m256i vDiag = wind_north(H[L - 1]);   /* lane 0 = north edge = 0 */
        __m256i vUp   = wind_north(vNEG);       /* lane 0 = 0; others: no mound yet */
        for (int p = 0; p < L; ++p) {
            __m256i vL = H[p];                                      /* sideways, free */
            __m256i vH = _mm256_add_epi16(vDiag, vP[p]);            /* with the grain / diagonal */
            vH = _mm256_max_epi16(vH, vL);
            vH = _mm256_max_epi16(vH, _mm256_add_epi16(vUp, vGAP)); /* the stumble */
            H[p] = vH;          /* the beaten mound is thrown away entirely */
            vDiag = vL;
            vUp   = vH;
        }
        /* --- SEED 2: the worm doubles back over ground already crossed,
           testing each junction from its northern face, keeping only the
           taller mound, and stopping when no junction improves at all.
           At most W-1 curls can ever be needed (one per lane boundary). */
        for (k = 0; k < W; ++k) {
            __m256i vF = _mm256_add_epi16(wind_north(H[L - 1]), vGAP);
            int moved = 0;
            for (int p = 0; p < L; ++p) {
                __m256i h  = H[p];
                __m256i nh = _mm256_max_epi16(h, vF);
                __m256i d  = _mm256_and_si256(_mm256_cmpgt_epi16(nh, h), M[p]);
                if (_mm256_testz_si256(d, d)) break;  /* nothing improved here */
                H[p]  = nh;
                moved = 1;
                vF = _mm256_add_epi16(nh, vGAP);
            }
            if (!moved) break;                        /* no mound anywhere improves */
        }
    }

    _mm256_storeu_si256((__m256i *)out, H[(n - 1) % L]);
    res = (int)out[(n - 1) / L] - 2 * n;
    free(slab);
    return res;
}
#endif

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
#if defined(__AVX2__)
    /* The regime is read off the tray itself, at runtime:
       n < 32    : cords shorter than the lanes -> just crawl.
       n > 9000  : mounds (|H'| <= 3n) outgrow a 16-bit measure -> just crawl.
       otherwise : lay the cords crosswise and wake the worm.            */
    if (n >= 32 && n <= 9000) return worm_lattice(n, a, (const char *)b);
#endif
    return worm_one_strip(n, a, b);
}
```

PREDICTION: speedup_vs_dp = 11

*(stated before any measurement; no measurement tool was reachable in this session)*

Basis: the inner loop is 16 cells per iteration with a 2-cycle loop-carried chain (`add`→`max` through `vUp`) and ~7 µops of throughput, i.e. ≈2 cycles / 16 cells = 0.125 cycles/cell, against ≈2 cycles/cell for the reference table DP → ~16x ceiling; discounted to ~11x for the crossing-marking pass (~4n slots, ≈5% at n=1000), the curl-back's fixed ~7 ops/column, and loop overhead. Secondary win: working set is `6L` vectors (24 KB at n=2000) versus the reference's 4n² bytes (16 MB), so no streaming stores.

## MEASUREMENT

**Not measured.** `alignment_bench` / `alignment_contract` were unavailable in this session, so I am reporting an unmeasured artifact — plainly, as a failure of the measurement step, not as a success. What I did verify is correctness by hand-executing the vector kernel, lane by lane, against the reference recurrence:

| case | hand-traced kernel | reference DP | agree |
|---|---|---|---|
| n=2 `AC`/`CA` | `H'[2][2]=2` → `2−4 = −2` | −2 | ✓ |
| n=3 `AAA`/`AAA` | `H'[3][3]=9` → `9−6 = 3` | 3 | ✓ |
| n=3 `AAA`/`TTT` | `H'[3][3]=3` → `3−6 = −3` | −3 | ✓ |
| n=1 `A`/`C` | `1 − 2 = −1` | −1 | ✓ |

Also checked by derivation, not by hope:
- **Exactness of the re-origin**: `H = H' − 2j` is an identity, and `H'[0][j] = 0` for all `j`, so the north edge needs no per-column insert (it is the zero-fill of `wind_north`).
- **Range**: `H' ∈ [−2n, 3n]`; at `n = 9000` that is `[−18000, 27000]`, and the sentinel `−30000 − 2 = −30002` neither overflows int16 nor can ever beat a real mound. Hence the `n ≤ 9000` guard.
- **Curl-back termination and sufficiency**: each curl carries one lane boundary's true northern mound, so ≤ `W−1 = 15` curls always suffice; the early `break` is sound because if a mound did not improve, everything downstream of it in its own lane was already computed from that same value.
- **Blank margin cannot contaminate**: padding sits at high `p` within a lane, and if lane `w`'s top slot is padding then lane `w+1` is entirely padding, so margin mounds only ever flow upward into more margin or shift out of the register. The mask `M[p]` additionally keeps margin mounds from provoking curls — without it, small `n` (many all-margin lanes) would pay up to 15 wasted curls per column.

Falsifiable: if measured speedup is below ~4x at n≈1000–4000, my throughput model is wrong (most likely the curl-back is firing far more than the ~1 pass/column I expect, which the bench's divergence level would explain); if it exceeds ~16x, the reference is memory-bound and I under-credited the O(n) working set.

## VERDICT

The core of this kernel **is** the native's mechanism, not a textbook method wearing its clothes: the lattice is marked by one crosswise laying (16 pairs per instruction), there is exactly one worm on exactly one strip of sand, every mound is compared against the standing mound and the beaten one is thrown away, and the column ends only when the worm has curled back over its own trail until no junction anywhere improves. No band, no heuristic bandwidth, no prefix-scan, no second data structure. The place it lands — striped 16-lane int16 with a lazy corrective loop — is Farrar's validated algorithm, which is the *desired* outcome under step 4: the metaphor arrived at the known-good technique instead of inventing an untested cousin of it.

What the metaphor bought that I would not have written otherwise: the re-origin `H' = H + 2j`, forced by SEED 3's insistence that the sideways crossing *spends nothing*. It removes one `add` from the hot loop and makes the horizontal term a bare register — a real, if small, win derived from taking "spending no sand" literally.

**Threading: deliberately absent.** The native wakes *a single* ink-worm and explicitly refuses to "send a hooded memory to kneel at each crossing in turn." The metaphor's own unit of work is one worm on one strip, and columns are strictly serial, so OpenMP would require inventing a second worm the native ruled out. No thread parallelism was added.

**Stated risks, each addressed rather than left hanging:**
- *16-bit mounds overflow on a very wide tray* → guarded at `n ≤ 9000` (proved bound `|H'| ≤ 3n`), falling back to the 32-bit one-strip crawl.
- *Vector setup dominates a tiny tray* → guarded at `n ≥ 32`, same fallback.
- *The curl-back could in principle run 15 passes per column* → bounded, not merely hoped away: worst case is ≈`97L` vector ops per column ≈ `6n²` total, against the reference's ≈`8n²` scalar ops at comparable per-op throughput, so even the pathological board degrades to rough parity and cannot lose. The margin mask removes the one input class (small `n`) where the pathology was actually reachable.
- *Regime recognition* (step 5): the two regimes named in `known_way` — full-width DP versus a bounded-score-range SIMD variant — are distinguished at runtime by the mound-height test `n ≤ 9000`, with the crawl as the other path; and within the SIMD path the curl-back is itself a runtime detector of the gap-heavy versus diagonal-heavy board, curling exactly as many times as the sand demands.

The honest caveat stands at the top of MEASUREMENT: this artifact is reasoned and hand-verified, not benchmarked, because the bench was not reachable here.