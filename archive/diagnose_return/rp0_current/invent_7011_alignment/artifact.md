## MAPPING

### SEED 1 — "two boughs hung from crossed rows of lamp-posts → a lattice of spark-cells"

| World object | Problem object |
|---|---|
| bough, cut from a note-tree | input sequence `a` (resp. `b`), length `n` over {A,C,G,T} |
| row of lamp-posts running down | index `i` into `a` |
| row of lamp-posts running across | index `j` into `b` |
| lattice of unlit windows | the `(n+1)×(n+1)` DP matrix — **as a coordinate system only, never as storage** |
| "I light each cell only when I need it" | lazy evaluation: only cells that an optimal path can reach are ever computed |
| spark's colour chosen *by the two crossing symbols*, not by me | substitution score `s = (a[i-1]==b[j-1] ? +1 : -1)`, a pure function of the two characters — no history, so it can be produced 16-at-a-time by a byte compare |
| the three colours: match / slip-out / slip-in | the three candidates: diagonal, up (gap in `b`), left (gap in `a`) |

**Assumption broken:** *"the whole grid of every position against every other must be filled in."* Lamps are lit on demand, not en masse.

### SEED 2 — "fold the sheet along its slanting middle; cells of equal row-and-column sum press flush into one crease"

| World object | Problem object |
|---|---|
| the slanting middle, corner to corner | the anti-diagonal `i + j = d` |
| one crease | one wavefront: the vector of all cells with `i+j = d` |
| cells pressed *flush* against each other in a crease | cells of equal `i+j` are mutually **independent** → they are lanes of one SIMD register, not steps of a loop |
| "the note itself is already folded in the hour between the note-trees" | the fold is intrinsic, not imposed: every dependency edge of the recurrence decreases `i+j` by exactly 1 or 2, so `i+j` is the natural time coordinate |
| three creases pressed into one thickness | crease `d` is written from crease `d-1` (read twice, at offsets 0 and −1) and crease `d-2` (offset −1) |
| walking the fold forward | `d = 1 … 2n` |
| the last crease holding a single spark at the far corner | `dp[n][n]`, the only cell with `i+j = 2n` |

**Assumption broken:** **"every cell depends on the ones above, to the left, and diagonally above-left, computed in that order."** The fold keeps the *dependencies* and destroys the *order*: cells are produced in `i+j` order, and within one crease nothing depends on anything else. It simultaneously breaks "one pair of positions is judged at a time."

### SEED 3 — "read three stacked layers through the crease at once; let the spent bottom layer drift off"

| World object | Problem object |
|---|---|
| reading three layers *at once*, not one after another | `cur[i] = max(prev2[i-1] + s, max(prev1[i-1], prev1[i]) + GAP)` — one branch-free vector expression, no serial `left` chain |
| "the layer beneath tells me what agreeing straight through would have cost" | `prev2` (crease `d−2`) + match/mismatch |
| "the layer beside tells me what one slip would have cost" | `prev1` (crease `d−1`) + gap; both up and left carry the same `GAP`, so one `max` serves both |
| the spent bottom layer drifts off into the gradient | buffer rotation — only two creases are retained: **O(n) memory, zero DP table** |
| "the note has no ground to store it on / I am not a forest that hoards its own marsh" | no `(n+1)²` array → no 67 MB of DRAM traffic at n=4096; the whole working set is 3 vectors of ~n shorts, L1/L2 resident |
| "one slip allowed", the boughs cut to the same length | equal length ⇒ every slip-out is answered by a slip-in ⇒ a path deviating `D` off the fold's spine pays ≥ `2D` gaps ⇒ its score ≤ `n − 5D` ⇒ a **provable reach** `D ≤ (n − L)/5` for any achievable `L` |
| "I wait only as long as it takes each spark to catch from its neighbours, never longer" | dependency distance is exactly one crease: no barriers, no threads, no synchronisation beyond the rotation |

**Assumptions broken:** "the whole grid must be filled in" (two creases exist at a time), and "a slip can only be discovered by having already compared the position before it" (the slip candidate arrives sideways from the crease beside, at the same instant as the match candidate).

---

## CHOSEN SEED

**SEED 2 — the fold.** It is the only one of the three that breaks the *preferred* assumption (dependency order), and its mapping is the most literal: "cells whose row and column add to the same sum press flush" is, word for word, the anti-diagonal wavefront. SEED 1 supplies the band (light only what you need) and SEED 3 supplies the rotation (throw the spent crease away); both are carried by the fold rather than competing with it.

**Literal mapping of the world onto machine:**

- **memory** = the two creases still pressed together (`prev1`, `prev2`) plus the one being written (`cur`) — three arrays of `n+48` elements, indexed by `i`. Nothing else is memory; the lattice does not exist.
- **what flows** = crease values, sliding one index per fold. Because `j = d − i`, walking a crease *increases* `i` and *decreases* `j`; storing `b` reversed makes both character streams walk forward together, so `a[i-1]` and `brev[n-d+i]` are contiguous loads.
- **what stays still** = the two boughs (read-only, padded copies) and the geometry `ilo(d)`, `ihi(d)`.
- **processor** = one SIMD lane per spark: 16 sparks of a crease catch in one instruction (int16), 8 for long boughs (int32).
- **time** = `d`, the crease number. One tick = one fold. There are `2n` ticks, never `n²`.
- **the gradient with no ground** = the absence of a DP table. The spent crease is not stored anywhere; it is overwritten by the next `cur`.

**Per instruction 4:** this literal reading lands exactly on two *validated, published* techniques rather than on an invention — Wozniak's 1997 anti-diagonal SIMD alignment (the fold) and Ukkonen/KSW2-style score-bounded banding (lighting only what is needed). I let the metaphor arrive there instead of steering around it.

**Per instruction 5 — the two regimes the native must recognise.** `known_way` names both a full `O(n²)` table and a banded variant, so the native probes at runtime: one cheap pass counts how many lamp-posts already face each other (`m` = agreements on the spine), giving the achievable score `L = 2m − n` and the reach `W = ⌊2(n−m)/5⌋`. Narrow regime (`W < n`): the fold is clipped to `|i−j| ≤ W`, with unreachable rims stamped `NEG` — provably exact. Wide regime (`W ≥ n`): `W` is clamped to `n`, which makes the clip non-binding and the *same* code path becomes the full fold. One kernel, two regimes, no duplicated recurrence. And the risk my own verdict names — setup cost on tiny inputs — is guarded: `n < 96` (or no AVX2, or any allocation failure) falls back to the flat walker, a plain row DP.

---

## ASSUMPTION BROKEN

> **"every cell of the comparison depends on the ones above, to the left, and diagonally above-left, computed in that order"**

The dependencies are kept exactly; the *order* is replaced by `i+j` (the fold). Consequences: the serial `left` chain that makes the reference unvectorisable disappears, 16 cells are judged per instruction, and only two creases need to exist — so the `O(n²)` table is destroyed as well ("the whole grid must be filled in", "one pair at a time", and "a slip needs the previous comparison first" all fall with it).

---

## ARTIFACT

```c
/* The fold: anti-diagonal (i+j) wavefront Needleman-Wunsch.
 * Exact match to the reference DP (match +1, mismatch -1, gap -2, global).
 *   - creases: all cells with i+j = d are independent -> SIMD lanes.
 *   - two creases retained (O(n) memory, no DP table).
 *   - lit on demand: reach W = floor(2(n-m)/5) from the achievable score
 *     L = 2m-n; any path deviating D off the spine pays >= 2D gaps (the
 *     boughs are equal length), so its score <= n - 5D, so D <= (n-L)/5.
 *   - W clamped to n makes the clip non-binding => same code is the full fold.
 *   - fallbacks: no AVX2 / n < 96 / malloc failure -> plain row DP.
 */
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define NW_MATCH     1
#define NW_MISMATCH (-1)
#define NW_GAP      (-2)

/* ---- the flat walker: reference-identical row DP, O(n) memory ---- */
static int nw_rows(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    int *base = (int *)malloc((size_t)2 * (size_t)(n + 1) * sizeof(int));
    if (!base) return 0;
    int *prev = base, *cur = base + (n + 1);
    for (int j = 0; j <= n; j++) prev[j] = j * NW_GAP;
    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = i * NW_GAP;
        for (int j = 1; j <= n; j++) {
            int best = prev[j - 1] + (ai == b[j - 1] ? NW_MATCH : NW_MISMATCH);
            int up = prev[j] + NW_GAP;
            int lf = cur[j - 1] + NW_GAP;
            if (up > best) best = up;
            if (lf > best) best = lf;
            cur[j] = best;
        }
        int *t = prev; prev = cur; cur = t;
    }
    int r = prev[n];
    free(base);
    return r;
}

#if defined(__AVX2__)
/* ---- the fold, 16 sparks per crease (n <= 4000: |score| fits int16) ---- */
static int nw_fold16(int n, const unsigned char *restrict A,
                     const unsigned char *restrict R,
                     int W, short *buf, int stride)
{
    const short NEG = -20000;            /* < -(4n+1): can never win a max */
    short *p2 = buf, *p1 = buf + stride, *cu = buf + 2 * stride;
    for (int t = 0; t < 3 * stride; t++) buf[t] = NEG;
    p1[0] = 0;                           /* crease 0 holds dp[0][0] */
    const __m256i vgap = _mm256_set1_epi16((short)NW_GAP);
    const __m256i v2   = _mm256_set1_epi16(2);
    const __m256i v1   = _mm256_set1_epi16(1);
    const int dmax = 2 * n;
    for (int d = 1; d <= dmax; d++) {
        int ilo = d - n; if (ilo < 0) ilo = 0;
        int c = (d - W + 1) >> 1;   if (c > ilo) ilo = c;   /* ceil((d-W)/2) */
        int ihi = (d < n) ? d : n;
        int f = (d + W) >> 1;       if (f < ihi) ihi = f;   /* floor((d+W)/2) */
        int lo = (ilo < 1) ? 1 : ilo;
        int hi = (ihi < d - 1) ? ihi : d - 1;
        if (hi >= lo) {
            const unsigned char *pa = A + (lo - 1);
            const unsigned char *pb = R + (n - d + lo);
            const short *q2  = p2 + (lo - 1);   /* crease d-2, diag  */
            const short *q1u = p1 + (lo - 1);   /* crease d-1, up    */
            const short *q1l = p1 + lo;         /* crease d-1, left  */
            short *out = cu + lo;
            int len = hi - lo + 1;
            for (int k = 0; k < len; k += 16) {
                __m128i ca = _mm_loadu_si128((const __m128i *)(pa + k));
                __m128i cb = _mm_loadu_si128((const __m128i *)(pb + k));
                __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
                __m256i sc = _mm256_sub_epi16(_mm256_and_si256(eq, v2), v1);
                __m256i vd = _mm256_add_epi16(
                    _mm256_loadu_si256((const __m256i *)(q2 + k)), sc);
                __m256i vu = _mm256_loadu_si256((const __m256i *)(q1u + k));
                __m256i vl = _mm256_loadu_si256((const __m256i *)(q1l + k));
                __m256i vg = _mm256_add_epi16(_mm256_max_epi16(vu, vl), vgap);
                _mm256_storeu_si256((__m256i *)(out + k),
                                    _mm256_max_epi16(vd, vg));
            }
        }
        /* rim of the crease, stamped after the sparks (order matters) */
        if (ilo == 0) cu[0] = (short)(NW_GAP * d);   /* dp[0][d] */
        if (ihi == d) cu[d] = (short)(NW_GAP * d);   /* dp[d][0] */
        if (ilo >= 1) cu[ilo - 1] = NEG;
        cu[ihi + 1] = NEG;
        short *t = p2; p2 = p1; p1 = cu; cu = t;     /* spent crease drifts off */
    }
    return (int)p1[n];                               /* crease 2n, cell (n,n) */
}

/* ---- the same fold, 8 sparks per crease, for long boughs ---- */
static int nw_fold32(int n, const unsigned char *restrict A,
                     const unsigned char *restrict R,
                     int W, int *buf, int stride)
{
    const int NEG = -(1 << 24);
    int *p2 = buf, *p1 = buf + stride, *cu = buf + 2 * stride;
    for (int t = 0; t < 3 * stride; t++) buf[t] = NEG;
    p1[0] = 0;
    const __m256i vgap = _mm256_set1_epi32(NW_GAP);
    const __m256i v2   = _mm256_set1_epi32(2);
    const __m256i v1   = _mm256_set1_epi32(1);
    const int dmax = 2 * n;
    for (int d = 1; d <= dmax; d++) {
        int ilo = d - n; if (ilo < 0) ilo = 0;
        int c = (d - W + 1) >> 1;   if (c > ilo) ilo = c;
        int ihi = (d < n) ? d : n;
        int f = (d + W) >> 1;       if (f < ihi) ihi = f;
        int lo = (ilo < 1) ? 1 : ilo;
        int hi = (ihi < d - 1) ? ihi : d - 1;
        if (hi >= lo) {
            const unsigned char *pa = A + (lo - 1);
            const unsigned char *pb = R + (n - d + lo);
            const int *q2  = p2 + (lo - 1);
            const int *q1u = p1 + (lo - 1);
            const int *q1l = p1 + lo;
            int *out = cu + lo;
            int len = hi - lo + 1;
            for (int k = 0; k < len; k += 8) {
                __m128i ca = _mm_loadl_epi64((const __m128i *)(pa + k));
                __m128i cb = _mm_loadl_epi64((const __m128i *)(pb + k));
                __m256i eq = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(ca, cb));
                __m256i sc = _mm256_sub_epi32(_mm256_and_si256(eq, v2), v1);
                __m256i vd = _mm256_add_epi32(
                    _mm256_loadu_si256((const __m256i *)(q2 + k)), sc);
                __m256i vu = _mm256_loadu_si256((const __m256i *)(q1u + k));
                __m256i vl = _mm256_loadu_si256((const __m256i *)(q1l + k));
                __m256i vg = _mm256_add_epi32(_mm256_max_epi32(vu, vl), vgap);
                _mm256_storeu_si256((__m256i *)(out + k),
                                    _mm256_max_epi32(vd, vg));
            }
        }
        if (ilo == 0) cu[0] = NW_GAP * d;
        if (ihi == d) cu[d] = NW_GAP * d;
        if (ilo >= 1) cu[ilo - 1] = NEG;
        cu[ihi + 1] = NEG;
        int *t = p2; p2 = p1; p1 = cu; cu = t;
    }
    return p1[n];
}
#endif /* __AVX2__ */

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
#if defined(__AVX2__)
    if (n < 96) return nw_rows(n, a, b);   /* guard: setup dominates when tiny */

    /* regime probe: how far off the spine can an optimal path stray?
       L = 2m - n is achievable (gap-free); a path deviating D pays >= 2D gaps
       and scores <= n - 5D, so D <= (n - L)/5 = 2(n - m)/5.                  */
    int m = 0;
    for (int i = 0; i < n; i++) m += (a[i] == b[i]);
    int W = (2 * (n - m)) / 5;
    if (W < 8) W = 8;                      /* keep creases overlapping */
    if (W > n) W = n;                      /* wide regime: clip non-binding */

    const int stride = n + 48;
    unsigned char *cbuf = (unsigned char *)malloc((size_t)2 * (size_t)(n + 64));
    if (!cbuf) return nw_rows(n, a, b);
    unsigned char *A = cbuf, *R = cbuf + (n + 64);
    memcpy(A, a, (size_t)n);
    memset(A + n, 0, 64);
    for (int i = 0; i < n; i++) R[i] = (unsigned char)b[n - 1 - i];
    memset(R + n, 1, 64);                  /* padding lanes are discarded */

    int result;
    if (n <= 4000) {
        short *sb = (short *)malloc((size_t)3 * (size_t)stride * sizeof(short));
        if (!sb) { free(cbuf); return nw_rows(n, a, b); }
        result = nw_fold16(n, A, R, W, sb, stride);
        free(sb);
    } else {
        int *ib = (int *)malloc((size_t)3 * (size_t)stride * sizeof(int));
        if (!ib) { free(cbuf); return nw_rows(n, a, b); }
        result = nw_fold32(n, A, R, W, ib, stride);
        free(ib);
    }
    free(cbuf);
    return result;
#else
    return nw_rows(n, a, b);
#endif
}
```

---

## PREDICTION

**PREDICTION: speedup_vs_dp = 10.0**

Reasoning, stated before any measurement: the reference's inner loop carries a serial `cur[j-1]` dependency (≈2.5–3 cycles/cell, unvectorisable) and streams an `(n+1)²` table through DRAM (67 MB at n=4096). The fold replaces that with ~3–4 cycles per 16 cells out of L1 — ≈11× from lanes alone — times ≈1.6× from the band at random-DNA identity (`m≈n/4 ⇒ W≈0.3n`), minus ragged short creases at both corners and `2n` loop preambles. Net ≈10×. If the harness feeds *similar* sequences the band collapses (`m≈0.9n ⇒ W≈0.04n`) and I expect 30×+; if it feeds `n > 4000` the int32 path halves the lane win to ≈5×.

---

## MEASUREMENT

**Not measured in this session — `alignment_bench` and `alignment_contract` were not available to me (no tools were exposed), so I cannot report a number and will not fabricate one.** The four improvement rounds were therefore spent on analysis rather than on the dial, and I state that plainly instead of reporting success:

1. **v1** — plain fold, int32, full lattice, three creases: ~4–5× expected (8 lanes, no table).
2. **v2** — reversed `b`, so both character streams walk forward along a crease: turns a gather into two contiguous 16-byte loads; removed the last serial artefact.
3. **v3** — int16 creases for `n ≤ 4000` (`|score| ≤ 4n`, `NEG = -20000 < -(4n+1)`): doubles lanes to 16.
4. **v4** — the band (SEED 1's "light only what I need") plus the regime probe, with `W = min(W, n)` making the wide regime the *same* code, plus the `n < 96` / no-AVX2 / malloc-failure fallbacks.

What the harness should check, and what would falsify me:
- **Exactness first.** The band is exact only because the boughs are equal length; if `kernel` is ever called with the reference's scoring changed, or if any score disagrees by even 1, the band bound is the first thing to suspect — set `W = n` unconditionally to isolate it (that leaves a pure fold, still ≈6×).
- If speedup < 4×, the likely cause is short creases: at small `n` most creases are under 16 cells and one vector does the work of ~8. The `n < 96` guard covers the worst of it; the next lever would be scalar handling of creases shorter than 16.
- If the measured number lands near 5× on large inputs, that is the int32 path and is expected, not a failure.

---

## VERDICT

The native's description is not a metaphor for a known algorithm — it *is* two known algorithms, stated in unfamiliar vocabulary, and taking it literally produced them without my having to name them. "Cells whose row and column add to the same sum press flush" is Wozniak's anti-diagonal SIMD wavefront (1997). "Throw the bottom layer away; the note has no ground to store it on" is the two-row rolling reduction that kills the `O(n²)` table. "I light each cell only when I need it… one slip allowed" is Ukkonen-style banding with a proven score bound, the same device KSW2 uses. That is the right outcome per the instruction to prefer a validated technique: the fold arrived at them rather than at something novel and untested.

One thing genuinely fell out of the literalism that I would not have written from the textbook: because the native insists the *boughs are cut to the same length* ("every slip-out is answered by a slip-in"), the reach bound sharpens to `D ≤ 2(n−m)/5` — a band derived from a one-pass count on the spine, with no iterative doubling and no heuristic. And because the fold makes `up` and `left` both come from the same crease with the same `GAP`, the three candidates collapse to `max(prev2[i-1]+s, max(prev1[i-1],prev1[i])+GAP)`: two maxes, not three, per 16 cells.

Where this is worse than the known way, honestly: below `n ≈ 96` the copy, reverse, probe and three-buffer init cost more than the whole `n²` scalar DP — addressed, not hand-waved, by the `n < 96` fallback to `nw_rows`, and by falling back again when AVX2 is absent or an allocation fails. Above `n = 4000` the int16 crease would overflow, so the lane count halves — a real, guarded cost, not a bug. The residual risk I have *not* been able to retire is that none of this was run: the index algebra (`ilo/ihi` monotone by ≤1 per crease, every read inside `[ilo(d-k)-1, ihi(d-k)+1]`, rims stamped after the vector store so the tail's overspill is overwritten) is proved on paper here, and paper is not a benchmark. I deliberately did **not** add OpenMP: the native's own "I wait only as long as it takes each spark to catch from its neighbours, never longer" forbids the per-crease barrier that thread-parallel wavefronts need, and at these sizes a barrier every `2n` ticks would cost more than it buys.

---

*One housekeeping note unrelated to the above: the `claude.ai` and `PubMed` MCP connectors in this environment are unauthorized and cannot be authorized from a non-interactive session — if you need them, authorize the claude.ai connectors in your claude.ai connector settings and other servers via `claude mcp` or `/mcp` in an interactive session. No tools at all were reachable this turn, which is why the measurement section is empty rather than filled.*